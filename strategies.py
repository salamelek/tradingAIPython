"""
The definitions of the strategies
The run() function of the strategies will calculate the strategy signals
"""

import faiss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from numpy.lib.stride_tricks import sliding_window_view


class Strategy:
    parameter_space = []

    def __init__(self, *args, **kwargs) -> None:
        # Check the parameter space to make sure it's not empty
        if not len(self.parameter_space):
            raise Exception("No parameters defined!")

        self.args = args
        self.kwargs = kwargs

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate the strategy signals and put them in the data frame
        """
        raise NotImplementedError("All subclasses must implement this!")


class SMACrossoverStrategy(Strategy):
    parameter_space = [
        {"name": "fastSMA", "type": "int", "low": 5, "high": 50},
        {"name": "slowSMA", "type": "int", "low": 10, "high": 100}
    ]

    def __init__(self, fastSMA: int = 5, slowSMA: int = 10):
        super().__init__()
        self.fastSMA = fastSMA
        self.slowSMA = slowSMA

    def generate_signals(self, data: pd.DataFrame) -> None:
        close = data["Close"]
        sma1 = close.rolling(window=self.fastSMA).mean()
        sma2 = close.rolling(window=self.slowSMA).mean()

        cross_above = (sma1 > sma2) & (sma1.shift(1) <= sma2.shift(1))
        cross_below = (sma1 < sma2) & (sma1.shift(1) >= sma2.shift(1))

        signal = np.select(
            [cross_above, cross_below],
            [1, -1],
            default=0
        )

        data["strategy_signal"] = signal


class KnnIndicatorsStrategy(Strategy):
    parameter_space = [
        # {"name": "k", "type": "int", "low": 1, "high": 5},    # The problem here is that we have to create a new index every time
        {"name": "sma_window", "type": "int", "low": 5, "high": 100},
        {"name": "atr_window", "type": "int", "low": 5, "high": 30},
        {"name": "rsi_window", "type": "int", "low": 5, "high": 30},
        {"name": "max_pos_len", "type": "int", "low": 5, "high": 500},
        {"name": "tp", "type": "float", "low": 0, "high": 0.1},
        {"name": "sl", "type": "float", "low": 0, "high": 0.1},
    ]

    def __init__(
            self,
            sma_window: int = 5,
            atr_window: int = 5,
            rsi_window: int = 5,
            max_pos_len: int = 24 * 12,
            
            k: int = 3,
            tp: float = 0.01,
            sl: float = 0.01,
            index: faiss.IndexFlatL2 = None,
            faiss_data: pd.DataFrame = None,
    ):
        super().__init__()
        self.index = index
        self.k = k
        self.sma_window = sma_window
        self.atr_window = atr_window
        self.rsi_window = rsi_window
        self.max_pos_len = max_pos_len
        self.take_profit = tp
        self.stop_loss = sl
        self.faiss_data = faiss_data

    def get_norm_indicators(self, data: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        sma = close.rolling(window=self.sma_window).mean()
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_window).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        data = data.copy()
        data["sma"] = sma
        data["atr"] = atr
        data["rsi"] = rsi

        data_clean = data.dropna()
        indicators = data_clean[["sma", "atr", "rsi"]].to_numpy()

        # Clip ATR outliers (99th percentile)
        q_high = np.quantile(indicators[:, 1], 0.99)
        indicators[:, 1] = np.clip(indicators[:, 1], None, q_high)

        # Normalize with RobustScaler
        scaler = RobustScaler()
        indicators_scaled = scaler.fit_transform(indicators)

        return indicators_scaled, data_clean

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This function will generate the strategy signals using a faiss index
        The index has to be trained beforehand with self.faiss_data. AVOID DATA LEAKAGE AT ALL COSTS

        Once you have the distances and indices, you can use them to calculate the strategy signals:
	        1) get the entry prices
	        2) create a 2D matrix for the entry prices to the posMaxLen
	        3) get the price window for all neighbours
	        4) simulate everything using vectorisation
	        5) profit
        """

        if self.index is None:
            raise Exception("The faiss index must be set!")
        if self.faiss_data is None:
            raise Exception("The faiss data must be set!")

        indicators, data = self.get_norm_indicators(data)
        distances, indices = self.index.search(indicators, self.k)

        close_prices = self.faiss_data["Close"].to_numpy()
        high_prices = self.faiss_data["High"].to_numpy()
        low_prices = self.faiss_data["Low"].to_numpy()

        entry_prices = close_prices[indices]

        # Create sliding windows for high and low prices
        high_windows = sliding_window_view(high_prices, window_shape=self.max_pos_len)
        low_windows = sliding_window_view(low_prices, window_shape=self.max_pos_len)

        valid_max_index = high_windows.shape[0] - 1
        # Adjust indices to start the window from the next bar after entry
        adjusted_indices = indices + 1
        clipped_indices = np.clip(adjusted_indices, 0, valid_max_index)

        neighbor_highs = high_windows[clipped_indices]
        neighbor_lows = low_windows[clipped_indices]

        # Long trade simulations
        long_tp_hit = neighbor_highs >= entry_prices[..., None] * (1 + self.take_profit)
        long_sl_hit = neighbor_lows <= entry_prices[..., None] * (1 - self.stop_loss)

        # Short trade simulations
        short_tp_hit = neighbor_lows <= entry_prices[..., None] * (1 - self.take_profit)
        short_sl_hit = neighbor_highs >= entry_prices[..., None] * (1 + self.stop_loss)

        # Determine first occurrence of TP/SL
        long_tp_idx = np.argmax(long_tp_hit, axis=2)
        long_sl_idx = np.argmax(long_sl_hit, axis=2)
        short_tp_idx = np.argmax(short_tp_hit, axis=2)
        short_sl_idx = np.argmax(short_sl_hit, axis=2)

        # Check if any trigger occurred
        long_tp_valid = long_tp_hit.any(axis=2)
        long_sl_valid = long_sl_hit.any(axis=2)
        short_tp_valid = short_tp_hit.any(axis=2)
        short_sl_valid = short_sl_hit.any(axis=2)

        # Assign time with max_pos_len +1 if not triggered
        long_tp_time = np.where(long_tp_valid, long_tp_idx, self.max_pos_len + 1)
        long_sl_time = np.where(long_sl_valid, long_sl_idx, self.max_pos_len + 1)
        short_tp_time = np.where(short_tp_valid, short_tp_idx, self.max_pos_len + 1)
        short_sl_time = np.where(short_sl_valid, short_sl_idx, self.max_pos_len + 1)

        # Determine which condition was met first
        long_win = (long_tp_time < long_sl_time) & (long_tp_time <= self.max_pos_len)
        long_loss = (long_sl_time < long_tp_time) & (long_sl_time <= self.max_pos_len)
        short_win = (short_tp_time < short_sl_time) & (short_tp_time <= self.max_pos_len)
        short_loss = (short_sl_time < short_tp_time) & (short_sl_time <= self.max_pos_len)

        # Generate signals per neighbor
        signal = np.zeros_like(long_win, dtype=np.int8)
        signal[long_win] = 1
        signal[long_loss] = 0
        signal[short_win] = -1
        signal[short_loss] = 0

        # Resolve conflicts where both long and short could trigger
        conflict = (long_win & short_win) | (long_win & short_loss) | (long_loss & short_win)
        signal[conflict] = 0

        # Apply consensus rule: all neighbors must agree
        long_consensus = np.all(signal == 1, axis=1)
        short_consensus = np.all(signal == -1, axis=1)

        final_signals = np.zeros(len(data), dtype=np.int8)
        final_signals[long_consensus] = 1
        final_signals[short_consensus] = -1

        data["strategy_signal"] = final_signals
        return data
