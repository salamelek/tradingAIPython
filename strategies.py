"""
The definitions of the strategies
The run() function of the strategies will calculate the strategy signals
"""

import faiss
import numpy as np
import pandas as pd
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
    ]

    def __init__(
            self,
            k: int = 3,
            sma_window: int = 5,
            atr_window: int = 5,
            rsi_window: int = 5,
            max_pos_len: int = 24 * 12,
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
        The index has to be trained beforehand. AVOID DATA LEAKAGE AT ALL COSTS

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

        # find the neighbours
        indicators, data = self.get_norm_indicators(data)
        
        # TODO get the real and distinct neighbours, these ones are all together
        distances, indices = self.index.search(indicators, self.k)

        # Entry prices
        close = self.faiss_data["Close"].to_numpy()
        entry_prices = close[indices]

        # Future price window
        price_windows = sliding_window_view(close, window_shape=self.max_pos_len)

        # Clip indices to avoid going out of bounds
        valid_max_index = price_windows.shape[0] - 1
        clipped_indices = np.minimum(indices, valid_max_index)

        # Price windows for all neighbours
        neighbor_windows = price_windows[clipped_indices]

        # Relative returns
        relative_returns = (neighbor_windows - entry_prices[..., None]) / entry_prices[..., None]

        tp_hit = relative_returns >= self.take_profit
        sl_hit = relative_returns <= -self.stop_loss

        tp_idx = np.argmax(tp_hit, axis=2)
        sl_idx = np.argmax(sl_hit, axis=2)

        tp_valid = tp_hit.any(axis=2)
        sl_valid = sl_hit.any(axis=2)

        tp_time = np.where(tp_valid, tp_idx, self.max_pos_len + 1)
        sl_time = np.where(sl_valid, sl_idx, self.max_pos_len + 1)

        result = np.zeros_like(tp_time, dtype=np.int8)
        result[(tp_time < sl_time) & (tp_time <= self.max_pos_len)] = 1
        result[(sl_time < tp_time) & (sl_time <= self.max_pos_len)] = -1

        padded_result = np.zeros((len(data), self.k), dtype=np.int8)
        padded_result[:result.shape[0]] = result

        long = np.all(padded_result == 1, axis=1)
        short = np.all(padded_result == -1, axis=1)

        final_signals = np.zeros(len(padded_result), dtype=np.int8)
        final_signals[long] = 1
        final_signals[short] = -1

        data["strategy_signal"] = final_signals

        return data
