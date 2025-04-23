"""
The definitions of the strategies
The run() function of the strategies will calculate the strategy signals
"""

import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class Strategy:
    parameter_space = []

    def __init__(self, *args, **kwargs) -> None:
        # Check the parameter space to make sure it's not empty
        if not len(self.parameter_space):
            raise Exception("No parameters defined!")

        self.args = args
        self.kwargs = kwargs

    def generate_signals(self, data: pd.DataFrame) -> None:
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
        {"name": "k", "type": "int", "low": 1, "high": 5},
        {"name": "sma_window", "type": "int", "low": 5, "high": 100},
        {"name": "atr_window", "type": "int", "low": 5, "high": 30},
        {"name": "rsi_window", "type": "int", "low": 5, "high": 30},
    ]

    def __init__(self, index: faiss.IndexFlatL2, k: int = 1, sma_window: int = 5, atr_window: int = 5, rsi_window: int = 5):
        super().__init__()
        self.index = index
        self.k = k
        self.sma_window = sma_window
        self.atr_window = atr_window
        self.rsi_window = rsi_window

    def get_norm_indicators(self, data: pd.DataFrame) -> np.ndarray:
        close = data["Close"]
        high = data["High"]
        low = data["Low"]

        # sma
        sma = close.rolling(window=self.sma_window).mean()

        # atr
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_window).mean()

        # rsi
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Stack the features into a NumPy array
        indicators = np.stack([sma, atr, rsi], axis=1)
        indicators = indicators[~np.isnan(indicators).any(axis=1)]

        # Remove outliers
        q_high = np.quantile(indicators[:, 1], 0.99)  # 99th percentile of ATR
        indicators[:, 1] = np.clip(indicators[:, 1], None, q_high)  # Clip upper outliers

        # Normalise using RobustScaler
        scaler = RobustScaler()
        indicators_scaled = scaler.fit_transform(indicators)

        return indicators_scaled

    def generate_signals(self, data: pd.DataFrame) -> None:
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

        distances, indices = self.index.search(self.get_norm_indicators(data), self.k)


