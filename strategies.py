import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    Abstract base class for strategies
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._precompute()

    def _precompute(self):
        """
        Vectorized indicator calculations for features and indicators
        that will get used in basically every strategy.
        Preferably don't put anything computationally heavy here :>
        """

        self.data['candle_returns'] = np.log(self.data["Close"]).diff().shift(-1)

    @abstractmethod
    def generate_features(self):
        """
        Here will go every other strategy-specific indicator.
        Example: Autoencoder outputs, RSI, ADX, ...
        """

        pass

    @abstractmethod
    def generate_signals(self) -> None:
        """
        Will calculate the strategy signal for every candle
        You better optimise the shit out of this function.
        Returns a np array of signals.
        """

        pass


class SMACrossoverStrategy(Strategy):
    """
    A simple moving average crossover strategy.
    Buy signal when short SMA crosses above long SMA.
    Sell signal when short SMA crosses below long SMA.
    """

    def __init__(self, data: pd.DataFrame, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window
        super().__init__(data)

    def generate_features(self):
        self.data['sma_short'] = self.data['Close'].rolling(window=self.short_window).mean()
        self.data['sma_long'] = self.data['Close'].rolling(window=self.long_window).mean()

    def generate_signals(self) -> None:
        self.generate_features()

        crossover_up = (self.data['sma_short'] > self.data['sma_long']) & \
                       (self.data['sma_short'].shift(1) <= self.data['sma_long'].shift(1))

        crossover_down = (self.data['sma_short'] < self.data['sma_long']) & \
                         (self.data['sma_short'].shift(1) >= self.data['sma_long'].shift(1))

        signal = np.zeros(len(self.data), dtype=np.int8)
        signal[crossover_up] = 1
        signal[crossover_down] = -1

        self.data["strategy_signal"] = signal
