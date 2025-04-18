"""
The definitions of the strategies
The run() function of the strategies will calculate the strategy signals
"""

import numpy as np
import pandas as pd


class Strategy:
    parameter_space = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        # Check the parameter space to make sure it's not empty
        if not len(self.parameter_space):
            raise Exception("No parameters defined!")

    def generate_signals(self, data: pd.DataFrame) -> None:
        """
        Generate the strategy signals and put them in the data frame
        """
        raise NotImplementedError("All subclasses must implement this!")


class SMACrossoverStrategy(Strategy):
    parameter_space = [
        {"name": "fastSMA", "type": "int", "low": 5, "high": 50},
        {"name": "slowSMA", "type": "int", "low": 10, "high": 200}
    ]

    def __init__(self, fastSMA: int = 5, slowSMA: int = 10):
        super().__init__(fastSMA=fastSMA, slowSMA=slowSMA)
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
