"""
The definitions of the strategies
"""

import pandas as pd


class Strategy:
    parameter_space = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        # Check the parameter space to make sure it's not empty
        if not len(self.parameter_space):
            raise Exception("No parameters defined!")

    def run(self, data: pd.DataFrame) -> float:
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

    def run(self, data: pd.DataFrame) -> float:
        pass
