import pandas as pd
from strategies import Strategy


class PerformanceMetric:
    def __init__(self, name):
        self.name = name

    def __call__(self, S: Strategy, D: pd.DataFrame) -> float:
        raise NotImplementedError()


class PnlMetric(PerformanceMetric):
    def __init__(self):
        super().__init__("PnL")


class PFMetric(PerformanceMetric):
    def __init__(self):
        super().__init__("Profit Factor")

    def __call__(self, S: Strategy, D: pd.DataFrame) -> float:
        return 0
