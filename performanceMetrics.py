import pandas as pd
from strategies import Strategy


class PerformanceMetric:
    def __init__(self, name, direction="maximize"):
        self.name = name
        self.direction = direction

    def __call__(self, S: Strategy, D: pd.DataFrame) -> float:
        raise NotImplementedError("Every subclass must implement this!")


class PnlMetric(PerformanceMetric):
    def __init__(self):
        super().__init__("PnL")

    def __call__(self, S: Strategy, D: pd.DataFrame) -> float:
        return 0


class PFMetric(PerformanceMetric):
    def __init__(self):
        super().__init__("Profit Factor")

    def __call__(self, S: Strategy, D: pd.DataFrame) -> float:
        return 0


class SharpeMetric(PerformanceMetric):
    def __init__(self):
        super().__init__("Sharpe Ratio")

    def __call__(self, S: type[Strategy], D: pd.DataFrame) -> float:
        return 0
