import pandas as pd
from strategies import Strategy


class PerformanceMetric:
    def __call__(self, S: Strategy, D: pd.DataFrame) -> float:
        raise NotImplementedError()


class PnlMetric(PerformanceMetric):
    def __init__(self):
        pass


class PFMetric(PerformanceMetric):
    def __init__(self):
        pass
