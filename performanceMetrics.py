"""
This class is intended to be used like a function.
I know I could just make a function instead, but having it like this allows inheritance
and thus it seemed better.

P = PerformanceMetric("name", direction)
performance = P(S, D)

Calling the performance "function" will run a backtest on D using S.
Thus, it needs to:
    1) Compute the return on each candle
    2) Compute every strategy's indicator
    3) Compute the strategy signals
    4) Compute the performance

Since the return on each candle can be re-used, it should be calculated elsewhere, since this function will be
called an innumerable amount of times.
"""

import pandas as pd
from strategies import Strategy


class PerformanceMetric:
    def __init__(self, name, direction="maximize"):
        self.name = name
        self.direction = direction
        self.required_columns = {"Close", "log_candle_returns"}

    def __call__(self, S: Strategy, D: pd.DataFrame) -> float:
        # Check required columns first
        if not self.required_columns.issubset(D.columns):
            missing = self.required_columns - set(D.columns)
            raise ValueError(f"Missing required columns: {missing}")

        return self._evaluate(S, D)

    def _evaluate(self, S: Strategy, D: pd.DataFrame) -> float:
        raise NotImplementedError("Subclasses must implement _evaluate().")


class PFMetric(PerformanceMetric):
    def __init__(self):
        super().__init__("Profit Factor")

    def _evaluate(self, S: Strategy, D: pd.DataFrame) -> float:
        return 0


class SharpeMetric(PerformanceMetric):
    def __init__(self):
        super().__init__("Sharpe Ratio")

    def _evaluate(self, S: Strategy, D: pd.DataFrame) -> float:
        return 0


class PnlMetric(PerformanceMetric):
    def __init__(self):
        super().__init__("PnL")

    def _evaluate(self, S: Strategy, D: pd.DataFrame) -> float:
        return 0
