"""
All the necessary functions to fully backtest a strategy
"""

import pandas as pd
from strategies import Strategy
from performanceMetrics import PerformanceMetric


def optimise_strategy(P: PerformanceMetric, S: type[Strategy], D: pd.DataFrame) -> (list[int], float):
    """
    Takes a strategy and some candles.
    Optimises the strategy's parameters to yield the best performance.
    """

    # TODO look into "optuna" library (ideal for these types of optimisations)

    return [], 0


def create_permutation(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Takes some candles and permutates them
    """

    return candles



