"""
All the necessary functions to fully backtest a strategy
"""

import optuna
import numpy as np
import pandas as pd
from strategies import Strategy
from performanceMetrics import PerformanceMetric


def optimise_strategy(P: PerformanceMetric, S: type[Strategy], D: pd.DataFrame, n_trials: int = 50) -> (dict, float):
    """
    Takes a strategy and some candles.
    Optimizes the strategy's parameters to yield the best performance.
    Uses the optuna library to find the best parameters.
    """

    def objective(trial):
        params = {}
        for p in S.parameter_space:
            if p["type"] == "int":
                params[p["name"]] = trial.suggest_int(p["name"], p["low"], p["high"])
            elif p["type"] == "float":
                params[p["name"]] = trial.suggest_float(p["name"], p["low"], p["high"])
            elif p["type"] == "bool":
                params[p["name"]] = trial.suggest_bool(p["name"], p["low"], p["high"])
            else:
                raise Exception(f"Unknown parameter type: {p['type']}")

        strategy = S(**params)
        performance = P(strategy, D)
        return performance

    study = optuna.create_study(direction=P.direction)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value


def create_permutation(candles: pd.DataFrame, seed=None) -> pd.DataFrame:
    """
    Takes some candles and permutes them
    """

    np.random.seed(seed)

    log_prices = np.log(candles[["Open", "High", "Low", "Close"]]).values
    n_bars = log_prices.shape[0]

    if n_bars == 0:
        return candles.copy()

    # Compute log returns
    r_o = log_prices[1:, 0] - log_prices[:-1, 3]  # Open[i] - Close[i-1]
    r_h = log_prices[1:, 1] - log_prices[1:, 0]  # High[i] - Open[i]
    r_l = log_prices[1:, 2] - log_prices[1:, 0]  # Low[i] - Open[i]
    r_c = log_prices[1:, 3] - log_prices[1:, 0]  # Close[i] - Open[i]

    # Generate permutations
    idx = np.arange(n_bars - 1)
    perm1 = np.random.permutation(idx)
    perm2 = np.random.permutation(idx)

    # Apply permutations
    r_o_perm = r_o[perm2]
    r_h_perm = r_h[perm1]
    r_l_perm = r_l[perm1]
    r_c_perm = r_c[perm1]

    # Vectorized computation of closes
    closes = np.empty(n_bars, dtype=np.float32)
    closes[0] = log_prices[0, 3]
    if n_bars > 1:
        cum_returns = (r_o_perm + r_c_perm).cumsum()
        closes[1:] = closes[0] + cum_returns

    # Vectorized computation of opens, highs, lows
    open_vals = np.empty(n_bars, dtype=np.float32)
    open_vals[0] = log_prices[0, 0]
    if n_bars > 1:
        open_vals[1:] = closes[:-1] + r_o_perm

    high_vals = np.empty_like(open_vals)
    high_vals[0] = log_prices[0, 1]
    if n_bars > 1:
        high_vals[1:] = open_vals[1:] + r_h_perm

    low_vals = np.empty_like(open_vals)
    low_vals[0] = log_prices[0, 2]
    if n_bars > 1:
        low_vals[1:] = open_vals[1:] + r_l_perm

    # Combine all components
    perm_bars = np.column_stack((open_vals, high_vals, low_vals, closes))

    permuted_candles = pd.DataFrame(
        np.exp(perm_bars, dtype=np.float32),
        index=candles.index,
        columns=["Open", "High", "Low", "Close"]
    )

    # Compute the log returns for each candle
    permuted_candles["log_candle_returns"] = np.log(permuted_candles["Close"]).diff()

    # Drop the NaN rows
    permuted_candles.dropna(inplace=True)

    return permuted_candles
