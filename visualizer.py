import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class CandlesVisualizer:
    def __init__(self, candles: pd.DataFrame):
        self.candles = candles

    def plot(self):
        returns = self.candles["strategy_return"].dropna()
        cum_returns = returns.cumsum()

        profit_factor = returns[returns > 0].sum() / returns[returns < 0].abs().sum()
        sharpe_ratio = returns.mean() / returns.std()
        total_return = cum_returns.iloc[-1]

        plt.plot(cum_returns)
        plt.show()
