"""
This file will take some backtested candles and plot stuff
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


candles = pd.read_csv("backtested-candles-overfit.csv")

candles["return"] = np.log(candles["Close"]).diff().shift(-1)
candles["strategy_return"] = candles["strategy_position"] * candles["return"]

ret = candles["strategy_return"]

profitFactor = ret[ret > 0].sum() / ret[ret < 0].abs().sum()
sharpeRatio = ret.mean() / ret.std()
candles["strategy_return"].cumsum().plot()
plt.show()

print(f"profit factor: {profitFactor}")
print(f"sharpe ratio: {sharpeRatio}")
