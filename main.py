"""
1) Create a strategy S in strategies.py
2) Optimise its parameters using training data D1
3) 
"""


from strategies import *
from dataGetter import *
from visualizer import *
from backtester import *


candles = getCandles("./marketData/BTCUSDT-5m-2024")

SMA_Strategy = SMACrossoverStrategy(candles, 5, 10)
SMA_Strategy.generate_signals()

backtest_strategy(candles)
