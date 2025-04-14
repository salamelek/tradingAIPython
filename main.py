from strategies import *
from dataGetter import *
from visualizer import *
from backtester import *


candles = getCandles("./marketData/BTCUSDT-5m-2024")

SMA_Strategy = SMACrossoverStrategy(candles, 5, 10)
SMA_Strategy.generate_signals()

backtest_strategy(candles)
