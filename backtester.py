from backtesting import Backtest, Strategy
from pathlib import Path
import pandas as pd
from dataGetter import *


def bot_predict(input_data):
    close = input_data["close"][-1]
    perc = close * 0.01
    return 1, close + perc, close - perc


class BotStrategy(Strategy):
    def init(self):
        self.bot_input = []

    def next(self):
        if len(self.data.Close) < 100:
            return

        last_100_candles = {
            'open': self.data.Open[-100:],
            'high': self.data.High[-100:],
            'low': self.data.Low[-100:],
            'close': self.data.Close[-100:],
        }

        side, tp, sl = bot_predict(last_100_candles)

        if side == 0:
            return

        # Execute the trade based on the bot's signal
        if side == 1:
            self.buy(size=10, sl=sl, tp=tp)
        elif side == -1:
            self.sell(size=10, sl=sl, tp=tp)


data = getCandles(list(Path("marketData/XRPUSDT-5m-2024").glob("*.csv")))

bt = Backtest(data, BotStrategy, cash=10_000, commission=.002)
stats = bt.run()
print(stats)
