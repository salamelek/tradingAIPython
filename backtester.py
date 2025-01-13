import backtrader as bt
import numpy as np


class MyStrategy(bt.Strategy):
    params = (
        ("nCandles", 100),  # Number of candles to normalize
    )

    def __init__(self):
        self.candles = []  # To store the last n candles for normalization

    def normalize_candle(self, high, low, close, open_):
        """
        Normalize a single candle.
        """

        return np.tanh([
            high / open_ - 1,
            low / open_ - 1,
            close / open_ - 1,
        ])

    def next(self):
        # Collect the last n candles
        self.candles.append({
            "open": self.datas[0].open[0],
            "high": self.datas[0].high[0],
            "low": self.datas[0].low[0],
            "close": self.datas[0].close[0],
        })

        if len(self.candles) < self.params.nCandles:
            return  # Wait until we have enough candles

        # Normalize the last n candles
        normalized_candles = [
            self.normalize_candle(c["high"], c["low"], c["close"], c["open"])
            for c in self.candles
        ]

        # Call the AI trading bot with normalized candles
        signal, stop_loss, take_profit = my_ai_trading_bot(normalized_candles)

        # Execute the bot's decision
        if signal == 1:
            self.buy()
        elif signal == -1:
            self.sell()

        # Remove the oldest candle to keep the list size to nCandles
        self.candles.pop(0)


# Example `my_ai_trading_bot` function (replace with your implementation)
def my_ai_trading_bot(normalized_candles):
    """
    Simulate AI trading bot decision.
    Replace this with your actual bot's logic.
    """
    # Example logic: randomly decide to buy, sell, or hold
    import random
    signal = random.choice(["buy", "sell", "hold"])
    stop_loss = 0.95  # Example stop-loss level
    take_profit = 1.05  # Example take-profit level
    return signal, stop_loss, take_profit


# Load your data into a Pandas DataFrame
import pandas as pd

# Replace this with your actual historical data
data = pd.DataFrame({
    "datetime": pd.date_range(start="2023-01-01", periods=100, freq="D"),
    "open": np.random.random(100) + 1,
    "high": np.random.random(100) + 1.5,
    "low": np.random.random(100) + 0.5,
    "close": np.random.random(100) + 1,
    "volume": np.random.randint(100, 1000, 100),
})
data.set_index("datetime", inplace=True)

# Initialize Backtrader
bt_data = bt.feeds.PandasData(dataname=data)

# Setup Cerebro
cerebro = bt.Cerebro()
cerebro.addstrategy(MyStrategy, nCandles=10)  # Use the strategy with 10 candles
cerebro.adddata(bt_data)

# Run the backtest
cerebro.run()
cerebro.plot()
