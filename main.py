from dataGetter import *
from tradingBot import *

# TODO re-train the autoencoder, since i added tanh to the normalisation of candles

bot = TradingBot(
    "./marketData/XRPUSDT-5m-2020-23",
    "300-100-50-5_4.33e-6",
)

candles = getDataBacktester("./marketData/XRPUSDT-5m-2024")

print("Backtesting...")
wins = 0
losses = 0
for i in range(10000):
    predictedPos, reason = bot.predict(candles[:100 + i])

    if predictedPos == 0:
        continue

    win = simulatePosition(candles, 100+i, predictedPos, bot.tp, bot.sl)

    if win == 1:
        wins += 1
    else:
        losses += 1

    print(f"\r[{i}] Wins: {wins}, Losses: {losses}, Ratio: {(wins / losses):.2f}", end="")
