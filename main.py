from dataGetter import *
from tradingBot import *

# TODO re-train the autoencoder, since i added tanh to the normalisation of candles

bot = TradingBot(
    "./marketData/XRPUSDT-5m-2020-23",
    "300-100-50-5_4.29e-6",
    sl=0.01,
    tp=0.02,
    minDistThreshold=1e-05,
    k=3,
    posMaxLen=50
)

candles = getDataBacktester("./marketData/XRPUSDT-5m-2024")

print("Backtesting...")
wins = 0
losses = 0
for i in range(48000):
    predictedPos, reason = bot.predict(candles[:100 + i])

    if predictedPos == 0:
        continue

    win = simulatePosition(candles, 100+i, predictedPos, bot.tp, bot.sl)

    if win == 1:
        wins += 1
    else:
        losses += 1

    print(f"\r[{i}] Wins: {wins}, Losses: {losses}, Ratio: {('NaN' if losses == 0 else round(wins / losses, 2))}", end="")
