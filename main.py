from tradingBot import *


bot = TradingBot(
    ["./marketData/XRPUSDT-5m-2020-23", "./marketData/ETHUSDT-5m-2020-24", "./marketData/BTCUSDT-5m-2020-24", "./marketData/XRPUSDT-5m-2024"],
    "ae_noShuffle_beta-03",
    [300, 100, 50, 25],
    sl=0.01,
    tp=0.02,
    minDistThreshold=0.12,
    k=2,
    posMaxLen=24,
    dimNum=25
)

candles = getCandles("./marketData/DOGEUSDT-5m-2024")

print("Backtesting...")
tp = 0.02
sl = 0.01
wins = 0
losses = 0
for i in range(50000):
    predictedPos, reason = bot.predict(candles[:100 + i])

    if predictedPos == 0:
        # print(f"\r{reason}", end="")
        continue

    win = simulatePosition(candles, 100+i, predictedPos, tp, sl)

    if win == 1:
        wins += 1
    else:
        losses += 1

    print(f"\r[{i}] Wins: {wins}, Losses: {losses}, Profit factor: {('NaN' if losses == 0 else round((wins / losses) * (tp / sl), 2))}", end="")
