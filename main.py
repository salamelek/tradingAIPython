from tradingBot import *


bot = TradingBot(
    ["./marketData/XRPUSDT-5m-2020-23"], # , "./marketData/ETHUSDT-5m-2020-24"],
    "eth+xrp_300-100-50-10_5.74e-6",
    sl=0.01,
    tp=0.022,
    minDistThreshold=1,  # e-05,
    k=3,
    posMaxLen=48,
    dimNum=10
)

candles = getDataBacktester("./marketData/XRPUSDT-5m-2024")

print("Backtesting...")
tp = 0.02
sl = 0.01
wins = 0
losses = 0
for i in range(100000):
    predictedPos, reason = bot.predict(candles[:100 + i])

    if predictedPos == 0:
        continue

    win = simulatePosition(candles, 100+i, predictedPos, tp, sl)

    if win == 1:
        wins += 1
    else:
        losses += 1

    print(f"\r[{i}] Wins: {wins}, Losses: {losses}, Profit factor: {('NaN' if losses == 0 else round((wins / losses) * (tp / sl), 2))}", end="")
