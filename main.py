from tradingBot import *


# e-01 no shuffle: [10271] Wins: 8, Losses: 23, Profit factor: 0.7
# e-02 no shuffle: [9622] Wins: 12, Losses: 19, Profit factor: 1.26
# e-03 no shuffle: [8904] Wins: 8,  Losses: 10, Profit factor: 1.6
#                  [53081] Wins: 386, Losses: 646, Profit factor: 1.2
# e-04 no shuffle: [3339] Wins: 6,  Losses: 8,  Profit factor: 1.5

# e-03 shuffle:    [9735] Wins: 4,  Losses: 3,  Profit factor: 2.67
#                  [53082] Wins: 117, Losses: 236, Profit factor: 0.99


bot = TradingBot(
    ["./marketData/XRPUSDT-5m-2020-23", "./marketData/ETHUSDT-5m-2020-24", "./marketData/BTCUSDT-5m-2020-24"],
    "ae_noShuffle_beta-03",
    [300, 100, 50, 25],
    sl=0.01,
    tp=0.022,
    minDistThreshold=0.12,
    k=2,
    posMaxLen=48,
    dimNum=25
)

candles = getCandles("./marketData/XRPUSDT-5m-2024")

print("Backtesting...")
tp = 0.02
sl = 0.01
wins = 0
losses = 0
for i in range(53083):
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
