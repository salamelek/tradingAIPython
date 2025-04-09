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

candles = getCandles("./marketData/XRPUSDT-5m-2024")


print("Backtesting...")
sl = 0.01
tp = 0.02

open_position = None  # None or dict like {'type': 1, 'entry_price': float, 'entry_index': int}
positions = np.zeros(len(candles), dtype=np.int8)

for i in range(len(candles) - 100):
    print(f"\r{i}/{len(candles)}", end="")

    current_candle = candles.iloc[100 + i]
    current_price = current_candle["Close"]

    # Check if there is an open position
    if open_position:
        # Fetch high and low of the current candle
        high = current_candle["High"]
        low = current_candle["Low"]

        entry_price = open_position["entry_price"]
        direction = open_position["type"]

        # Check for TP/SL hit
        if direction == 1:  # long
            if high >= entry_price * (1 + tp):
                open_position = None
            elif low <= entry_price * (1 - sl):
                open_position = None
            else:
                positions[100 + i] = 1

        elif direction == -1:  # short
            if low <= entry_price * (1 - tp):
                open_position = None
            elif high >= entry_price * (1 + sl):
                open_position = None
            else:
                positions[100 + i] = -1

    else:
        # No position open — check for new signal
        predictedPos, reason = bot.predict(candles.iloc[i:100 + i])  # window of 100

        if predictedPos != 0:
            open_position = {
                "type": predictedPos,  # 1 for long, -1 for short
                "entry_price": current_price,
                "entry_index": 100 + i
            }

candles["strategy_position"] = positions

print()
print(candles)
candles.to_csv("backtested-candles-overfit.csv")
