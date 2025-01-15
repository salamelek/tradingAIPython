import pandas as pd


def simulatePosition(candles: pd.DataFrame, entryIndex: int, side: int, tp, sl, posMaxLen=-1):
    """
    Returns 1 if the position wins, -1 if loses, 0 if can't decide
    """

    candlesLen = len(candles)

    entryPrice = candles["Open"].iloc[entryIndex]
    tpPrice = entryPrice + entryPrice * tp * side
    slPrice = entryPrice - entryPrice * sl * side

    for i in range(entryIndex, candlesLen):
        if side == 1 and candles["Low"].iloc[i] <= slPrice:
            return -1

        elif side == -1 and candles["High"].iloc[i] >= slPrice:
            return -1

        elif side == 1 and candles["High"].iloc[i] >= tpPrice:
            return 1

        elif side == -1 and candles["Low"].iloc[i] <= tpPrice:
            return 1

    return 0
