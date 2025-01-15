import pandas as pd


def simulatePosition(candles: pd.DataFrame, entryIndex: int, side: int, posMaxLen=-1):
    """
    Returns 1 if the position wins, -1 if loses, 0 if can't decide
    """

    candlesLen = len(candles)

    for i in range(entryIndex, candlesLen):
            # get the open position price
            entryPrice = self.trainCandles["Open"].iloc[candleIndex + 1]
            print(entryPrice)


    return 0
