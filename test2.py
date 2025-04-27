from strategies import *
from dataGetter import *


S = KnnIndicatorsStrategy()

D = getCandles(
    "./marketData/BTCUSDT-5m-2024"
)

ind, D = S.get_norm_indicators(D)

print(D)
print(ind)
print(ind.shape)
