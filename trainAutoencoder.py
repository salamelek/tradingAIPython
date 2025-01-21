from dataGetter import *
from autoencoder import *


candlesNum = 100
inputSize = candlesNum * 3
bottleneck = 20

ae = Autoencoder(inputSize, bottleneck).double()


xrpCandles = getNormCandles("./marketData/XRPUSDT-5m-2020-23").to_numpy()
ethCandles = getNormCandles("./marketData/ETHUSDT-5m-2020-24").to_numpy()
btcCandles = getNormCandles("./marketData/BTCUSDT-5m-2020-24").to_numpy()

trainCandles = np.concatenate([xrpCandles, ethCandles, btcCandles], axis=0)
validCandles = getNormCandles("./marketData/XRPUSDT-5m-2024").to_numpy()

trainAutoencoder(ae, trainCandles, validCandles, inputSize, epochs=5)
