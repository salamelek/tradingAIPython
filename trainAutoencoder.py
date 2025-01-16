from dataGetter import *
from autoencoder import *


candlesNum = 100
inputSize = candlesNum * 3
bottleneck = 10

ae = Autoencoder(inputSize, bottleneck)


trainData1 = getShapedData("./marketData/XRPUSDT-5m-2020-23", candlesNum)
trainData2 = getShapedData("./marketData/ETHUSDT-5m-2020-24", candlesNum)
validData = getShapedData("./marketData/XRPUSDT-5m-2024", candlesNum)

trainData = np.concatenate([trainData1, trainData2], axis=0)

trainAutoencoder(ae, trainData, validData, epochs=5)
