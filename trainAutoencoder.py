from dataGetter import *
from autoencoder import *


candlesNum = 100
inputSize = candlesNum * 3
bottleneck = 5

ae = Autoencoder(inputSize, bottleneck)


trainData = getShapedData("./marketData/XRPUSDT-5m-2020-23", candlesNum)
validData = getShapedData("./marketData/XRPUSDT-5m-2024", candlesNum)

trainAutoencoder(ae, trainData, validData)
