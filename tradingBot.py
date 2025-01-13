from pathlib import Path
from dataGetter import *
from autoencoder import *


class TradingBot:
    def __init__(self, trainDataFolder, candleWindowLen=100, sl=0.01, tp=0.02, normCandlesFeatureNum=3, dimNum=5):
        self.normCandlesFeatureNum = normCandlesFeatureNum
        self.candleWindowLen = candleWindowLen
        self.dimNum = dimNum
        self.sl = sl
        self.tp = tp

        csvFiles = list(Path(trainDataFolder).glob("*.csv"))
        trainData = getShapedData(csvFiles, candleWindowLen)
        self.autoencoder = Autoencoder(inputSize=trainData.shape[1], bottleneckSize=dimNum)

    def predictCandles(self, candles):
        pass

    def predictNormCandles(self, normCandles):
        pass
