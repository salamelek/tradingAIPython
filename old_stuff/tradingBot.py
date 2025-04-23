import faiss
import bisect

import torch

from dataGetter import *
from autoencoder import *
from positionSim import *


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


class TradingBotKNN:
    def __init__(self, trainDataFolders: list, autoencoderFile: str, dimensions: list, minDistThreshold=5e-06, minIndexDistance=10, candleWindowLen=100, sl=0.01, tp=0.02, normCandlesFeatureNum=3, dimNum=5, k=3, posMaxLen=100):
        """
        trainDataFolders: list folder names
        """

        self.normCandlesFeatureNum = normCandlesFeatureNum
        self.minDistThreshold = minDistThreshold
        self.minIndexDistance = minIndexDistance
        self.candleWindowLen = candleWindowLen
        self.posMaxLen = posMaxLen
        self.dimNum = dimNum
        self.sl = sl
        self.tp = tp
        self.k = k

        # load historical kline data
        self.candlesList = []
        self.normCandlesList = []
        for trainDataFolder in trainDataFolders:
            candles = getCandles(trainDataFolder)
            self.candlesList.append(candles)
            self.normCandlesList.append(normaliseCandles(candles))

        # load autoencoder
        self.autoencoder = Autoencoder(dimensions).double()
        self.autoencoder.load_state_dict(torch.load(autoencoderFile, weights_only=True, map_location=torch.device('cpu')))
        self.autoencoder.eval()

        # autoencoder pass
        encodedCandlesList = []
        for normCandles in self.normCandlesList:
            normCandlesTensor = torch.from_numpy(normCandles.to_numpy()).reshape(-1).to(device)
            windows = normCandlesTensor.unfold(0, self.normCandlesFeatureNum * self.candleWindowLen, self.normCandlesFeatureNum)
            self.autoencoder.eval()

            # we have to use batches because it cant fit all in memory :p
            batchSize = 100000
            encodedBatches = []
            for i in range(0, windows.size(0), batchSize):
                batch = windows[i:i + batchSize]
                encoded_batch = self.autoencoder.encode(batch)
                encodedBatches.append(encoded_batch)

            # Concatenate all the encoded batches
            encodedCandlesList.append(torch.cat(encodedBatches, dim=0).numpy())

        # FAISS knn
        # with autoencoder
        # self.knnIndex = faiss.IndexFlatL2(dimNum) # euclidean distance
        self.knnIndex = faiss.IndexFlatIP(dimNum)   # cosine similarity

        for encodedCandles in encodedCandlesList:
            # faiss only supports float32 apparently :p
            self.knnIndex.add(encodedCandles.astype(np.float32))

    def getKnn(self, normCandles: torch.Tensor, k=None) -> tuple:
        """
        normCandles must be normalised and encoded
        """
        compressedQueryNumpy = normCandles.cpu().numpy().astype(np.float32)  # Convert to NumPy

        if k is None:
            k = self.k

        dist, indx = self.knnIndex.search(compressedQueryNumpy, k)

        return dist[0], indx[0]

    def getDifferentKnn(self, normCandles: torch.Tensor) -> tuple:
        """
        Returns the k nearest neighbors of the given tensor, ensuring the indices are
        at least self.minIndexDistance apart.

        :param normCandles: Normalized encoded input tensor
        :return: A tuple (selected_distances, selected_indices), where:
                 - selected_distances: List of distances for the selected neighbors.
                 - selected_indices: List of indices for the selected neighbors.
        """
        selected_distances = []
        selected_indices = []
        seen_indices = []
        mult = 1
        max_mult = 10

        while len(selected_indices) < self.k:
            if mult > max_mult:
                print(
                    f"Unable to find {self.k} distinct neighbors with the given constraints. "
                    f"Consider relaxing minIndexDistance or increasing the dataset size."
                )
                return [], []

            # Request more candidates with an expanded pool
            distances, indexes = self.getKnn(normCandles, self.k * 5 * mult)

            for d, i in zip(distances, indexes):
                # Use binary search for efficient index checking
                pos = bisect.bisect_left(seen_indices, i)
                if (pos == 0 or i - seen_indices[pos - 1] >= self.minIndexDistance) and \
                        (pos == len(seen_indices) or seen_indices[pos] - i >= self.minIndexDistance):
                    selected_distances.append(d)
                    selected_indices.append(i)
                    bisect.insort(seen_indices, i)

                    # Break early if we've collected enough neighbors
                    if len(selected_indices) >= self.k:
                        return selected_distances, selected_indices

            # Increase the candidate pool size
            mult += 1

        return selected_distances, selected_indices

    def predict(self, candles: pd.DataFrame) -> (int, str):
        """
        Given a list of candles, returns the predicted market direction
        Also returns a string "Reason" That explains why the return was 0
        """

        candlesLen = len(candles)

        if candlesLen < self.candleWindowLen:
            return 0, "Too few candles to make a prediction."

        # normalise candles and convert them to a tensor
        candles = candles.iloc[-self.candleWindowLen:].copy()
        normCandles = normaliseCandles(candles).to_numpy()
        normCandlesTensor = torch.from_numpy(normCandles).reshape(1, -1)

        # encode
        encoded = self.autoencoder.encode(normCandlesTensor)

        # get knn
        distances, indexes = self.getDifferentKnn(encoded)

        if len(indexes) < self.k:
            return 0, "No enough neighbours"

        # no good enough neighbours
        # Euclidean distance (high value for ew, 0 for good)
        """
        if distances[-1] > self.minDistThreshold:
            return 0, f"No good enough neighbours (worst: {distances[-1]})"
        """

        # cosine similarity (-1 for ew, 1 for good)
        if distances[-1] < self.minDistThreshold:
            return 0, f"No good enough neighbours (worst: {distances[-1]})"

        # simulate nns
        candleIndexes = [(knnIndex + self.candleWindowLen - 1) for knnIndex in indexes]

        posTot = 0
        for i, candleIndex in enumerate(candleIndexes):
            # select the correct candles
            trainCandlesIndex = 0
            lenSum = 0
            for j in range(len(self.candlesList)):
                currLen = len(self.candlesList[j])

                if candleIndex >= currLen and j < len(self.candlesList) - 1:
                    lenSum += currLen
                    candleIndex -= currLen  # Adjust candleIndex for the next dataset
                    continue

                trainCandlesIndex = j
                break  # Found the correct dataset

            longRes = simulatePosition(self.candlesList[trainCandlesIndex], candleIndex + 1, 1, self.tp, self.sl, self.posMaxLen)
            shortRes = simulatePosition(self.candlesList[trainCandlesIndex], candleIndex + 1, -1, self.tp, self.sl, self.posMaxLen)
            if longRes == 1:
                posTot += 1

            if shortRes == 1:
                posTot -= 1

            if abs(posTot) != i+1:
                return 0, "Disagreement"

        return int(posTot / self.k), ""


class TradingBotSMA:
    def __init__(self, p1: int, p2: int):
        self.p1 = p1
        self.p2 = p2
        self.prev_state = None  # Track previous relationship between SMAs

    def predict(self, candles: pd.DataFrame) -> (int, str):
        if len(candles) < max(self.p1, self.p2) + 1:
            return 0, "Not enough data"

        close = candles["Close"]
        sma1 = close.rolling(window=self.p1).mean()
        sma2 = close.rolling(window=self.p2).mean()

        # Current relationship
        current_state = 'above' if sma1.iloc[-1] > sma2.iloc[-1] else 'below'

        # Only generate signal if crossover JUST happened
        signal = 0
        reason = "No crossover"

        if self.prev_state is not None:
            if self.prev_state == 'below' and current_state == 'above':
                signal = 1
                reason = "SMA crossover up (buy signal)"
            elif self.prev_state == 'above' and current_state == 'below':
                signal = -1
                reason = "SMA crossover down (sell signal)"

        # Update previous state for next call
        self.prev_state = current_state

        return signal, reason


class TradingBotIndicatorKnn:
    def __init__(self):
        pass

    def calculateIndicators(self):
        pass

    def predict(self, candles: pd.DataFrame) -> (int, str):


        return 0, ""
