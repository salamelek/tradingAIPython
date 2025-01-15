import faiss
import bisect

from dataGetter import *
from autoencoder import *
from positionSim import *


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


# noinspection PyArgumentList
class TradingBot:
    def __init__(self, trainDataFolder, autoencoderFile, minDistThreshold=1e-05, minIndexDistance=10, candleWindowLen=100, sl=0.01, tp=0.02, normCandlesFeatureNum=3, dimNum=5, k=3, posMaxLen=100):
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
        self.trainCandles = getDataBacktester(trainDataFolder)

        # load autoencoder
        trainData = getShapedData(trainDataFolder, candleWindowLen)
        self.autoencoder = Autoencoder(inputSize=trainData.shape[1], bottleneckSize=dimNum)
        self.autoencoder.load_state_dict(torch.load(autoencoderFile, weights_only=True, map_location=torch.device('cpu')))
        self.autoencoder.eval()

        # autoencoder pass
        trainTensor = torch.from_numpy(trainData).float().to(device)
        compressedTrain = self.autoencoder.encode(trainTensor)
        compressedNumpy = compressedTrain.cpu().numpy()

        # FAISS knn
        self.knnIndex = faiss.IndexFlatL2(dimNum)
        self.knnIndex.add(compressedNumpy)

    def getKnn(self, normCandles: torch.Tensor, k=None) -> tuple:
        """
        normCandles must be normalised and encoded
        """
        compressedQueryNumpy = normCandles.cpu().numpy()  # Convert to NumPy

        if k is None:
            k = self.k

        return self.knnIndex.search(compressedQueryNumpy, k)

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

            for d, i in zip(distances[0], indexes[0]):
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
        candleTensor = convertCandles(candles)
        # Reshape to (1, 300)
        candleTensor = candleTensor.view(1, -1)

        # encode
        encoded = self.autoencoder.encode(candleTensor)

        # get knn
        distances, indexes = self.getDifferentKnn(encoded)

        if len(indexes) < self.k:
            return 0, "No enough neighbours"

        # no good enough neighbours
        if distances[-1] > self.minDistThreshold:
            return 0, f"No good enough neighbours (worst: {distances[-1]})"

        """
        [1.6416595e-07, 3.722046e-07, 4.9566995e-07, 5.3198426e-07, 7.001139e-07]
        [177451, 383051, 112201, 222643, 95730]
        """

        # simulate nns
        candleIndexes = [(knnIndex + self.candleWindowLen - 1) for knnIndex in indexes]

        posTot = 0
        for i, candleIndex in enumerate(candleIndexes):
            longRes = simulatePosition(self.trainCandles, candleIndex + 1, 1, self.tp, self.sl, self.posMaxLen)
            shortRes = simulatePosition(self.trainCandles, candleIndex + 1, -1, self.tp, self.sl, self.posMaxLen)

            if longRes == 1:
                posTot += 1

            if shortRes == 1:
                posTot -= 1

            if abs(posTot) != i+1:
                return 0, "Disagreement"

        return int(posTot / self.k), f"{posTot} / {self.k}"
