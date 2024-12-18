from pathlib import Path

from dataGetter import *
from autoencoder import *


# check for GPU for faster yes
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


filePaths = list(Path("marketData/XRPUSDT-5m-2020-23").glob("*.csv"))  # Adjust to your directory
normData = load_and_normalize_csv(filePaths)

normData = normData.to_numpy()
inputCandlesNum = 100
bottleneckSize = 20

# reshape the array for autoencoder training
nSamples = normData.shape[0] - inputCandlesNum + 1
nFeatures = normData.shape[1] * inputCandlesNum
slidingWindow = np.lib.stride_tricks.sliding_window_view(normData, window_shape=(inputCandlesNum, normData.shape[1]))
reshapedWindow = slidingWindow.reshape(nSamples, nFeatures)

autoencoder = Autoencoder(
    inputSize=reshapedWindow.shape[1],
    bottleneckSize=bottleneckSize
)

trainAutoencoder(
    autoencoder,
    reshapedWindow,
    epochs=10,
    batchSize=100,
    lr=0.0001,
    device=device
)
