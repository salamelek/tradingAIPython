from pathlib import Path

from dataGetter import *
from autoencoder import *


# check for GPU for faster yes
device = "cpu"
if torch.cuda.is_available():
    print("GPU found")
    device = "cuda"
else:
    print("GPU not found, using CPU")


trainFilePaths = list(Path("marketData/XRPUSDT-5m-2020-23").glob("*.csv"))
valFilePaths = list(Path("marketData/XRPUSDT-5m-2024").glob("*.csv"))

inputCandlesNum = 100
bottleneckSize = 20

print("Getting data...")
trainData = getShapedData(trainFilePaths, inputCandlesNum)
valData = getShapedData(valFilePaths, inputCandlesNum)

autoencoder = Autoencoder(inputSize=trainData.shape[1], bottleneckSize=bottleneckSize)
# trainAutoencoder(autoencoder, trainData, valData, epochs=10, batchSize=100, lr=0.0001, device=device)

autoencoder.load_state_dict(torch.load("autoencoder100-20", weights_only=True))
autoencoder.eval()
