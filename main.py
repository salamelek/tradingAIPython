from pathlib import Path
import torch

from dataGetter import *
from autoencoder import *


# check for GPU for faster yes
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


filePaths = list(Path("marketData/XRPUSDT-5m-2020-23").glob("*.csv"))  # Adjust to your directory
normData = load_and_normalize_csv(filePaths)

# shape: (419239, 4)
normData = normData.to_numpy()  # Convert DataFrame to NumPy
inputSize = normData.shape[1]   # TODO This seems fishy
print(f"Input size: {inputSize}")
bottleneckSize = 20

autoencoder = Autoencoder(inputSize=inputSize, bottleneckSize=bottleneckSize)
trainAutoencoder(autoencoder, normData, epochs=50, batchSize=64, device=device)
