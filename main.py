from pathlib import Path
import faiss

from dataGetter import *
from autoencoder import *


# check for GPU for faster yes
device = "cpu"
if torch.cuda.is_available():
    print("GPU found")
    device = "cuda"
else:
    print("GPU not found, using CPU")

# get the train and validation data
trainFilePaths = list(Path("marketData/XRPUSDT-5m-2020-23").glob("*.csv"))
valFilePaths = list(Path("marketData/XRPUSDT-5m-2024").glob("*.csv"))

# some autoencoder parameters
inputCandlesNum = 100
normCandleFeaturesNum = 3
bottleneckSize = 5

# get the data
print("Getting data...")
trainData = getShapedData(trainFilePaths, inputCandlesNum)
valData = getShapedData(valFilePaths, inputCandlesNum)

# define the autoencoder
autoencoder = Autoencoder(inputSize=trainData.shape[1], bottleneckSize=bottleneckSize)

# train the autoencoder
# trainAutoencoder(autoencoder, trainData, valData, epochs=10, batchSize=100, lr=0.0001, device=device)

# load the autoencoder
print("Loading autoencoder...")
autoencoder.load_state_dict(torch.load("300-100-50-5_4.33e-6", weights_only=True, map_location=torch.device('cpu')))
autoencoder.eval()

# encode the training data
print("Encoding training data...")
trainTensor = torch.from_numpy(trainData).float().to(device)
compressedTrain = autoencoder.predict(trainTensor)
compressedNumpy = compressedTrain.cpu().numpy()

print(compressedNumpy.shape)

# FAISS knn
print("Creating FAISS index...")
index = faiss.IndexFlatL2(bottleneckSize)
index.add(compressedNumpy)



# Example query tensor
queryTensor = torch.randn(1, trainData.shape[1]).float().to(device)  # Example query with random data

# Encode the query using the autoencoder
compressedQuery = autoencoder.predict(queryTensor)  # Compressed query
compressedQueryNumpy = compressedQuery.cpu().numpy()  # Convert to NumPy

# Perform the KNN search
k = 5  # Number of nearest neighbors to find
distances, indices = index.search(compressedQueryNumpy, k)

# Output results
print(f"Indices of the {k} nearest neighbors: {indices}")
print(f"Distances to the {k} nearest neighbors: {distances}")
