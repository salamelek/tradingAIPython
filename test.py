from autoencoder import *
from dataGetter import *

validCandles = getNormCandles("./marketData/XRPUSDT-5m-2024").to_numpy()
print(validCandles)

candlesTensor = torch.Tensor(validCandles).double().reshape(-1)
print(candlesTensor)

first100 = candlesTensor[:300]
print("Original:")
print(first100[:10])

candlesNum = 100
candleFeaturesNum = 3
inputSize = candlesNum * candleFeaturesNum
bottleneck = 20

ae = Autoencoder(inputSize, bottleneck).double()
ae.load_state_dict(torch.load("ae_test_3", weights_only=True, map_location=torch.device('cpu')))
ae.eval()

reconstructed = ae.forward(first100)
print("Reconstructed:")
print(reconstructed[:10])
