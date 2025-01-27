from autoencoder import *
from dataGetter import *

validCandles = getNormCandles("./marketData/XRPUSDT-5m-2024").to_numpy()
candlesTensor = torch.Tensor(validCandles).double().reshape(-1)

first10 = candlesTensor[:30]

dimensions = [30, 10, 5]

ae = Autoencoder(dimensions).double()
ae.load_state_dict(torch.load("ae_miniTest_30-10-5", weights_only=True, map_location=torch.device('cpu')))
ae.eval()

reconstructed = ae.forward(first10)


print("Original:")
print(first10)
print("Reconstructed:")
print(reconstructed)
