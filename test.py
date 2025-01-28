from autoencoder import *
from dataGetter import *

validCandles = getNormCandles("./marketData/XRPUSDT-5m-2024").to_numpy()
candlesTensor = torch.Tensor(validCandles).double().reshape(-1)

first10 = candlesTensor[:300]

dimensions = [300, 100, 50, 25]

ae = Autoencoder(dimensions).double()
ae.load_state_dict(torch.load("ae_300-100-50-25", weights_only=True, map_location=torch.device('cpu')))
ae.eval()

reconstructed = ae.decode(ae.encode(first10))

print("Original:")
print(first10[:10])
print("Reconstructed:")
print(reconstructed[:10])

print("Difference:")
print(first10 - reconstructed)
