from autoencoder import *
from dataGetter import *


autoencoder = Autoencoder(inputSize=300, bottleneckSize=10)
autoencoder.load_state_dict(torch.load("eth+xrp_300-100-50-10_5.74e-6", weights_only=True, map_location=torch.device('cpu')))
autoencoder.eval()


xrpData = getShapedData("./marketData/XRPUSDT-5m-2020-23", 100)
ethData = getShapedData("./marketData/ETHUSDT-5m-2020-24", 100)


# autoencoder pass
xrpTensor = torch.from_numpy(xrpData).float().to("cpu")
ethTensor = torch.from_numpy(ethData).float().to("cpu")

compressedXrp = autoencoder.encode(xrpTensor)
compressedEth = autoencoder.encode(ethTensor)

print("XRP:")
print(compressedXrp)

print("ETH:")
print(compressedEth)