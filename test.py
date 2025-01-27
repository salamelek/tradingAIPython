from autoencoder import *
from dataGetter import *

xrpCandles = getNormCandles("./marketData/XRPUSDT-5m-2020-23").to_numpy()
ethCandles = getNormCandles("./marketData/ETHUSDT-5m-2020-24").to_numpy()
btcCandles = getNormCandles("./marketData/BTCUSDT-5m-2020-24").to_numpy()

trainCandles = np.concatenate([xrpCandles, ethCandles, btcCandles], axis=0)

# 1D tensor
inputTensor = torch.from_numpy(trainCandles).double().reshape(-1)

window = 6
step = 3
swd = SlidingWindowDataset(inputTensor, window, step)

dl = DataLoader(swd, batch_size=100)
