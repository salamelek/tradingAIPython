from strategies import *
from dataGetter import *
from backtester import *
from performanceMetrics import *

from rich.progress import track
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_indicator_scatter(indicators: np.ndarray):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot()

    sma_vals = indicators[:, 0]
    atr_vals = indicators[:, 1]
    rsi_vals = indicators[:, 2]

    ax.scatter(sma_vals, atr_vals, rsi_vals, c=rsi_vals, cmap='viridis', s=20)

    ax.set_xlabel('SMA')
    ax.set_ylabel('ATR')
    ax.set_zlabel('RSI')
    ax.set_title('3D Scatter Plot of SMA, ATR, and RSI')

    plt.show()


# Get the train and validation candles
D1 = getCandlesFromFolders([
    "./marketData/test",
])


S = KnnIndicatorsStrategy()
ind = S.get_norm_indicators(D1)
plot_indicator_scatter(ind)
