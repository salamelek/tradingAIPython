import numpy as np

"""
Reshapes the data from the files into a sliding window of candlesNum candles.
Returns it as a 2D numpy array.

[
    [h1, l1, c1, h2, l2, c2, ...],
    [h2, l2, c2, h3, l3, c3, ...],
    ...,
]
"""
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

candlesNum = 2

nSamples = data.shape[0] - candlesNum + 1
nFeatures = data.shape[1] * candlesNum
slidingWindow = np.lib.stride_tricks.sliding_window_view(data, window_shape=(candlesNum, data.shape[1]))
reshapedWindow = slidingWindow.reshape(nSamples, nFeatures)

print(reshapedWindow)