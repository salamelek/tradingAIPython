import pandas as pd
import numpy as np


def load_and_normalize_csv(file_paths):
    """
    Load and normalize OHLCV data from multiple CSV files.
    (Volume not included anymore)

    Normalization:
    h -> h/o - 1
    l -> l/o - 1
    c -> c/o - 1
    v -> log(v + 1)
    """
    normalized_data = []

    for file_path in file_paths:
        # Load CSV
        df = pd.read_csv(file_path)

        # Ensure the required columns exist
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            raise ValueError(f"Missing required columns in {file_path}")

        # Apply normalization
        df["h_norm"] = df["high"] / df["open"] - 1
        df["l_norm"] = df["low"] / df["open"] - 1
        df["c_norm"] = df["close"] / df["open"] - 1
        # df["v_norm"] = np.log(df["volume"] + 1)

        # Select normalized columns
        # normalized_data.append(df[["h_norm", "l_norm", "c_norm", "v_norm"]])
        normalized_data.append(df[["h_norm", "l_norm", "c_norm"]])

    # Combine all normalized data
    combined_data = pd.concat(normalized_data, ignore_index=True)
    return combined_data


def getShapedData(filePaths, candlesNum):
    """
    Reshapes the data from the files into a sliding window of candlesNum candles.
    Returns it as a 2D numpy array.

    [
        [h1, l1, c1, h2, l2, c2, ...],
        [h2, l2, c2, h3, l3, c3, ...],
        ...,
    ]
    """
    data = load_and_normalize_csv(filePaths)
    data = data.to_numpy()

    nSamples = data.shape[0] - candlesNum + 1
    nFeatures = data.shape[1] * candlesNum
    slidingWindow = np.lib.stride_tricks.sliding_window_view(data, window_shape=(candlesNum, data.shape[1]))
    reshapedWindow = slidingWindow.reshape(nSamples, nFeatures)

    return reshapedWindow
