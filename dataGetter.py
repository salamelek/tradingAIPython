import pandas as pd
import numpy as np
import torch
from pathlib import Path


def convertCandles(candles: pd.DataFrame) -> torch.Tensor:
    """
    Converts a dataFrame of candles to a normalised tensor
    The returned tensor is 1D
    """

    if not {"open", "high", "low", "close", "volume"}.issubset(candles.columns):
        raise ValueError(f"Missing required columns in given candles dataframe")

    # normalise
    candles.loc[:, "hNorm"] = np.tanh(candles["high"] / candles["open"] - 1)
    candles.loc[:, "lNorm"] = np.tanh(candles["low"] / candles["open"] - 1)
    candles.loc[:, "cNorm"] = np.tanh(candles["close"] / candles["open"] - 1)
    # candles["vNorm"] = np.log(candles["volume"] + 1)

    # normalised arr
    normalizedCandles = candles[["hNorm", "lNorm", "cNorm"]]

    # Flatten the DataFrame row-wise and convert to a 1D tensor
    flattenedTensor = torch.from_numpy(normalizedCandles.values.flatten()).float()

    return flattenedTensor


def load_data(folderName):
    """
    Returns a pd dataframe with the data from the files.
    """

    file_paths = list(Path(folderName).glob("*.csv"))

    data = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data.append(df)

    combined_data = pd.concat(data, ignore_index=True)

    return combined_data


def getDataBacktester(folderName):
    """
    Loads the data such that the backtesting.py library can read it
    """

    data = load_data(folderName)

    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data.set_index('open_time', inplace=True)
    data = data.sort_index()
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })[['Open', 'High', 'Low', 'Close', 'Volume']]

    return data


def load_and_normalize_csv(folderName):
    """
    Load and normalize OHLCV data from multiple CSV files.
    (Volume not included anymore)

    Normalization:
    h -> h/o - 1
    l -> l/o - 1
    c -> c/o - 1
    v -> log(v + 1)
    """
    file_paths = list(Path(folderName).glob("*.csv"))
    normalized_data = []

    for file_path in file_paths:
        # Load CSV
        df = pd.read_csv(file_path)

        # Ensure the required columns exist
        if not {"open", "high", "low", "close", "volume"}.issubset(df.columns):
            raise ValueError(f"Missing required columns in {file_path}")

        # Apply normalization
        df.loc[:, "h_norm"] = np.tanh(df["high"] / df["open"] - 1)
        df.loc[:, "l_norm"] = np.tanh(df["low"] / df["open"] - 1)
        df.loc[:, "c_norm"] = np.tanh(df["close"] / df["open"] - 1)
        # df.loc[:, "v_norm"] = np.log(df["volume"] + 1)

        # Select normalized columns
        # normalized_data.append(df[["h_norm", "l_norm", "c_norm", "v_norm"]])
        normalized_data.append(df[["h_norm", "l_norm", "c_norm"]])

    # Combine all normalized data
    combined_data = pd.concat(normalized_data, ignore_index=True)

    return combined_data


def getShapedData(folderName, candlesNum):
    """
    Reshapes the data from the files into a sliding window of candlesNum candles.
    Returns it as a 2D numpy array.

    [
        [h1, l1, c1, h2, l2, c2, ...],
        [h2, l2, c2, h3, l3, c3, ...],
        ...,
    ]
    """
    data = load_and_normalize_csv(folderName)
    data = data.to_numpy()

    nSamples = data.shape[0] - candlesNum + 1
    nFeatures = data.shape[1] * candlesNum
    slidingWindow = np.lib.stride_tricks.sliding_window_view(data, window_shape=(candlesNum, data.shape[1]))
    reshapedWindow = slidingWindow.reshape(nSamples, nFeatures)

    return reshapedWindow
