import pandas as pd
import numpy as np
from pathlib import Path


def normaliseCandles(candles: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises the given candles

    Normalization:
    h -> h/o - 1
    l -> l/o - 1
    c -> c/o - 1
    v -> log(v + 1)
    """

    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(candles.columns):
        raise ValueError(f"Missing required columns in given candles dataframe")

    # normalise
    candles.loc[:, "hNorm"] = np.tanh(candles["High"] / candles["Open"] - 1)
    candles.loc[:, "lNorm"] = np.tanh(candles["Low"] / candles["Open"] - 1)
    candles.loc[:, "cNorm"] = np.tanh(candles["Close"] / candles["Open"] - 1)
    # candles["vNorm"] = np.log(candles["Volume"] + 1)

    # normalised arr
    return candles[["hNorm", "lNorm", "cNorm"]]


def getCandles(folderName: str) -> pd.DataFrame:
    """
    Returns a pd dataframe with the data from the files.
    It is nicely formatted so that the backtesting.py library can use it
    """

    file_paths = list(Path(folderName).glob("*.csv"))

    dfs = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

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


def getNormCandles(folderName: str) -> pd.DataFrame:
    return normaliseCandles(getCandles(folderName))
