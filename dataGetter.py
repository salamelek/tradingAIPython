import pandas as pd
import numpy as np


def load_and_normalize_csv(file_paths):
    """
    Load and normalize OHLCV data from multiple CSV files.

    Normalization:
    h -> h/o - 1
    l -> l/o - 1
    c -> c/o - 1
    v -> unchanged (optional discussion)
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
        df["v_norm"] = np.log(df["volume"] + 1)

        # Select normalized columns
        normalized_data.append(df[["h_norm", "l_norm", "c_norm", "v_norm"]])

    # Combine all normalized data
    combined_data = pd.concat(normalized_data, ignore_index=True)
    return combined_data


