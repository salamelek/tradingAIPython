from pathlib import Path

from dataGetter import *


file_paths = list(Path("marketData/XRPUSDT-5m-2020-23").glob("*.csv"))  # Adjust to your directory
normalized_data = load_and_normalize_csv(file_paths)

print(normalized_data.head())
