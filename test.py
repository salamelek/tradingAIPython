from pathlib import Path

from dataGetter import *

filePaths = list(Path("marketData/XRPUSDT-5m-2020-23").glob("*.csv"))  # Adjust to your directory
normData = load_and_normalize_csv(filePaths)

print(normData.shape)
