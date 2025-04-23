import faiss
import numpy as np

from dataGetter import *


faiss_train_data = getCandlesFromFolders([
    "./marketData/ETHUSDT-5m-2020",
    "./marketData/ETHUSDT-5m-2021",
    "./marketData/ETHUSDT-5m-2022",
    "./marketData/ETHUSDT-5m-2023",
])

test_data = getCandles("./marketData/ETHUSDT-5m-2024")



# Create index
index = faiss.IndexFlatL2(32)  # 32 = dimension, L2 = Euclidean
index.add(X_train)             # Add training vectors to index

# Search
k = 5  # number of neighbors
distances, indices = index.search(X_query, k)

print(distances)
print(indices)
