import faiss
import numpy as np

# Training data
X_train = np.random.rand(1000, 32).astype("float32")

# Query points
X_query = np.random.rand(10, 32).astype("float32")

# Create index
index = faiss.IndexFlatL2(32)  # 32 = dimension, L2 = Euclidean
index.add(X_train)             # Add training vectors to index

# Search
k = 5  # number of neighbors
distances, indices = index.search(X_query, k)

print(distances)
print(indices)
