import faiss
import numpy as np


# if dataset is very large, use clusters with IndexIVFFlat


# Example data: n-dimensional vectors (n samples of d dimensions)
n, d = 1000, 128  # 1000 samples, 128 dimensions
np.random.seed(42)
data = np.random.random((n, d)).astype('float32')

# Create the FAISS index
index = faiss.IndexFlatL2(d)  # L2 (Euclidean) distance

# Add data to the index
index.add(data)
print(f"Number of vectors in the index: {index.ntotal}")

# Query with k-nearest neighbors
k = 5  # Number of neighbors to retrieve
query = np.random.random((1, d)).astype('float32')  # Single query vector

# Perform the search
distances, indices = index.search(query, k)
print("Query vector:", query)
print(f"Top {k} nearest neighbors (indices):", indices)
print(f"Distances to the neighbors:", distances)
