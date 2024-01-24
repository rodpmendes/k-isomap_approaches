import numpy as np

# Create your matrix (replace this with your actual matrix)
matrix = np.array([[1, 0, 0, 2],
                   [0, 0, 0, 0],
                   [0, 3, 0, 0]])

# Calculate the sparsity ratio
sparsity = 1.0 - (np.count_nonzero(matrix) / matrix.size)

print(f"Sparsity ratio: {sparsity:.2%}")
print('teste')