import torch
import numpy as np


if torch.cuda.is_available():
    print("CUDA is available. GPU(s) found.")
else:
    print("CUDA is not available. Using CPU.")

# Create a large matrix (replace this with your actual matrix)
large_matrix = torch.FloatTensor(np.random.rand(4096*2, 4096*2))

# Compute eigenvalues and eigenvectors using PyTorch on the CPU
eigenvalues, eigenvectors = torch.linalg.eigh(large_matrix)

input()