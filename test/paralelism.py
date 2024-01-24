import torch
import numpy as np
import time

# Create a random 4096x4096 matrix with values between 0 and 10000
matrix = np.random.randint(0, 10001, (4096, 4096)).astype(np.float32)

# Move the matrix to the GPU if available (PyTorch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matrix_torch = torch.tensor(matrix, dtype=torch.float32).to(device)

# Measure the time taken for PyTorch eigenvalue computation
start_time_torch = time.time()
eigenvalues_torch, eigenvectors_torch = torch.linalg.eig(matrix_torch)
elapsed_time_torch = time.time() - start_time_torch

# Measure the time taken for NumPy eigenvalue computation
start_time_numpy = time.time()
eigenvalues_numpy, eigenvectors_numpy = np.linalg.eig(matrix)
elapsed_time_numpy = time.time() - start_time_numpy

# Move the PyTorch results back to the CPU if needed
eigenvalues_torch = eigenvalues_torch.cpu()
eigenvectors_torch = eigenvectors_torch.cpu()

# Print the elapsed times
print(f"PyTorch Eigenvalue computation took {elapsed_time_torch:.2f} seconds.")
print(f"NumPy Eigenvalue computation took {elapsed_time_numpy:.2f} seconds.")
