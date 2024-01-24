import numpy as np
import torch
import time

from scipy.linalg import eigh  # eigh for symmetric matrices
from scipy.sparse.linalg import eigs

matrix_size = 4096*2

large_matrix = np.random.rand(matrix_size, matrix_size)
torch_cpu_matrix = torch.rand(matrix_size, matrix_size)

if torch.cuda.is_available():
    torch_gpu_matrix = torch.rand(matrix_size, matrix_size).to("cuda")
else:
    print("No GPU available, skipping GPU computation.")

start_time = time.time()
torch_cpu_eigenvalues, torch_cpu_eigenvectors = torch.linalg.eig(torch_cpu_matrix)
torch.cuda.synchronize()
torch_cpu_time = time.time() - start_time
print('torch cpu time eig', torch_cpu_time)

start_time = time.time()
torch_cpu_eigenvalues, torch_cpu_eigenvectors = torch.linalg.eigh(torch_cpu_matrix)
torch.cuda.synchronize()
torch_cpu_time = time.time() - start_time
print('torch cpu time', torch_cpu_time)


if torch.cuda.is_available():
    t1 = time.time()
    torch.linalg.eig(torch_gpu_matrix)
    torch.cuda.synchronize()
    t2= time.time()
    gpu_time = t2-t1
    print('torch gpu time eig', gpu_time)
    
    t1 = time.time()
    torch.linalg.eigh(torch_gpu_matrix)
    torch.cuda.synchronize()
    t2= time.time()
    gpu_time = t2-t1
    print('torch gpu time', gpu_time)
else:
    print("No GPU available, skipping GPU computation.")

# Krylov Subspace Methods using Arnoldi and Lanczos Methods
# Compute the eigenvalues and eigenvectors using Arnoldi method
start_time = time.time()
num_eigenvalues_to_compute = 10  # Number of eigenvalues/eigenvectors to compute
eigenvalues, eigenvectors = eigs(large_matrix, k=num_eigenvalues_to_compute, which='LM')
lm_time = time.time() - start_time
print('lm time', lm_time)


start_time = time.time()
symmetric_matrix = np.dot(large_matrix, large_matrix.T)  # Ensure it's symmetric
eigenvalues, eigenvectors = eigh(symmetric_matrix)
scipy_time = time.time() - start_time
print('scipy time', scipy_time)

start_time = time.time()
numpy_eigenvalues, numpy_eigenvectors = np.linalg.eig(large_matrix)
numpy_time = time.time() - start_time
print('numpy time', numpy_time)

input()