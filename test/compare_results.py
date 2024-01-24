import numpy as np
import torch

cov_matrix_np = [[3, 9, 7, 5],
                 [1, 1, 0, 8],
                 [6, 7, 0, 9],
                 [8, 2, 1, 1]]

eigenvalues_np, eigenvectors_np = np.linalg.eigh(cov_matrix_np)

cov_matrix_torch = torch.tensor(cov_matrix_np, dtype=torch.float32)

eigenvalues_torch, eigenvectors_torch = torch.linalg.eigh(cov_matrix_torch)

eigenvalues_torch = eigenvalues_torch.real
eigenvectors_torch = eigenvectors_torch.real

eigenvalues_torch = eigenvalues_torch.cpu().numpy()
eigenvectors_torch = eigenvectors_torch.cpu().numpy()

eigenvalues_match = np.allclose(np.sort(eigenvalues_np), np.sort(eigenvalues_torch))
eigenvectors_match = np.allclose(np.sort(eigenvectors_np), np.sort(eigenvectors_torch))

print("Are eigenvalues equivalent?", eigenvalues_match)
print("Are eigenvectors equivalent?", eigenvectors_match)
