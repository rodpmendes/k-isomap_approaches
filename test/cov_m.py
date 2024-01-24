import numpy as np

# Sample data (each row represents a variable, and each column represents an observation)
data = np.array([[1, 2, 3, 4, 5, 6],
                 [4, 5, 6, 7, 8, 9],
                 [7, 8, 9, 1, 2, 3]])

# Calculate the covariance matrix
cov_matrix = np.cov(data.T)

print("Covariance Matrix:")
print(cov_matrix)
