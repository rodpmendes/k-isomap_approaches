from time import time
import torch
import numpy as np

matrices = torch.randn(10000, 200, 200)
print('matrix size', matrices.size())

# Create a random 4096x4096 matrix with values between 0 and 10000
matrix = np.random.randint(0, 10001, (4096, 4096)).astype(np.float32)

t1 = time()
v, w = np.linalg.eig(np.cov(matrices[0].T))
t2 = time()
np_time = t2-t1
print('np time', np_time)

t1=time()
torch.linalg.eigh(matrices[0])
torch.cuda.synchronize()
t2=time()
cpu_time = t2-t1
print('cpu time', cpu_time)

matrices = matrices[0].to(torch.device('cuda'))

t1=time()
torch.linalg.eigh(matrices)
torch.cuda.synchronize()
t2=time()
gpu_time = t2-t1
print('gpu time', gpu_time)



input()