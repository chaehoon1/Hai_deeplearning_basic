import torch
import numpy as np

t1 = torch.rand(2, 3, 4)

n = np.ones((2, 4, 5), dtype = np.float32)

t2 = torch.from_numpy(n)

t3 = torch.matmul(t1, t2)

print(t3.shape)
print(t3.dtype)
