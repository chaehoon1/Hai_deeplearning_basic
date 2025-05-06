import torch
import numpy as np

#tensor 연산

#1
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

#2
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#3
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
print(f"tensor * tensor \n {tensor * tensor}")
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

#4
print(tensor, "\n")
tensor.add_(5)
print(tensor)

#5
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")


