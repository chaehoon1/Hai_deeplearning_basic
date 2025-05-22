import torch
import torch.nn as nn

input = torch.tensor([[[[7., 5., 0., 3.],
    [10., 4., 21., 2.],
    [6., 1., 7., 0.,],
    [5., 0., 8., 4.]]]])

pool = nn.MaxPool2d(kernel_size=2, stride=2)

output = pool(input)

print(output)


