import torch
import torch.nn as nn

con_layer = nn.Conv2d(3, 8, (3, 5), stride = (2, 1), padding = (4, 2), dilation = (3, 1))

input = torch.randn(20, 3, 28, 28)
output = con_layer(input)

con_layer2 = nn.Conv2d(3, 8, 5, padding = "same")

output2 = con_layer2(input)

print(output)
print(output2)
