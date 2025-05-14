import torch
import torch.nn as nn
from modelDef import myFirstModel

loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(myFirstModel.parameters(), lr=0.001)

for epoch in range(100):
    inputs = torch.randn(1, 32)
    labels = torch.randn(1, 1)  

    optimizer.zero_grad()  
    outputs = myFirstModel(inputs) 
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()
