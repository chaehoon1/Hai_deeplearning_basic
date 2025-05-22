import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.seq = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(16, 32, 3, padding = 1),
                nn.ReLU(), 
                nn.MaxPool2d(2, 2),

                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 10)
                )
    def forward(self, x):
        y = self.seq(x)
        return y

model = MyCNN()
x = torch.randn(4, 1, 28, 28)  
y = model(x)
print(y.shape)  


