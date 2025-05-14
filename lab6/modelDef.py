import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.seq_modules = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.seq_modules(x)
        return y

myFirstModel = MyModel()


