import torch.nn as nn


class Detectuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.Compile=nn.Sequential(
            nn.Linear(2,32),
            nn.Sigmoid(),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, 2),

        )


    def forward(self, x):
        x=self.Compile(x)
        return x
