import torch.nn as nn
import rff
import params


class Locaid(nn.Module):

    def __init__(self):
        super().__init__()

        self.Ziplayer=nn.Sequential(
            nn.Linear(4*params.R,128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.Classifier=nn.Linear(64,params.M)

    def PositionRFF(self,x,device):
        x = x.to('cpu')
        encoding = rff.layers.PositionalEncoding(sigma=params.sigma, m=params.R)
        xg = encoding(x)
        xg = xg.to(device)
        return xg

    def forward(self, x,device='cpu'):
        x=self.PositionRFF(x,device)
        x=self.Ziplayer(x)
        x=self.Classifier(x)
        return x
