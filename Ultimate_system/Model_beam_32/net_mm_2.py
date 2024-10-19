import torch.nn as nn
import params

class Mmaid(nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding=nn.Sequential(
            nn.Linear(in_features=params.M, out_features=2*params.M, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.GRU = nn.GRU(input_size=2*params.M , hidden_size=params.M,
                          num_layers=1, bias=True, batch_first=True, bidirectional=False)

        self.Classifier=nn.Linear(params.M,params.M)

    def forward(self, x):
        x = self.Embedding(x)
        out, _ = self.GRU(x)
        out = self.Classifier(out)
        return out
