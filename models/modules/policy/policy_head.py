import torch as t
import torch.nn as nn
from modules.imagination.imagination_core import Imagination_Core
from modules.environment.EM import environment_model

class Policy_Head(nn.Module):
    def __init__(self,args):
        super().__init__()

        # System parameters
        self.args = args

        self.network = nn.Sequential(
            nn.Linear(encoding_size,3000),
            nn.ReLU(),
            nn.Linear(3000,1500),
            nn.ReLU(),
            nn.Linear(1500,args.n_actions)
        )

    def forward(self,in):

        out = self.network(in)

        return out
