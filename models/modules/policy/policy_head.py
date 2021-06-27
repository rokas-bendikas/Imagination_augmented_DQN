import torch as t
import torch.nn as nn
from models.modules.imagination.imagination_core import Imagination_Core
from models.modules.environment.EM import environment_model

class policy_head(nn.Module):
    def __init__(self,args):
        super().__init__()

        # System parameters
        self.args = args

        self.network = nn.Sequential(
            nn.Linear(6000,3000),
            nn.ReLU(),
            nn.Linear(3000,1500),
            nn.ReLU(),
            nn.Linear(1500,args.n_actions)
        )

    def forward(self,x):

        out = self.network(x)

        return out
