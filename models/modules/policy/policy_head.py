import torch as t
import torch.nn as nn
from models.base import BaseModel


class PolicyHead(BaseModel):
    def __init__(self,args):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(768*args.n_actions,2304),
            nn.ReLU(),
            nn.Linear(2304,1152),
            nn.ReLU(),
            nn.Linear(1152,576),
            nn.ReLU(),
            nn.Linear(576,288),
            nn.ReLU(),
            nn.Linear(288,args.n_actions))

    def forward(self,x):

        out = self.network(x)

        return out
