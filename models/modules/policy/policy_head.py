import torch as t
import torch.nn as nn
from models.base import BaseModel


class PolicyHead(BaseModel):
    def __init__(self,args):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4608,2304),
            nn.LeakyReLU(0.2),
            nn.Linear(2304,1152),
            nn.LeakyReLU(0.2),
            nn.Linear(1152,576),
            nn.LeakyReLU(0.2),
            nn.Linear(576,args.n_actions),
        )

    def forward(self,x):


        out = self.network(x)

        return out
