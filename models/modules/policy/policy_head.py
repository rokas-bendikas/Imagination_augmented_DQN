import torch as t
import torch.nn as nn
from models.base import BaseModel


class PolicyHead(BaseModel):
    def __init__(self,args):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1536,768),
            nn.LeakyReLU(0.2),
            nn.Linear(768,384),
            nn.LeakyReLU(0.2),
            nn.Linear(384,args.n_actions),
        )

    def forward(self,x):

        out = self.network(x)

        return out
