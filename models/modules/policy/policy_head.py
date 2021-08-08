import torch as t
import torch.nn as nn
from models.base import BaseModel


class PolicyHead(BaseModel):
    def __init__(self,args):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(512*(args.n_actions+1),1280),
            nn.ReLU(),
            nn.Linear(1280,640),
            nn.ReLU(),
            nn.Linear(640,args.n_actions))

    def forward(self,x):

        out = self.network(x)

        return out
