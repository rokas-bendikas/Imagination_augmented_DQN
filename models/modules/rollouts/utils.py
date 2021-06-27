

import torch as t
import torch.nn as nn
from models.base import BaseModel
from models.modules.environment.utils import DoubleConv,Down,Up,OutConv
from utils.utils import as_tensor



class rollout_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Init the network
        self.encoder = nn.Sequential(
            DoubleConv(6,64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024),
            nn.Flatten())


    # Predict the next state representation given a single action
    def forward(self,state):

        if(len(state.shape)==3):
            state = state.unsqueeze(0)

        # Encode the representation
        encoding = self.encoder(state)

        return encoding
