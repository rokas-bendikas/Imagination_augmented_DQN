

import torch as t
from models.base import BaseModel
from models.modules.EM_utils import DoubleConv,Down,Up,OutConv
from utils.utils import as_tensor



class rollout_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Init the network
        self._init_encoder


    # Predict the next state representation given a single action
    def forward(self,state,reward,device):

        if(len(state.shape)==3):
            state = state.unsqueeze(0).permute(0,3,1,2)

        # Create a tiled reward representation, shape [batch_size,1,img_height,img_width]
        reward_tile = t.full((state.shape[0],1,state.shape[2],state.shape[3]),reward,device=device)

        # Stack one-hot tiled tensors on top of the input image
        state_reward = t.cat((state,reward_tile),1)

        # Encode the representation
        encoding = self.encode(state_reward)

        return encoding

    def encode(self,x):

        out = self.encoder(x)

        return out


    def _init_encoder(self):

        self.encoder = nn.Sequatial(
            DoubleConv(7,64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024),
            nn.Flatten())
