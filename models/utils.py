import torch as t
from models.base import BaseModel
from models.modules.EM_utils import DoubleConv,Down,Up,OutConv
from utils.utils import as_tensor

class policy_head(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.args = args

        self.network = nn.Sequential(
            nn.Linear(self.encoding_size+4608,(self.encoding_size+4608) // 2),
            nn.ReLU(),
            nn.Linear((self.encoding_size+4608) // 2, 7)
        )


    def forward(self,x,args,device):

        action = self.model_free.get_action(x,greedy=True)

        next_state,reward = self.env_model(x,action,args,device)

        return [next_state,reward]
