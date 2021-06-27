import torch as t
import torch.nn as nn
from modules.imagination.imagination_core import Imagination_Core
from modules.environment.EM import environment_model

class rollout_engine(nn.Module):
    def __init__(self,main_model,args):
        super().__init__()

        # System parameters
        self.args = args

        # Model-free network
        self.model_free = main_model

        # Environment model
        self.env_model = environment_model()

        # Imagination core
        self.imagination_core = Imagination_Core(self.env_model,self.model_free)

        # Transformer
        self.transformer = nn.Transformer(batch_first=True)

    def forward(self,state,action,device):

        encoding_list = list()

        for i in self.args.num_rollouts:

            # Predicting successor state and reward
            if i == 0:
                next_state, reward = self.imagination_core(state,self.args,device,action)
            else:
                next_state, reward = self.imagination_core(state,self.args,device)

            # Encoding state representation
            encoding = self.env_model.encode(next_state)

            # Adding encoding to the encoding to the list
            encoding_list.append(encoding)

            # Setting state value for the next rollout
            state = next_state

        ############ Transformer ##############




        ########################################

        out = encoding

        return out
