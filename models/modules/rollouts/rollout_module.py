import torch as t
import torch.nn as nn
from models.modules.imagination.imagination_core import Imagination_Core
from models.modules.environment.EM import environment_model
from models.modules.rollouts.utils import rollout_encoder


class rollout_engine(nn.Module):
    def __init__(self,main_model,args):
        super().__init__()

        # System parameters
        self.args = args

        # Model-free network
        self.model_free = main_model

        # Environment model
        self.env_model = environment_model(args).share_memory()

        # Imagination core
        self.imagination_core = Imagination_Core(self.env_model,self.model_free).share_memory()

        # State encoder
        self.state_encoder = rollout_encoder()

        # Transformer
        #self.transformer = nn.Transformer(batch_first=True).share_memory()

    def forward(self,state,action,device):

        encoding_list = list()

        for i in range(self.args.num_rollouts):

            # Predicting successor state and reward
            if i == 0:
                next_state = self.imagination_core(state,self.args,device,action)
            else:
                next_state= self.imagination_core(state,self.args,device)

            # Encoding state representation
            encoding = self.state_encoder(next_state)

            # Adding encoding to the encoding to the list
            encoding_list.append(encoding)

            # Setting state value for the next rollout
            state = next_state

        ############ Transformer ##############




        ########################################

        out = encoding

        return out

    def get_loss(self,batch,device):

        # Environmental model loss
        loss = self.env_model(batch,device)

        return loss
