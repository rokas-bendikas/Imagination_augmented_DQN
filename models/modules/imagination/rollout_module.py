import torch as t
import torch.nn as nn
from models.modules.imagination.imagination_core import ImaginationCore
from models.modules.dynamics.dynamics_model import DynamicsModel
from models.base import BaseModel


class RolloutEngine(BaseModel):
    def __init__(self,action_distiller,args):
        super().__init__()

        # System parameters
        self.args = args

        # Model-free network
        self.distiller = action_distiller

        # Environment model
        self.dynamics_model = DynamicsModel()

        # Imagination core
        self.imagination_core = ImaginationCore(self.dynamics_model,self.distiller)

        # LSTM
        self.lstm = nn.LSTM(256, 256)


    def forward(self,state,action,device):

        encodings = t.zeros((self.args.num_rollouts,self.args.batch_size,256),device=device)

        for i in range(self.args.num_rollouts):

            # Predicting successor state and reward
            if i == 0:
                next_state = self.imagination_core(state,action)
            else:
                next_state = self.imagination_core(state)

            # Adding encoding to the encoding to the list
            encodings[i,:,:] = next_state

            # Setting state value for the next rollout
            state = next_state

        ############ Transformer ##############
        h0 = t.zeros((1,self.args.batch_size,256),device=device)
        c0 = t.zeros((1,self.args.batch_size,256),device=device)

        self.lstm.flatten_parameters()
        _, (hn,_) = self.lstm(encodings, (h0, c0))

        ########################################

        return hn

    def get_loss(self,batch,device):

        loss = dict()

        # Environmental model loss
        loss['dynamics_model'] = self.dynamics_model.get_loss(batch)


        return loss
