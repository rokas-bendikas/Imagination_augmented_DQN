import torch as t
import torch.nn as nn


class ImaginationCore(nn.Module):
    def __init__(self, dynamics_model):
        super().__init__()

        self.dynamics_model = dynamics_model


    def forward(self,state,DQN,action = None):

        with t.no_grad():

            if action==None:
                action = DQN.get_action(state).detach()

            next_state = self.dynamics_model(state,action).detach()


        return next_state
