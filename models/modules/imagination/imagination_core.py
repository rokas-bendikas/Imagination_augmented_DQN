import torch as t
import torch.nn as nn


class ImaginationCore(nn.Module):
    def __init__(self, dynamics_model,action_distiller):
        super().__init__()

        self.distiller = action_distiller
        self.dynamics_model = dynamics_model


    def forward(self,state,action = None):

        with t.no_grad():

            if action==None:
                action = self.distiller.get_action(state)

            next_state = self.dynamics_model(state,action)


        return next_state
