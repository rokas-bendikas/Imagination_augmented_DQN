import torch.nn as nn


class Imagination_Core(nn.Module):
    def __init__(self, env_model,main_model):
        super().__init__()

        self.model_free = main_model
        self.env_model = env_model


    def forward(self,state,args,device,action = None):

        if not action:
            _,action = self.model_free(state)

        next_state = self.env_model(state,action,args,device)

        return next_state
