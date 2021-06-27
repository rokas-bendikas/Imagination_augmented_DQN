import torch.nn as as nn


class Imagination_Core(nn.Module):
    def __init__(self, env_model,main_model):
        super().__init__()

        self.model_free = main_model
        self.env_model = env_model


    def forward(self,x,args,device,action = None):

        if not action:
            action = self.model_free(x)

        next_state = self.env_model(x,action,args,device)

        return next_state
