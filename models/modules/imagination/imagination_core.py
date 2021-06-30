import torch as t
import torch.nn as nn


class Imagination_Core(nn.Module):
    def __init__(self, env_model,main_model):
        super().__init__()

        self.model_free = main_model
        self.env_model = env_model


    def forward(self,state,args,device,action = None):

        self.env_model.eval()

        with t.no_grad():
            if not action:
                action = self.model_free.get_action(state)

            next_state = self.env_model(state,action,device)

        self.env_model.train()

        return next_state
