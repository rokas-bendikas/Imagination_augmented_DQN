import torch as t
import torch.nn as nn


class BaseModel(nn.Module):
    
   
        
    def forward(self,x,rollout=None):
        raise NotImplementedError('You need to define a speak method!')
        
        
    def save(self, file):
        if file is None or file == '':
            return

        t.save(self.state_dict(), file)

    def load(self, file):
        if file is None or file == '':
            return

        self.load_state_dict(t.load(file, map_location='cpu'))
