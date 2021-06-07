import torch as t
import torch.nn as nn
from models.base import BaseModel





class DQN(BaseModel):
    def __init__(self):
        super(DQN,self).__init__()
        

        self.network = nn.Sequential(
            
            #96x96x48
            
            nn.Conv2d(in_channels=48,out_channels=64,kernel_size=5,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            #48x48x64
            
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            #24x24x128
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            #12x12x256
            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=5,padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            #6*6*512
            
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=5,padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            #3*3*512
            
            nn.Flatten(),
            nn.Linear(4608, out_features=2304), 
            nn.ReLU(),
            nn.Linear(2304, out_features=7))
            

    def forward(self,x,rollout):
        
        if(len(x.shape)==3):
            x = x.unsqueeze(0).permute(0,3,1,2)
            
        inp = t.cat((x,rollout),1)
        
        y = self.network(inp)
    
        
        
        return y
