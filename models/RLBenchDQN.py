import torch.nn as nn
from models.base import BaseModel




class DQN(BaseModel):
    def __init__(self):
        super().__init__()
        

        self.network = nn.Sequential(
            
            #96x96x6
            
            nn.Conv2d(in_channels=6,out_channels=32,kernel_size=5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            #48x48x32
            
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            #24x24x64
            
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            #12x12x128
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            #6*6*256
            
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            #3*3*256
            
            nn.Flatten(),
            nn.Linear(2304, out_features=1152), 
            nn.ReLU(),
            nn.Linear(1152, out_features=7))
            

    def forward(self,x):
        
        if(len(x.shape)==3):
            x = x.unsqueeze(0).permute(0,3,1,2)
        
        y = self.network(x)
    
        
        
        return y
