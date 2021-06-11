#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:38:51 2021

@author: rokas
"""


import torch as t
import torch.nn as nn
from models.base import BaseModel

class A2C_model(BaseModel):
    def __init__(self,args):
        super(A2C_model,self).__init__()
        
        # Save args
        self.args = args
        
        # Input dimensions
        if self.args.accelerator:
            self.in_channels = 48
        else:
            self.in_channels = 6
        
        # Actor network architecture
        self.actor = nn.Sequential(
            
            #96x96x48
            
            nn.Conv2d(in_channels=self.in_channels,out_channels=64,kernel_size=5,padding=2),
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
            nn.Linear(2304, out_features=args.n_actions),
            nn.Softmax(dim=1))
        
        
        # Critic network architecture
        self.critic = nn.Sequential(
            
            #96x96x48
            
            nn.Conv2d(in_channels=self.in_channels,out_channels=64,kernel_size=5,padding=2),
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
            nn.Linear(2304, out_features=1))
            

    def forward(self,x,rollout=None):
        
        # For a single input processing
        if(len(x.shape)==3):
            x = x.unsqueeze(0).permute(0,3,1,2)
        
        # Add rollout values
        if self.args.accelerator:
            x = t.cat((x,rollout),1)
        
        
        # Policy head
        policy = self.actor(x)
        
        # Value head
        value = self.critic(x)
        
        return (value,policy)
    
    