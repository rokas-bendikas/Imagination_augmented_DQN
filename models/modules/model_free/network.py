#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:35:04 2021

@author: rokas
"""

import torch as t
import torch.nn as nn
import numpy as np
from models.base import BaseModel

class action_predictor(nn.Module):
    def __init__(self,args):
        super(action_predictor,self).__init__()

        self.args = args


        self.in_channels = 9


        self.network = nn.Sequential(

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
            # 4608
            nn.Linear(4608,2304),
            nn.ReLU(),
            nn.Linear(2304,self.args.n_actions),
            nn.Softmax(dim=1)
            )


    def forward(self,state):

        # For a single input processing
        if(len(state.shape)==3):
            state = state.unsqueeze(0)

        action_discrete = self.network(state)

        return action_discrete

    def get_action(self,state):

        # For a single input processing
        if(len(state.shape)==3):
            state = state.unsqueeze(0)

        with t.no_grad():

            action_discrete = self.network(state).argmax(dim=1)

        return action_discrete
