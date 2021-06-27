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

class action_predictor(BaseModel):
    def __init__(self,args):
        super(action_predictor,self).__init__()

        self.args = args


        self.in_channels = 6


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
            nn.Softmax(dim=1))


    def forward(self,state):

        # For a single input processing
        if(len(state.shape)==3):
            state = state.unsqueeze(0)


        action_discrete = self.network(state).argmax(dim=1)


        # Convert DQN discrete action to continuous
        action = self._action_discrete_to_continous(action_discrete)

        out = [action,action_discrete]


        return out



    def _action_discrete_to_continous(self,actions):

        actions_continous = list()

        for a in actions.cpu().numpy():

            # delta orientation
            d_quat = np.array([0, 0, 0, 1])

            # delta position
            d_pos = np.zeros(3)

            # For positive magnitude
            if(a%2==0):
                a_val = int(a/2)
                d_pos[a_val] = 0.02

            # For negative magnitude
            else:
                a_val = int((a-1)/2)
                d_pos[a_val] = -0.02

            # Forming action as expected by the environment
            actions_continous.append(np.concatenate([d_pos, d_quat, [0]]))

        return actions_continous
