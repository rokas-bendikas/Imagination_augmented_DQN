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

class ActionDistiller(BaseModel):
    def __init__(self,args):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(768,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048,4096),
            nn.LeakyReLU(0.2),
            nn.Linear(4096,2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,args.n_actions),
            nn.Softmax(dim=1))


    def forward(self,state):

        action_discrete = self.network(state)

        return action_discrete

    def get_action(self,state):

        with t.no_grad():

            action_discrete = self.network(state).argmax(dim=1)

        return action_discrete.unsqueeze(1)
