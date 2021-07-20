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
            nn.Linear(768,384),
            nn.ReLU(),
            nn.Linear(384,192),
            nn.ReLU(),
            nn.Linear(192,args.n_actions))


    def forward(self,state):

        out = self.network(state)

        return out

    def get_action(self,state):

        ###############################
        ############ DQN ##############
        ###############################

        with t.no_grad():
            action = self(state).argmax(dim=1)

        return action.unsqueeze(1)
