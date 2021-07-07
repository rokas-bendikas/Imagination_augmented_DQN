#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 22:27:43 2021

@author: rokas
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as f






class DynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Init network modules
        self.model = nn.Sequential(
            nn.Linear(512,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.ReLU())


    # Predict the next state representation given a single action
    def forward(self,state,action):


        # Set the action tile to 1
        action_tiled = action.expand(action.shape[0],256)

        # Stack one-hot tiled tensors on top of the input image
        state_action = t.cat((state,action_tiled),1)

        # Pass through the network for state predicition
        next_state = self.model(state_action)

        return next_state


    def get_loss(self,batch):

        state, action, next_state = batch

        predicted = self(state,action)

        loss = f.mse_loss(predicted,next_state)

        return loss
