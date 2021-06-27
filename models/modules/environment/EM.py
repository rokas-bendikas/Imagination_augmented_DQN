#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 22:27:43 2021

@author: rokas
"""

import torch as t
import torch.nn as nn
from models.modules.environment.utils import DoubleConv,Down,Up,OutConv



class environment_model(nn.Module):
    def __init__(self,args, bilinear=True):
        super().__init__()

        self.args = args

        # Network settings
        self.bilinear = bilinear
        self.factor = 2 if bilinear else 1

        # Init network modules
        self._init_encoder()
        self._init_decoder()


    # Predict the next state representation given a single action
    def forward(self,state,action,args,device):

        if(len(state.shape)==3):
            state = state.unsqueeze(0)

        # Create a tiled one-hot action representation, shape [batch_size,num_actions,img_height,img_width]
        action_tiled = t.zeros(state.shape[0],args.n_actions,state.shape[2],state.shape[3],device=device)

        # Set the action tile to 1
        action_tiled[:,action,:,:] = 1

        # Stack one-hot tiled tensors on top of the input image
        state_action = t.cat((state,action_tiled),1)

        # Pass through the network for state predicition
        next_state = self.encode_decode(state_action)

        return next_state


    def get_loss(self,batch,device):

        state, action, next_state = batch

        predicted = self(state,action,device)

        if self.args.plot:
            plot_data(batch,predicted)

        loss = {'environment-model': f.mse_loss(predicted,next_state)}

        return loss



    def encode(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        out = self.down4(x4)

        return out

    def encode_decode(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)

        return out


    def _init_encoder(self):

        self.inc = DoubleConv(6+self.args.n_actions,64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // self.factor)

    def _init_decoder(self):
        self.up1 = Up(1024, 512 // self.factor, self.bilinear)
        self.up2 = Up(512, 256 // self.factor, self.bilinear)
        self.up3 = Up(256, 128 // self.factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, 6)
