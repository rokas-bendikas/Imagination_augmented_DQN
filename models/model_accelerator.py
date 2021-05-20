#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 22:27:43 2021

@author: rokas
"""

import torch as t
from models.base import BaseModel
from models.accelerator_utils import *
import time


class Accelerator(BaseModel):
    def __init__(self, bilinear=True):
        super(Accelerator, self).__init__()
        
     
        self.bilinear = bilinear
        

        self.inc = DoubleConv(13,64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 6)
        
        
             

    def forward(self,state,action,args,device):
        
        if(len(state.shape)==3):
            state = state.unsqueeze(0).permute(0,3,1,2)
        
        
        
        action_tiled = t.zeros(state.shape[0],args.n_actions,state.shape[2],state.shape[3],device=device)
        action_tiled[:,action,:,:] = 1
        
        state_action = t.cat((state,action_tiled),1)
        
        
        x1 = self.inc(state_action)
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
    
    
    def imagination_rollout(self, state, args, device):
        
        state = state.to(device)
        
        rollout = t.empty(state.shape[0],args.n_actions*6,state.shape[2],state.shape[3],device=device)
        
        for i in range(args.n_actions):
            
            action_tiled = t.zeros(state.shape[0],args.n_actions,state.shape[2],state.shape[3],device=device)
            action_tiled[:,i,:,:] = 1
            
            state_action = t.cat((state,action_tiled),1)
        
        
            x1 = self.inc(state_action)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            out = self.outc(x)
            
            rollout[:,6*i:6*(i+1),:,:]  = out
            
            
        return rollout
            
        
        