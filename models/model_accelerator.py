#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 22:27:43 2021

@author: rokas
"""


from models.base import BaseModel
from models.accelerator_utils import *


class Accelerator(BaseModel):
    def __init__(self, bilinear=True):
        super(Accelerator, self).__init__()
        
     
        self.bilinear = bilinear
        

        self.inc = DoubleConv(6,64)
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
        
        
             

    def forward(self,x):
        
        if(len(x.shape)==3):
            x = x.unsqueeze(0).permute(0,3,1,2)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits