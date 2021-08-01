#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:32:03 2021

@author: rokas
"""

import torch as t
import torch.nn.functional as f
import numpy as np
import time
from cpprb import PrioritizedReplayBuffer,ReplayBuffer
from utils.utils import process_batch,plot_batch
from PIL import Image
from torchvision import transforms
import time
from copy import deepcopy



class ReplayBufferDQN:
    def __init__(self,args):


        self.memory = PrioritizedReplayBuffer(args.buffer_size,
                              {"obs": {"shape": (4,128,128)},
                               "act": {},
                               "rew": {},
                               "terminal": {}},
                              alpha=0.5,
                              next_of="obs")



        self.length = 0
        self.args = args


    def sample_batch(self,device,beta,model,target_net,warmup,batch_size=None):

        if batch_size==None:
            batch_size = self.args.batch_size

        # Get the priotatized batch
        sample = self.memory.sample(batch_size,beta)

        # Structure to fit the network
        states = t.tensor(sample['obs'],device=device) / 255
        actions = t.tensor(sample['act'],dtype=t.long,device=device)
        rewards = t.tensor(sample['rew'],device=device)
        next_states = t.tensor(sample['next_obs'],device=device) / 255
        terminals = t.tensor(sample['terminal'],dtype=t.bool,device=device)

        if self.args.plot:
            plot_batch(states)


        weights = t.tensor(sample['weights'],device=device).unsqueeze(dim=1)

        # Get the indices of the samples
        indices = sample["indexes"]

        if not warmup:

            with t.no_grad():

                target = rewards + (1 - terminals.int()) * self.args.gamma * target_net(next_states,device).max(dim=1,keepdim=True)[0]

                predicted = model(states,device).gather(1,actions)

            new_priorities = f.smooth_l1_loss(predicted, target,reduction='none')

            self.memory.update_priorities(indices,new_priorities.squeeze().cpu().numpy())


        return (states,actions,rewards,next_states,terminals,weights)



    def append(self,batch):

        state,action,reward,next_state,terminal = process_batch(batch)

        # Store for replay buffer
        self.memory.add(obs=state,
                        act=action,
                        rew=reward,
                        next_obs=next_state,
                        terminal=terminal)


        # Set the buffer current length
        self.length = min(self.args.buffer_size,self.length+1)


    def __len__(self):

        return self.length
