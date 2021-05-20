#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:32:03 2021

@author: rokas
"""

import torch as t
from device import Device
import torch.nn.functional as f
from cpprb import PrioritizedReplayBuffer




class ReplayBufferDQN:
    def __init__(self,args):
        
        self.memory = PrioritizedReplayBuffer(args.buffer_size,
                              {"obs": {"shape": (96,96,6)},
                               "act": {},
                               "rew": {},
                               "next_obs": {"shape": (96,96,6)},
                               "terminal": {}})
        
        self.length = 0
        self.args = args
        

          
    def to_buffer(self,data):
        
        self.memory.add(obs=data[0],
                                act=data[1],
                                rew=data[2],
                                next_obs=data[3],
                                terminal=data[4])
                
        self.length = min(self.args.buffer_size,self.length+1)
        
        
        
    def sample_batch(self,model,target_net,accelerator,device):
        
        sample = self.memory.sample(self.args.batch_size)
        
        s = t.tensor(sample['obs'])
        a = t.tensor(sample['act'])
        r = t.tensor(sample['rew'])
        ns = t.tensor(sample['next_obs'])
        term = t.tensor(sample['terminal'])
    
        states = s.permute(0,3,1,2).to(Device.get_device())
        actions = a.type(t.int64).to(Device.get_device())
        rewards = r.to(Device.get_device())
        next_states = ns.permute(0,3,1,2).to(Device.get_device())
        terminals = term.to(Device.get_device())
        
        
        with t.no_grad():
            state_img = accelerator.imagination_rollout(states,self.args,device)
            next_state_img = accelerator.imagination_rollout(next_states,self.args,device)
        
        # Concatinate state obs and model prediction
        state_augmented = t.cat((states,state_img),1)
        next_state_augmented = t.cat((next_states,next_state_img),1)
        
        
        indexes = sample["indexes"]
            
        with t.no_grad():
              
            target = rewards + terminals * self.args.gamma * target_net(next_state_augmented).max()
            predicted = model(state_augmented).gather(1,actions)
                  
            
        new_priorities = f.smooth_l1_loss(predicted, target,reduction='none').cpu().numpy()
        new_priorities[new_priorities<1] = 1
            
        self.memory.update_priorities(indexes,new_priorities)
        
        
        return states,actions,rewards,next_states,terminals
        
        
    def __len__(self):
        
        return self.length