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
from utils import queue_to_data



class ReplayBufferDQN:
    def __init__(self,args):
        
        self.memory = PrioritizedReplayBuffer(args.buffer_size,
                              {"obs": {"shape": (96,96,6)},
                               "act": {},
                               "rew": {},
                               "terminal": {}},
                              alpha=0.7,
                              next_of="obs")
        
        self.length = 0
        self.args = args
        
  
        
    def sample_batch(self,model,target_net,accelerator,device,beta):
        
        # Get the batch
        sample = self.memory.sample(self.args.batch_size,beta)
        
        # Structure to fit the network
        s = t.tensor(sample['obs'])
        a = t.tensor(sample['act'])
        r = t.tensor(sample['rew'])
        ns = t.tensor(sample['next_obs'])
        term = t.tensor(sample['terminal'])
        
        # Move to the training device memory
        states = s.permute(0,3,1,2).to(Device.get_device())
        actions = a.type(t.int64).to(Device.get_device())
        rewards = r.to(Device.get_device())
        next_states = ns.permute(0,3,1,2).to(Device.get_device())
        terminals = term.to(Device.get_device())
        
        
        with t.no_grad():
            
            # Create a rollout for the whole batch 
            states_rollout = accelerator.rollout(states,self.args,device)
            next_states_rollout = accelerator.rollout(next_states,self.args,device)
        
         
            # Calculate the loss to determine utility
            target = rewards + terminals * self.args.gamma * target_net(next_states,next_states_rollout).max()
            predicted = model(states,states_rollout).gather(1,actions)
                  
            
        new_priorities = f.smooth_l1_loss(predicted, target,reduction='none').cpu().numpy()
        #new_priorities = f.smooth_l1_loss(predicted, target,reduction='none').item()
        
        # Get the indices of the samples
        indexes = sample["indexes"]
            
        self.memory.update_priorities(indexes,new_priorities)
        
        
        return states,actions,rewards,next_states,terminals
    
    
    def load_queue(self,queue,lock,args):
        
            for i in range(int(queue.qsize())):
                
                
                # Read from the queue
                # The critical section begins
                lock.acquire()
                data = queue_to_data(queue.get())
                lock.release()
                
                # Convert to numpy for storage
                state = data[0].numpy()
                action = data[1].numpy()
                reward = data[2].numpy()
                next_state = data[3].numpy()
                terminal = data[4].numpy()
                
                
                
                self.memory.add(obs=state,
                                act=action,
                                rew=reward,
                                next_obs=next_state,
                                terminal=terminal)
                
                self.length = min(self.args.buffer_size,self.length+1)
        
        
    def __len__(self):
        
        return self.length