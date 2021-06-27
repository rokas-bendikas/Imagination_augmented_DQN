#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 22:32:03 2021

@author: rokas
"""

import torch as t
import torch.nn.functional as f
import time
from cpprb import PrioritizedReplayBuffer,ReplayBuffer
from utils.utils import queue_to_data,as_tensor



class ReplayBufferDQN:
    def __init__(self,args):

        self.memory = PrioritizedReplayBuffer(args.buffer_size,
                              {"obs": {"shape": (96,96,6)},
                               "act": {},
                               "rew": {},
                               "terminal": {}},
                              alpha=0.7,
                              next_of="obs")


        self.accelerator_memory = ReplayBuffer(1000,
                          {"obs": {"shape": (96,96,6)},
                           "act": {}},
                          next_of="obs")

        self.length = 0
        self.args = args



    def sample_batch(self,model,target_net,device,beta):

        batch = list()

        # Get the priotatized batch
        sample = self.memory.sample(self.args.batch_size,beta)

        # Structure to fit the network
        states = as_tensor(sample['obs'],device=device).permute(0,3,1,2)
        actions = as_tensor(sample['act'],t.long,device=device)
        rewards = as_tensor(sample['rew'],device=device)
        next_states = as_tensor(sample['next_obs'],device=device).permute(0,3,1,2)
        terminals = as_tensor(sample['terminal'],device=device)

        batch.append([states,actions,rewards,next_states,terminals])


        with t.no_grad():
            # Calculate the loss to determine utility
            target = rewards + terminals * self.args.gamma * target_net(next_states).max()
            predicted = model(states).gather(1,actions)


        new_priorities = f.smooth_l1_loss(predicted, target,reduction='none').cpu().numpy()

        # Get the indices of the samples
        indexes = sample["indexes"]

        self.memory.update_priorities(indexes,new_priorities)

        # Get accelerator batch
        sample = self.accelerator_memory.sample(self.args.batch_size)

        # Structure to fit the network
        states = as_tensor(sample['obs'],device=device).permute(0,3,1,2)
        actions = as_tensor(sample['act'],t.long,device=device)
        next_states = as_tensor(sample['next_obs'],device=device).permute(0,3,1,2)

        batch.append([states,actions,next_states])


        return batch

    def sample_batch_accelerator(self,device):


        # Get accelerator batch
        sample = self.accelerator_memory.sample(self.args.batch_size)

        # Structure to fit the network
        states = as_tensor(sample['obs'],device=device).permute(0,3,1,2)
        actions = as_tensor(sample['act'],t.long,device=device)
        next_states = as_tensor(sample['next_obs'],device=device).permute(0,3,1,2)

        return [states,actions,next_states]


    def load_queue(self,queue,lock):

        # Count of samples loaded
        num_loaded = 0

        # Load until reaching batch size
        while num_loaded < self.args.batch_size:

            # If the queue is not empty
            if not queue.empty():

                # Increasing loaded sample count
                num_loaded += 1

                # Read from the queue
                lock.acquire()
                data = queue_to_data(queue.get())
                lock.release()

                # Expand
                state = data[0]
                action = data[1]
                reward = data[2]
                next_state = data[3]
                terminal = data[4]


                # Store for replay buffer
                self.memory.add(obs=state,
                                act=action,
                                rew=reward,
                                next_obs=next_state,
                                terminal=terminal)

                # Set the buffer current length
                self.length = min(self.args.buffer_size,self.length+1)

                # Update accelerator buffer
                
                self.accelerator_memory.add(obs=state,
                            act=action,
                            next_obs=next_state)



    def __len__(self):

        return self.length
