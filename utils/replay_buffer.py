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
from utils.utils import queue_to_data
from PIL import Image
from torchvision import transforms
import time



class ReplayBufferDQN:
    def __init__(self,args):

        self.memory = PrioritizedReplayBuffer(args.buffer_size,
                              {"obs": {"shape": (9,96,96)},
                               "act": {},
                               "rew": {},
                               "terminal": {}},
                              alpha=0.7,
                              next_of="obs")

        self.length = 0
        self.args = args


    def sample_batch(self,warmup,device,beta,model=None,target_net=None):

        # Get the priotatized batch
        sample = self.memory.sample(self.args.batch_size,beta)

        # Structure to fit the network
        states = t.tensor(sample['obs'],device=device) / 255
        actions = t.tensor(sample['act'],dtype=t.long,device=device)
        rewards = t.tensor(sample['rew'],device=device)
        next_states = t.tensor(sample['next_obs'],device=device) / 255
        terminals = t.tensor(sample['terminal'],dtype=t.bool,device=device)

        if not warmup:

            print('here')

            with t.no_grad():
                # Calculate the loss to determine utility
                target = rewards + terminals * self.args.gamma * target_net(next_states,device).max()
                predicted = model(states,device).gather(1,actions)


            new_priorities = f.smooth_l1_loss(predicted, target,reduction='none').cpu().numpy()

            # Get the indices of the samples
            indexes = sample["indexes"]

            self.memory.update_priorities(indexes,new_priorities)


        return (states,actions,rewards,next_states,terminals)


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


    def load_queue_warmup(self,queue,lock):

        # Load until reaching batch size
        for i in range(queue.qsize()):


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



    def __len__(self):

        return self.length
