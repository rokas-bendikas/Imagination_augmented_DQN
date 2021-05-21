#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:01:18 2021

@author: rokas
"""

import logging
from copy import deepcopy
from itertools import count
import torch as t
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from calculate_loss import calculate_loss_DQN, calculate_loss_accelerator
from device import Device
from replay_buffer import ReplayBufferDQN
import numpy as np
from utils import as_tensor
import getch
from datetime import datetime
from models.model_accelerator import Accelerator

class DQN:
    def __init__(self,model,SIMULATOR,args):
        
        self.model = model
        self.target = deepcopy(self.model)
        self.accelerator = Accelerator().to(Device.get_device())
        self.simulator = SIMULATOR(args.headless)
        self.args = args
        self.buffer = ReplayBufferDQN(args)
        self.optimiser_DQN = t.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.optimiser_accelerator = t.optim.Adam(params=self.accelerator.parameters(), lr=self.args.lr)
        self.args.n_actions = self.simulator.n_actions()
        
        self.total_reward = 0
        

    def train(self):
            
            writer = SummaryWriter('runs/')
            
            logging.basicConfig(filename='logs/training.log',
                                    filemode='w',
                                    format='%(message)s',
                                    level=logging.DEBUG)
            
        
            # allocate a device
            n_gpu = t.cuda.device_count()
            if n_gpu > 0:
                Device.set_device(0)
                
        
            self.model.to(Device.get_device())
            self.model.train()
            
            self.accelerator.to(Device.get_device())
            self.accelerator.train()
            
            self.target.to(Device.get_device())
            self.target.eval()
            

            for itr in tqdm(range(self.args.max_episodes), desc='train'):
                
                state = self.simulator.reset()
                state_processed = np.concatenate((state.front_rgb,state.wrist_rgb),axis=2)
                
                episode_reward = 0
                
                for e in count():
                    
                    
                    # During warmup epsilon is one
                    if (itr < self.args.warmup):
                        eps = 1
                    
                    # Calculating epsilon
                    else:
                        eps = max(self.args.eps ** (itr+ self.args.advance_iteration - self.args.warmup), self.args.min_eps)
                        
                    # Epsilon-greedy policy
                    if np.random.RandomState().rand() < eps:
                        action = np.random.RandomState().randint(self.simulator.n_actions())
                    else:
                        action = self.model(as_tensor(state_processed)).argmax().item()
                    
                    """
                    
                    inp = getch.getch()
                    
                    
                    action = -1
                    
                    if inp is 'w':
                        action = 0
                        
                    if inp is 's':
                        action = 1
                    
                    if inp is 'a':
                        action = 2
                        
                    if inp is 'd':
                        action = 3
                        
                    if inp is 'o':
                        action = 4
                        
                    if inp is 'l':
                        action = 5
                    
                    if inp is 'm':
                        action = 6
                        
                   """
                
                    # Agent step   
                    next_state, reward, terminal = self.simulator.step(action,state)
                    
                    # Concainating diffrent cameras
                    next_state_processed = np.concatenate((next_state.front_rgb,next_state.wrist_rgb),axis=2)
                    state_processed = np.concatenate((state.front_rgb,state.wrist_rgb),axis=2)
    
                    # Putting to buffer
                    self.buffer.to_buffer([state_processed, action, reward, next_state_processed, terminal])
                    
                    # Updating running metrics
                    episode_reward += reward
                    self.total_reward += reward
                    state = next_state
                    
                    
                    if ((itr >= self.args.warmup) and (len(self.buffer) > self.args.batch_size)):
                        
                        
                        # Sample a data point from dataset
                        batch = self.buffer.sample_batch(self.model,self.target,self.accelerator,Device.get_device())
                        
                        ###################
                        ### Accelerator ###
                        ###################
                        
                        # Calculate loss for the batch
                        loss_acc = calculate_loss_accelerator(self.accelerator, batch, self.args, Device.get_device())
                        
                        # Predict all possible actions for state and next_stage
                        with t.no_grad():
                            state_img = self.accelerator.imagination_rollout(batch[0],self.args,Device.get_device())
                            next_state_img = self.accelerator.imagination_rollout(batch[3],self.args,Device.get_device())
                        
                        # Compute gradients
                        loss_acc.backward()
                        
                        # Update weights
                        self.optimiser_accelerator.step()
                        
                        # Zero gradients
                        self.accelerator.zero_grad()
                    
                        # Convert to numbers
                        loss_acc = loss_acc.item()
                        
                        
                        
                        ###################
                        ####### DQN #######
                        ###################
                        
                        # Calculate loss for the batch
                        loss_DQN = calculate_loss_DQN(self.model, self.target, batch, state_img, next_state_img, self.args, Device.get_device())
                        
                        # Compute gradients
                        loss_DQN.backward()
                        
                        # Update weights
                        self.optimiser_DQN.step()
                        
                        # Zero gradients
                        self.model.zero_grad()
                    
                        # Convert to numbers
                        loss_DQN = loss_DQN.item()
                        
                        
                        
                    else:
                        
                        loss_DQN = 0
                        
                    
                    # Early termination conditions
                    if (terminal or (e>self.args.episode_length)):
                        break
                
                # Log the results
                logging.debug('Episode reward: {:.2f}, Epsilon: {:.2f}, DQN loss: {:.6f}, Accelerator loss: {:.6f} Buffer size: {}'.format(episode_reward, eps, loss_DQN, loss_acc, len(self.buffer)))
                writer.add_scalar('DQN loss', loss_DQN,itr)
                writer.add_scalar('Accelerator loss', loss_acc,itr)
                writer.add_scalar('Episode reward', episode_reward, itr)
                writer.add_scalar('Total reward',self.total_reward,itr)
                writer.add_scalar('Epsilon value',eps,itr)
                    
                    
                if itr % self.args.target_update_frequency == 0:
                    self.target.load_state_dict(self.model.state_dict())
                
                if itr % self.args.checkpoint_frequency == 0:
                    now = datetime.now()
                    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
                    t.save(self.model.state_dict(),'./checkpoints/model_{}.pt'.format(dt_string))
                
                          
            writer.close()
        
            
            
    