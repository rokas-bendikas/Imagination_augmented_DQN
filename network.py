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
import numpy as np
from utils import as_tensor,data_to_queue
from replay_buffer import ReplayBufferDQN
from optimise_model import optimise_DQN,optimise_accelerator

        

def collect(SIMULATOR,model_shared,accelerator_shared,queue,lock,args):
    writer = SummaryWriter('runs/')
            
    logging.basicConfig(filename='logs/collector.log',
                                    filemode='w',
                                    format='%(message)s',
                                    level=logging.DEBUG)
            
        
    # allocate a device
    Device.set_device('cpu')
            
    simulator = SIMULATOR(args.headless)
          
    args.n_actions = simulator.n_actions()
        
    total_reward = 0
             

    for itr in tqdm(count(), position=0, desc='collector'):
                
        state = simulator.reset()
        state_processed = np.concatenate((state.front_rgb,state.wrist_rgb),axis=2)
                
        episode_reward = 0
                
        for e in count():
            # During warmup epsilon is one
            if (itr < args.warmup):
                eps = 1
                    
            # Calculating epsilon
            else:
                eps = max(args.eps ** (itr+ args.advance_iteration - args.warmup), args.min_eps)
                        
            # Epsilon-greedy policy
            if np.random.RandomState().rand() < eps:
                action = np.random.RandomState().randint(simulator.n_actions())
            else:
                
                rollout = accelerator_shared.rollout(as_tensor(state_processed),args,Device.get_device())
                
                lock.acquire()
                action = model_shared(as_tensor(state_processed),rollout).argmax().item()
                lock.release()
                
            # Agent step   
            next_state, reward, terminal = simulator.step(action,state)
            
            # Concainating diffrent cameras
            next_state_processed = np.concatenate((next_state.front_rgb,next_state.wrist_rgb),axis=2)
            state_processed = np.concatenate((state.front_rgb,state.wrist_rgb),axis=2)
            
            # Storing the data in the queue
            lock.acquire()
            queue.put(data_to_queue(state_processed, action, reward, next_state_processed, terminal))
            lock.release()

            
            # Updating running metrics
            episode_reward += reward
            total_reward += reward
            state = next_state
            
            
            # Early termination conditions
            if (terminal or (e>args.episode_length)):
                break
        
        # Log the results
        logging.debug('Episode reward: {:.2f}, Epsilon: {:.2f}'.format(episode_reward, eps))
        writer.add_scalar('Episode reward', episode_reward, itr)
        writer.add_scalar('Total reward',total_reward,itr)
        writer.add_scalar('Epsilon value',eps,itr)
         
                  
    writer.close()
            
            
    
def optimise(model_shared,accelerator_shared,queue,lock,args):
        
        writer = SummaryWriter('runs/')
        
        logging.basicConfig(filename='logs/optimiser.log',
                                filemode='w',
                                format='%(message)s',
                                level=logging.DEBUG)
        
    
        # allocate a device
        n_gpu = t.cuda.device_count()
        if n_gpu > 0:
            Device.set_device(0)
            
            
        args.n_actions = 7
            
            
            
        model = deepcopy(model_shared)
        model.to(Device.get_device())
        model.train()
        
        accelerator = deepcopy(accelerator_shared)
        accelerator.to(Device.get_device())
        accelerator.train()
        
        
        target = deepcopy(model)
        target.to(Device.get_device())
        target.eval()
        
        buffer = ReplayBufferDQN(args)
        
        optimiser_DQN = t.optim.Adam(params=model.parameters(), lr=args.lr)
        optimiser_accelerator = t.optim.Adam(params=accelerator.parameters(), lr=args.lr)
        

        
        
        for itr in tqdm(count(), position=1, desc='optimiser'):
            
            loss_a = 0
            loss_d = 0
            
            buffer.load_queue(queue,lock,args)
            
            while (len(buffer) < args.batch_size):
                buffer.load_queue(queue,lock,args)
                
            
                
            # Sample a data point from dataset
            batch = buffer.sample_batch(model,target,accelerator,Device.get_device())
            
            ###################
            ### Accelerator ###
            ###################
            
            # Calculate loss for the batch
            loss = calculate_loss_accelerator(accelerator, batch, args, Device.get_device())
            
    
            # Update shared accelerator model
            loss_acc = optimise_accelerator(accelerator_shared,accelerator,loss,optimiser_accelerator,lock)
            
            
            
            ###################
            ####### DQN #######
            ###################
            
            # Create a rollout from the model accelerator
            with t.no_grad():
                state_rollout = accelerator.rollout(batch[0],args,Device.get_device())
                next_state_rollout = accelerator.rollout(batch[3],args,Device.get_device())
            
            # Calculate loss for the batch
            loss = calculate_loss_DQN(model, target, batch, state_rollout, next_state_rollout, args, Device.get_device())
            
            # Update shared model
            loss_DQN = optimise_DQN(model_shared,model,loss,optimiser_DQN,lock)
                
              
                
            loss_a += loss_acc
            loss_d += loss_DQN
                
            
            
            
            # Log the results
            logging.debug('DQN loss: {:.6f}, Accelerator loss: {:.6f}, Buffer size: {}'.format(loss_d, loss_a, len(buffer)))
            writer.add_scalar('DQN loss', loss_DQN,itr)
            writer.add_scalar('Accelerator loss', loss_acc,itr)
            
                
            if itr % args.target_update_frequency == 0:
                lock.acquire()
                target.load_state_dict(model.state_dict())
                lock.release()
            
                   
        writer.close()
    
        
            
            
    