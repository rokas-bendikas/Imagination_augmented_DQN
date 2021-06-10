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
from utils.device import Device
import time
import numpy as np
from utils.utils import as_tensor,data_to_queue,copy_weights
from utils.replay_buffer import ReplayBufferDQN
from utils.optimise_model import optimise_model
from rlbench.task_environment import InvalidActionError
from pyrep.errors import ConfigurationPathError
t.multiprocessing.set_sharing_strategy('file_system')
        

def collect(simulator,model_shared,queue,lock,args,flush_flag,warmup_flag,beta):
    
    writer = SummaryWriter('tensorboard/col')
            
    logging.basicConfig(filename='logs/collector.log',
                                    filemode='w',
                                    format='%(message)s',
                                    level=logging.DEBUG)
          
            
    n_gpu = t.cuda.device_count()
    if n_gpu > 0:
        Device.set_device(1 % n_gpu)
    
    simulator.launch()
             
    total_reward = 0
    
    # Beta for IS  linear annealing
    beta.value = 0.4
    beta_step = (1 - beta.value)/args.num_episodes
    
    model_local = deepcopy(model_shared)
    model_local.to(Device.get_device())
    
    
    
    ###############################
    ############ DQN ##############
    ###############################
    
    if args.model == "DQN":
    
        # Epsilon linear annealing
        epsilon = args.eps
        epsilon_step = args.eps/args.num_episodes
   

    for itr in tqdm(count(), position=0, desc='collector'):
        
        [copy_weights(local,shared) for shared,local in zip(model_shared.models.values(),model_local.models.values())]
        
        state = simulator.reset()
                 
        state_processed = np.concatenate((state.front_rgb,state.wrist_rgb),axis=2)
                
        episode_reward = 0
        
        #print("collector: {}".format(next(model_local.models['model'].parameters()).device))
                
        for e in count():
            
            
            action,action_discrete = model_local.get_action(as_tensor(state_processed,device=Device.get_device()),itr,warmup_flag,writer)
                
            # Agent step 
            try:
                next_state, reward, terminal = simulator.step(action)
                
            # Handling failure in planning and wrong action for inverse Jacobian
            except (ConfigurationPathError,InvalidActionError):
                continue
            
            except Exception as e:
                print(e)
                break
            
             
            # Concainating diffrent cameras
            next_state_processed = np.concatenate((next_state.front_rgb,next_state.wrist_rgb),axis=2)
            state_processed = np.concatenate((state.front_rgb,state.wrist_rgb),axis=2)
            
            # Storing the data in the queue
            lock.acquire()
            queue.put(data_to_queue(state_processed, action_discrete, reward, next_state_processed, terminal))
            lock.release()

            
            # Updating running metrics
            episode_reward += reward
            total_reward += reward
            state = next_state
            
            
            # Early termination conditions
            if (terminal or (e>args.episode_length)):
                with flush_flag.get_lock():
                    flush_flag.value = True
                    
                if not warmup_flag.value:
                    beta.value += beta_step
                    beta.value = min(beta.value,1)
                    
                    epsilon -= epsilon_step
                    
                break
        
        # Log the results
        logging.debug('Episode reward: {:.2f}'.format(episode_reward))
        writer.add_scalar('Episode reward', episode_reward, itr)
        writer.add_scalar('Total reward',total_reward,itr)
         
                  
    writer.close()
            
            
    
def optimise(simulator,model_shared,queue,lock,args,flush_flag,warmup_flag,beta):
        
        writer = SummaryWriter('tensorboard/opt')
        
        logging.basicConfig(filename='logs/optimiser.log',
                                filemode='w',
                                format='%(message)s',
                                level=logging.DEBUG)
        
        
        n_gpu = t.cuda.device_count()
        if n_gpu > 0:
            Device.set_device(0 % n_gpu)
        
        model_local = deepcopy(model_shared)
        model_local.to(Device.get_device())
        
        target = deepcopy(model_local)
        
       
        buffer = ReplayBufferDQN(args)
        
        #print("optimiser: {}".format(next(model_local.models['model'].parameters()).device))
        
        for itr in tqdm(count(), position=1, desc='optimiser'):
            
            # Flushing the buffer
            if flush_flag.value:
                with flush_flag.get_lock():
                    flush_flag.value = False
                buffer.memory.on_episode_end()
            
            # Loading the data from the queue
            buffer.load_queue(simulator,queue,lock,args)
            
            # During warmup or whilst the buffer is too small
            if ((len(buffer) < args.batch_size) or warmup_flag.value):
                
                # Loading the data from the queue
                buffer.load_queue(simulator,queue,lock,args)
                
                # Training the accelerator
                if args.accelerator:
                    batch_accelerator = buffer.sample_batch_accelerator(Device.get_device())
                    loss = model_local.calculate_loss_accelerator(batch_accelerator, Device.get_device())
                    model_local.optimisers[1].zero_grad()
                    loss.backward()
                    model_local.optimisers[1].step()
                    
                    # Accelerator logs
                    logging.debug('DQN loss: {:.6f}, Accelerator loss: {:.6f}, Buffer size: {}, Beta value: {:.6f}'.format(0, loss.item(), len(buffer),beta.value))
                    writer.add_scalar('DQN loss', 0,itr)
                    writer.add_scalar('Accelerator loss', loss.item(),itr)
                    writer.add_scalar('Beta value', beta.value,itr)
                    
                # Avoid high number of iteration when training without the accelerator
                time.sleep(1)
                continue
                    
            
                
            # Sample a data point from dataset
            batches = buffer.sample_batch(model_local,target,Device.get_device(),beta.value)
            
            # Calculate loss for the batch
            loss = model_local.calculate_loss(batches,target, Device.get_device())
            
            # Updated the shared model
            optimise_model(model_shared,model_local,loss,lock)
            
            # Copy accelerator weights
            if args.accelerator:
                copy_weights(target.models['accelerator'],model_local.models['accelerator'])
            
            # Calculate the loss values
            loss_model = loss[0].item()
            
            if args.accelerator:
                loss_accelerator = loss[1].item()
            else:
                loss_accelerator = 0
                
              
            # Log the results
            logging.debug('DQN loss: {:.6f}, Accelerator loss: {:.6f}, Buffer size: {}, Beta value: {:.6f}'.format(loss_model, loss_accelerator, len(buffer),beta.value))
            writer.add_scalar('DQN loss', loss_model,itr)
            writer.add_scalar('Accelerator loss', loss_accelerator,itr)
            writer.add_scalar('Beta value', beta.value,itr)
            
                
            if itr % args.target_update_frequency == 0:
                target.models['model'].load_state_dict(deepcopy(model_local.models['model'].state_dict()))

            
        writer.close()
    
        
            
            
    
