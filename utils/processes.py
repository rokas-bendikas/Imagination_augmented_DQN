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
from utils.utils import as_tensor,data_to_queue,copy_weights,plot_data2
from utils.replay_buffer import ReplayBufferDQN
from utils.optimise_model import optimise_model
from rlbench.task_environment import InvalidActionError
from pyrep.errors import ConfigurationPathError

#t.multiprocessing.set_sharing_strategy('file_system')
        

def collect_DQN(simulator,model_shared,queue,args,flush_flag,warmup_flag,beta,lock):
    
    # Logging devices
    writer = SummaryWriter('tensorboard/col')     
    logging.basicConfig(filename='logs/collector.log',
                                    filemode='w',
                                    format='%(message)s',
                                    level=logging.DEBUG)
          
     
    # Determining the processing device
    n_gpu = t.cuda.device_count()
    if n_gpu > 0:
        Device.set_device(1 % n_gpu)
   
    # Copying the shared model
    lock.acquire()
    model_local = deepcopy(model_shared)
    lock.release()
    
    # Preparing the local model
    model_local.to(Device.get_device())
    
    # Launching the simulator
    simulator.launch()
       
    # Total reward counter
    total_reward = 0
    
    # Beta for IS  linear annealing
    beta.value = 0.4
    beta_step = (1 - beta.value)/args.num_episodes
    
    # Epsilon linear annealing
    epsilon = args.eps
    epsilon_step = args.eps/args.num_episodes
    
    # Episode loop
    for itr in tqdm(count(), position=0, desc='collector'):
        
        # Updating local model
        lock.acquire()
        model_local.copy_from_model(model_shared)
        lock.release()
        
        # Reseting the scene
        state = simulator.reset()
                 
        # Processing state type
        state_processed = np.concatenate((state.front_rgb,state.wrist_rgb),axis=2)
                
        # Episode reward counter
        episode_reward = 0
        
        # Count for failed action attempts
        counts_failed = 0
        
        # Step loop    
        for e in count():
            
            """
            with t.no_grad():
                
                bat = [as_tensor(state_processed,device=Device.get_device()).unsqueeze(0).permute(0,3,1,2),as_tensor(state_processed,device=Device.get_device()).unsqueeze(0).permute(0,3,1,2),as_tensor(state_processed,device=Device.get_device()).unsqueeze(0).permute(0,3,1,2)]
                #print(bat.shape)
                pred = model_local.models['accelerator'].predict_single_action(as_tensor(state_processed,device=Device.get_device()).unsqueeze(0).permute(0,3,1,2),0,args,Device.get_device())
                plot_data2(bat,pred)
            """
            
            # Get the action from the model
            action,action_discrete = model_local.get_action(as_tensor(state_processed,device=Device.get_device()),itr,warmup_flag,writer)  
            
            # Agent step 
            try:
                next_state, reward, terminal = simulator.step(action)
            # Handling failure in planning and wrong action for inverse Jacobian
            except (ConfigurationPathError,InvalidActionError):
                # If failed multiple times reseting the environment
                if counts_failed == 4:
                    break
                # Otherwise increase the count and try another action
                else:
                    counts_failed += 1
                    continue
                break
            
            # Reset the failure count
            counts_failed = 0
            
            """
            except Exception as e:
                print(e)
                break
            """
             
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
            
            
    
def optimise_DQN(model_shared,queue,args,flush_flag,warmup_flag,beta,lock):
    
    # Logging devices
    writer = SummaryWriter('tensorboard/opt')
    logging.basicConfig(filename='logs/optimiser.log',
                            filemode='w',
                            format='%(message)s',
                            level=logging.DEBUG)
    
    # Determining the processing device
    n_gpu = t.cuda.device_count()
    if n_gpu > 0:
        Device.set_device(0 % n_gpu)
       
    # Copying the shared model
    lock.acquire()
    model_local = deepcopy(model_shared)
    lock.release()
    
    # Preparing the local model
    model_local.to(Device.get_device())
    model_local.load()
    
    # Preparing the target network
    target = deepcopy(model_local)
       
    # Preparing the replay buffer
    buffer = ReplayBufferDQN(args)
    
    # Optimisation loop
    for itr in tqdm(count(), position=1, desc='optimiser'):
        
        # Flushing the buffer
        if flush_flag.value:
            with flush_flag.get_lock():
                flush_flag.value = False
            buffer.memory.on_episode_end()
            if args.accelerator:
                buffer.accelerator_memory.on_episode_end()
        
        # Loading the data from the queue
        buffer.load_queue(queue,lock)
        
        # Whilst the buffer is too small
        while (len(buffer) < args.batch_size):
            # Loading the data from the queue
            buffer.load_queue(queue,lock)
            
        # During warmup
        if (warmup_flag.value):
            
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
            #time.sleep(1)
            continue
                 
        # Sample a data point from dataset
        batches = buffer.sample_batch(model_local,target,Device.get_device(),beta.value)
        
        # Calculate loss for the batch
        loss = model_local.calculate_loss(batches,Device.get_device(),target)
        
        # Updated the shared model
        optimise_model(model_shared,model_local,loss,lock)
        
        # Calculate the loss values
        loss_model = loss[0].item()
        
        if args.accelerator:
            loss_accelerator = loss[1].item()
            copy_weights(target.models['accelerator'],model_local.models['accelerator'])
        else:
            loss_accelerator = 0
               
        # Log the results
        logging.debug('DQN loss: {:.6f}, Accelerator loss: {:.6f}, Buffer size: {}, Beta value: {:.6f}'.format(loss_model, loss_accelerator, len(buffer),beta.value))
        writer.add_scalar('DQN loss', loss_model,itr)
        writer.add_scalar('Accelerator loss', loss_accelerator,itr)
        writer.add_scalar('Beta value', beta.value,itr)
          
        if itr % args.target_update_frequency == 0:
            target.copy_from_model(model_local)
                  
    writer.close()
    
     
    
    
def train_A2C(simulator,model_shared,args):
    
    writer = SummaryWriter('tensorboard/A2C')
            
    logging.basicConfig(filename='logs/A2C.log',
                                    filemode='w',
                                    format='%(message)s',
                                    level=logging.DEBUG)
          
            
    n_gpu = t.cuda.device_count()
    if n_gpu > 0:
        Device.set_device(1 % n_gpu)
    
    simulator.launch()
             
    args.lock.acquire()
    model_local = deepcopy(model_shared)
    args.lock.release()
    
    model_local.to(Device.get_device())
    
    
    entropy_term = 0
    
    total_reward = 0

    for itr in tqdm(count(), position=0, desc='collector'):
                
        log_probs = []
        
        values = []
        
        rewards = []
        
        episode_reward = 0
        
        state = simulator.reset()
                 
        state_processed = np.concatenate((state.front_rgb,state.wrist_rgb),axis=2)
        
        for e in range(args.episode_length):
            
            
            action,action_discrete,log_prob,entropy,value = model_local.get_action(as_tensor(state_processed,device=Device.get_device()))
                
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
            
            state_processed = next_state_processed
            
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            
            episode_reward += reward
            
            
            
            # Early termination conditions
            if (terminal or (e == args.episode_length - 1)):
                
                Qval, _ = model_local.models['model'].forward(as_tensor(next_state_processed,device=Device.get_device()))
                Qval = Qval.detach().numpy()[0,0]
                break
        
        total_reward += episode_reward
        
        batch = [values,rewards,Qval,log_probs,entropy_term] 
        
        loss = model_local.calculate_loss([batch],Device.get_device())
        
        
        # Updated the shared model
        optimise_model(model_shared,model_local,loss,args.lock)
        
        # Log the results
        logging.debug('Episode reward: {:.2f}, A2C loss: {:.6f}, Accelerator loss: {:.6f}'.format(episode_reward,loss[0].item(),0))
        writer.add_scalar('Episode reward', episode_reward, itr)
        writer.add_scalar('Total reward',total_reward,itr)
        writer.add_scalar('A2C loss', loss[0].item(),itr)
        writer.add_scalar('Accelerator loss', 0,itr)
        
                  
    writer.close()     
            
    
