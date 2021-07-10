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
from utils.utils import data_to_queue,copy_weights,plot_data2
from utils.replay_buffer import ReplayBufferDQN
from utils.optimise_model import optimise_model
from rlbench.task_environment import InvalidActionError
from pyrep.errors import ConfigurationPathError

#t.multiprocessing.set_sharing_strategy('file_system')


def collect_DQN(NETWORK,simulator,model_shared,queue,args,flush_flag,warmup_flag,beta,lock):

    # Logging devices
    writer = SummaryWriter('tensorboard/col')


    # Determining the processing device
    n_gpu = t.cuda.device_count()
    if n_gpu > 0:
        Device.set_device(1 % n_gpu)


    # Creating a local model
    model_local = NETWORK(args)

    # Copying the shared model
    #lock.acquire()
    model_local._copy_from_model(model_shared)
    #lock.release()

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
        model_local._copy_from_model(model_shared)
        lock.release()

        # Reseting the scene
        state = simulator.reset()

        # Processing state type
        state_processed = np.concatenate((state.front_rgb,state.left_shoulder_rgb,state.right_shoulder_rgb),axis=2).transpose(2,0,1)

        # Episode reward counter
        episode_reward = 0

        # Count for failed action attempts
        counts_failed = 0

        # Step loop
        for e in count():

            # Get the action from the model
            action,action_discrete = model_local.get_action(t.tensor(state_processed,device=Device.get_device(),dtype=t.float32).unsqueeze(0),Device.get_device(),itr,warmup_flag,writer)

            # Agent step
            try:
                next_state, reward, terminal = simulator.step(action)


            # Handling failure in planning and wrong action for inverse Jacobian
            except (ConfigurationPathError,InvalidActionError):
                # If failed multiple times reseting the environment
                if counts_failed == 4:
                    break
                    print("Breaking!")
                # Otherwise increase the count and try another action
                else:
                    counts_failed += 1
                    continue


            # Reset the failure count
            counts_failed = 0


            # Concainating diffrent cameras
            next_state_processed = np.concatenate((next_state.front_rgb,next_state.left_shoulder_rgb,next_state.right_shoulder_rgb),axis=2).transpose(2,0,1)

            state_processed = np.concatenate((state.front_rgb,state.left_shoulder_rgb,state.right_shoulder_rgb),axis=2).transpose(2,0,1)

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
                print("Terminal!")
                with flush_flag.get_lock():
                    flush_flag.value = True
                if not warmup_flag.value:
                    beta.value += beta_step
                    beta.value = min(beta.value,1)
                    epsilon -= epsilon_step
                break

        # Log the results
        writer.add_scalar('Episode reward', episode_reward, itr)
        writer.add_scalar('Total reward',total_reward,itr)

    writer.close()



def optimise_DQN(NETWORK,model_shared,queue,args,flush_flag,warmup_flag,beta,lock):

    # Logging devices
    writer = SummaryWriter('tensorboard/opt')

    # Determining the processing device
    n_gpu = t.cuda.device_count()
    if n_gpu > 0:
        Device.set_device(0 % n_gpu)

    # Copying the shared model

    model_local = NETWORK(args)





    # Updating local model
    lock.acquire()
    model_local._copy_from_model(model_shared)
    lock.release()



    # Preparing the local model
    model_local.to(Device.get_device())

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


        # During warmup
        if (warmup_flag.value):

            buffer.load_queue_warmup(queue,lock)

            while (len(buffer) <= args.batch_size):

                # Loading the data from the queue
                buffer.load_queue_warmup(queue,lock)

            autoencoder_batch = buffer.sample_batch(warmup_flag.value,Device.get_device(),1)

            model_local.train_autoencoder(autoencoder_batch,writer,itr)
            continue

        # Loading the data from the queue
        buffer.load_queue(queue,lock)

        # Sample a data point from dataset
        batch = buffer.sample_batch(warmup_flag.value,Device.get_device(),beta.value,model_local,target)

        # Calculate loss for the batch
        loss = model_local.get_losses(batch,target,Device.get_device())

        # Updated the shared model
        optimise_model(model_shared,model_local,loss,lock)

        # Log the results
        for key,value in loss.items():
            writer.add_scalar(key, value.item(),itr)


        if itr % args.target_update_frequency == 0:
            target._copy_from_model(model_local)

    writer.close()
