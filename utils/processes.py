#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:01:18 2021

@author: rokas
"""


import torch as t
from torch.utils.tensorboard import SummaryWriter
from itertools import count
from tqdm import tqdm
import time
import numpy as np
from utils.utils import copy_weights,process_state,rgb_to_grayscale
from utils.replay_buffer import ReplayBufferDQN
from rlbench.task_environment import InvalidActionError
from pyrep.errors import ConfigurationPathError
import sys
from copy import deepcopy


def train_DQN(model_shared,NETWORK,SIMULATOR,args,lock):

    writer = SummaryWriter('tensorboard/col')

    # Determining the processing device
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        print("Using CUDA device:")
        print(t.cuda.get_device_name())

    # Creating a simulator instance
    simulator = SIMULATOR(args.headless)
    simulator.launch()

    # Initialising the model
    model = NETWORK(args)

    # Loading the model
    model.load()

    # Moving to GPU
    model.to(device)

    target = deepcopy(model)

    # Initialising the Replay Buffer
    buffer = ReplayBufferDQN(args)
    minibatch_size = 8

    # Total reward counter
    total_reward = 0

    # Beta for IS  linear annealing
    beta = 0.4
    beta_step = (1 - beta)/args.num_episodes

    # Iterations
    encoder_itr = 0
    training_itr = 0



    # MAIN TRAINING LOOP
    for itr in tqdm(count(), position=0, desc='Epochs'):

        # Determining if its a warmup iteration
        warmup = (itr < args.warmup)

        # Reseting the scene
        state = simulator.reset()

        # Processing state type
        state_processed = rgb_to_grayscale(process_state(state,device))

        ##################################################################
        ####################### EXPLORATION PART #########################
        ##################################################################

        # Episode reward counter
        episode_reward = 0

        # Exploration loop
        for e in count():

            state_tensor = t.tensor(state_processed,device=device,dtype=t.float32).unsqueeze(0)

            # Get the action from the model
            action,action_discrete = model.get_action(state_tensor,device,itr,warmup,writer)

            # Agent step
            try:
                next_state, reward, terminal = simulator.step(action)

            # Handling failure in planning and wrong action for inverse Jacobian
            except (ConfigurationPathError,InvalidActionError):
                next_state = state
                reward = -0.1
                terminal = False

            # Concainating diffrent cameras
            next_state_processed = rgb_to_grayscale(process_state(next_state,device))
            state_processed = rgb_to_grayscale(process_state(state,device))

            # Storing the data in the buffer
            buffer.append([state_processed, action_discrete, reward, next_state_processed, terminal])


            ##################################################################
            ####################### OPTIMISATION PART ########################
            ##################################################################


            # Train autoencoder
            if (e%minibatch_size==0 and (len(buffer) > minibatch_size)):

                    autoencoder_batch = buffer.sample_batch(warmup,device,beta=1,batch_size=minibatch_size, update_weights=False)
                    model.train_autoencoder(autoencoder_batch,writer,encoder_itr)
                    encoder_itr += 1

            if (not warmup and (len(buffer) > args.batch_size) and (e%args.batch_size==0)):

                # Sample a data point from dataset
                batch = buffer.sample_batch(warmup,device,beta,model,target)

                # Calculate loss for the batch
                loss = model.get_losses(batch,target,device)

                # Delete the gradients
                [o.zero_grad() for o in model.optimisers.values()]

                # Compute gradients
                [l.backward() for l in loss.values()]

                # Step in the model
                [o.step() for o in model.optimisers.values()]

                # Log the results
                for key,value in loss.items():
                    if key == 'encoder':
                        writer.add_scalar(key, value.item(),encoder_itr)
                        encoder_itr += 1
                    else:
                        writer.add_scalar(key, value.item(),training_itr)

                training_itr += 1


                if training_itr % args.target_update_frequency == 0:
                    target._copy_from_model(model)


            # Updating running metrics
            episode_reward += reward
            total_reward += reward
            state = next_state

            # Early termination conditions
            if (terminal or (e>args.episode_length)):
                buffer.memory.on_episode_end()
                beta += beta_step
                beta = min(beta,1)
                break

        # Log the results
        writer.add_scalar('Episode reward', episode_reward, itr)
        writer.add_scalar('Total reward',total_reward,itr)


        lock.acquire()
        model_shared._copy_from_model(model)
        lock.release()
