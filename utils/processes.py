#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:01:18 2021

@author: rokas
"""


import torch as t
from torch.nn.utils import clip_grad_norm_
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


def train_DQN(model_shared,NETWORK,SIMULATOR,args,lock)->None:

    '''
    Main A2I training function.

            Parameters:
                    model_shared: CPU based shared model, used to save the parameters.
                    NETWORK: Class definition to initialise local models.
                    SIMULATOR: Class definition to initialise the simulator interface.
                    args: used defined hyper-parameters.
                    lock: lock handle.

    '''



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

    # Target network
    target = deepcopy(model)

    # Initialising the Replay Buffer
    buffer = ReplayBufferDQN(args)

    # Total reward counter
    total_reward = 0

    # Beta for IS  linear annealing
    beta = 0.4
    beta_step = (1.0 - beta) / args.num_episodes

    # Iterations
    training_itr = 0
    warmup_itr = 0


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
            action,action_discrete = model.get_action(state_tensor,device,target,itr,warmup,writer)

            # Agent step
            try:
                next_state, reward, terminal = simulator.step(action)

            # Handling failure in planning and wrong action for inverse Jacobian
            except (ConfigurationPathError,InvalidActionError):
                next_state = state
                reward = -0.5
                terminal = False

            # Concainating diffrent cameras
            next_state_processed = rgb_to_grayscale(process_state(next_state,device))
            state_processed = rgb_to_grayscale(process_state(state,device))

            # Storing the data in the buffer
            buffer.append([state_processed, action_discrete, reward, next_state_processed, terminal])


            # Updating running metrics
            episode_reward += reward
            total_reward += reward
            state = next_state

            # Early termination conditions
            if (terminal or (e>args.episode_length)):
                buffer.memory.on_episode_end()
                if not warmup:
                    beta += beta_step
                    beta = min(1,beta)
                break

            ##################################################################
            ####################### OPTIMISATION PART ########################
            ##################################################################

            # Train autoencoder
            if (len(buffer) > args.batch_size):


                # Main training procedure
                if not warmup:
                    # Sample a data point from dataset
                    batch = buffer.sample_batch(device,beta,model,target,warmup)

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
                            if key == 'DQN':
                                writer.add_scalar(key, value.item(),warmup_itr)
                            elif key == 'dynamics':
                                writer.add_scalar(key, value.item(),warmup_itr)
                            else:
                                writer.add_scalar(key, value.item(),training_itr)

                    training_itr += 1
                    warmup_itr += 1

                # Training some modules during warmup
                else:

                    if itr >= args.warmup/2:

                        # Sample a data point from dataset
                        batch = buffer.sample_batch(device,beta,model,target,warmup)

                        # Train model
                        model.train_warmup(target,batch,writer,warmup_itr,device)

                        warmup_itr += 1

                target.copy_from_model(model,tau=0.0001)

        # Log the results
        writer.add_scalar('Episode reward', episode_reward, itr)
        writer.add_scalar('Total reward',total_reward,itr)
        writer.add_scalar('Beta', beta, itr)


        lock.acquire()
        model_shared.copy_from_model(model)
        lock.release()
