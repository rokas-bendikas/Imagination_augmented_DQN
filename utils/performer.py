import torch as t
import numpy as np
from rlbench.task_environment import InvalidActionError
from pyrep.errors import ConfigurationPathError
from copy import deepcopy
from utils.utils import process_state,rgb_to_grayscale


def perform(NETWORK,simulator,args):


    # allocate a device
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        print("Using CUDA device:")
        print(t.cuda.get_device_name())

    model = NETWORK(args,testing = True)
    model.load()
    model.to(device)
    model.eval()

    # Target network
    target = deepcopy(model)

    num_reached = 0
    num_no_collision = 0

    simulator.launch()

    for n in range(args.n_tests):


        state = simulator.reset()

        state_processed = rgb_to_grayscale(process_state(state,device))

        episode_reward = 0

        terminal = False

        for i in range(args.episode_length):

            # Process the state to fit the network
            state_tensor = t.tensor(state_processed,device=device,dtype=t.float32).unsqueeze(0)

            action,_ = model.get_action(state_tensor,device,target)

            try:
                next_state, reward, terminal = simulator.step(action)

            except (ConfigurationPathError,InvalidActionError):
                next_state = state
                reward = -0.5
                terminal = False


            # Concainating diffrent cameras
            next_state_processed = rgb_to_grayscale(process_state(next_state,device))
            state_processed = rgb_to_grayscale(process_state(state,device))


            episode_reward += reward
            state = next_state

            if (terminal):

                num_reached += 1

                if abs(reward - 5.0) < 1e-5:
                    num_no_collision += 1
                break

        print("\nEpisode {} reward: {}".format(n+1,episode_reward))

    print("\n\nSuccess rate: {}/{}".format(num_reached,args.n_tests))
    print("Number of trials with no collisions: {}/{}".format(num_no_collision,args.n_tests))
