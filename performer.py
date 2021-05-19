from copy import deepcopy

import torch as t

from device import Device
from utils import as_tensor

import numpy as np


class Performer():
    
    def __init__(self,idx,model,sim):
        
        self.idx = idx
        self.model = model
        self.simulator = sim(False)
    

    def perform(self,args):
        
        
        # allocate a device
        n_gpu = t.cuda.device_count()
        if n_gpu > 0:
            Device.set_device(1)
        
        q_network = deepcopy(self.model)
        q_network.to(Device.get_device())
        q_network.eval()
        
        num_reached = 0
        
        
        for n in range(args.n_tests):
            
        
            state = self.simulator.reset()
            state_processed = np.concatenate((state.front_rgb,state.wrist_rgb),axis=2)
                
            episode_reward = 0
            
            terminal = False
            
            for i in range(800):
             
                if np.random.RandomState().rand() < 0.1:
                    action = np.random.RandomState().randint(self.simulator.n_actions())
                else:
                    action = q_network(as_tensor(state_processed)).argmax().item()
                             
                    
                next_state, reward, terminal = self.simulator.step(action,state)
                    
                episode_reward += reward
                            
                state_processed = np.concatenate((next_state.front_rgb,next_state.wrist_rgb),axis=2)
                state = next_state
                
                if (terminal):
                    print("\nTrial {} reached the goal!".format(n+1))
                    num_reached += 1
                    break
                	
            print("\nEpisode reward: {}".format(episode_reward))
            
        print("\n\nSuccess rate: {}/{}".format(num_reached,args.n_tests))
                
            
       
    
   
        
