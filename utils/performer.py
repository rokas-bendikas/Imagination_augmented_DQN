import torch as t
from utils.device import Device
from utils.utils import as_tensor,plot_data2
import numpy as np
from rlbench.task_environment import InvalidActionError
from pyrep.errors import ConfigurationPathError



def perform(NETWORK,simulator,args):
    
    
    # allocate a device
    n_gpu = t.cuda.device_count()
    if n_gpu > 0:
        Device.set_device(0)
    
    model = NETWORK(args,testing = True)
    model.load()
    model.to(Device.get_device())
    model.eval()
    
    num_reached = 0
    
    simulator.launch()
    
    for n in range(args.n_tests):
        
    
        state = simulator.reset()
        state_processed = np.concatenate((state.front_rgb,state.wrist_rgb),axis=2)
            
        episode_reward = 0
        
        terminal = False
        
        for i in range(args.episode_length):
         
            bat = [as_tensor(state_processed,device=Device.get_device()).unsqueeze(0).permute(0,3,1,2),as_tensor(state_processed,device=Device.get_device()).unsqueeze(0).permute(0,3,1,2),as_tensor(state_processed,device=Device.get_device()).unsqueeze(0).permute(0,3,1,2)]
            #print(bat.shape)
            pred = model.models['accelerator'].predict_single_action(as_tensor(state_processed,device=Device.get_device()).unsqueeze(0).permute(0,3,1,2),0,args,Device.get_device())
            plot_data2(bat,pred)
    
            action,_ = model.get_action(as_tensor(state_processed,device=Device.get_device()))
            
            try:
                next_state, reward, terminal = simulator.step(action)
            except (ConfigurationPathError,InvalidActionError):
                continue
            
            episode_reward += reward
                        
            state_processed = np.concatenate((next_state.front_rgb,next_state.wrist_rgb),axis=2)
            state = next_state
            
            if (terminal):
                print("\nTrial {} reached the goal!".format(n+1))
                num_reached += 1
                break
            	
        print("\nEpisode reward: {}".format(episode_reward))
        
    print("\n\nSuccess rate: {}/{}".format(num_reached,args.n_tests))
        
    
        
                
            
       
    
   
        
