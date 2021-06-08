import torch as t
import torch.nn as nn
import torch.nn.functional as f
from models.modules.DQN import DQN
from models.modules.model_accelerator import Accelerator
from utils.utils import plot_data
import numpy as np
from utils.device import Device



class RLBench_models:
    def __init__(self,args):
        super(RLBench_models,self).__init__()
        
        # Saving args
        self.args = args
        
        # Device to use
        #self.device = device
        
        # Models
        self.models = dict()
        
        # Setting the model
        if self.args.model == "DQN":
            network = DQN(self.args)
            network.load(self.args.load_model)
            #network.to(self.device)
            self.models['model'] = network
            
            # Epsilon linear annealing
            #self.epsilon = self.args.eps
            self.epsilon_step = self.args.eps/self.args.num_episodes
            
            # Flags
            self.gripper_open = 1.0
            
        self.optimisers = [t.optim.SGD(params=self.models['model'].parameters(), lr=args.lr,momentum=0.9)]
        
        # Adding accelerator if function on
        if self.args.accelerator:
            accelerator = Accelerator()
            accelerator.load(self.args.load_model_acc)
            self.models['accelerator'] = accelerator
            
            self.optimisers.append(t.optim.Adam(params=self.models['accelerator'].parameters(), lr=5e-5))
            

    # Forward function for both model and accelerator
    def __call__(self,x):
        
        return self.forward(x)
    
    
    # Converting network outputs to discrete actions:
    def get_action(self,state,itr,simulator,warmup_flag,writer):
        
        # Epsilon-greedy for DQN
        if self.args.model == "DQN":
            
            # During warmup 
            if (itr < self.args.warmup):
                eps = 1
                    
            # After warmup
            else:
                
                # Shared flag for the end of warmup
                if warmup_flag.value:
                    with warmup_flag.get_lock():
                        warmup_flag.value = False  
                        
                # Updating decay parameters
                epsilon = self.args.eps - (itr-self.args.warmup)*self.epsilon_step
                eps = max(epsilon, self.args.min_eps)
                
                        
            # Epsilon-greedy policy
            if np.random.RandomState().rand() < eps:
                action_discrete = np.random.RandomState().randint(simulator.n_actions())
            else:
                action_discrete = self.forward(state).argmax().item()
                
            
            
            # delta orientation
            d_quat = np.array([0, 0, 0, 1])
            
            # delta position
            d_pos = np.zeros(3)
            
            if action_discrete == 6:
                # gripper state
                self.gripper_open = abs(self.gripper_open - 1)
            else:
                # For positive magnitude
                if(action_discrete%2==0):
                    action = int(action_discrete/2)
                    d_pos[action] = 0.02
                    
                # For negative magnitude
                else:
                    action = int((action_discrete-1)/2)
                    d_pos[action] = -0.02
            
            # Forming action as expected by the environment
            action = np.concatenate([d_pos, d_quat, [self.gripper_open]])
                
            writer.add_scalar('Epsilon value',eps,itr)
            
                
        return action,action_discrete
            
    
    
    def forward(self,x):
        
        # Adjust for single-input batch
        if(len(x.shape)==3):
            x = x.unsqueeze(0).permute(0,3,1,2)
        
        # If using DQN model
        if self.args.model == "DQN":
        
            # If using accelerator module
            if self.args.accelerator:
                
                # Accelerator 
                rollout = self.models['accelerator'].rollout(x,self.args,Device.get_device())
                out = self.models['model'](x,rollout)
            
            else:
                out = self.models['model'](x)  
                
        return out
    
    
    
    def calculate_loss(self,batch,target,args,device):
        
        loss_DQN = self._calculate_loss_model(target, batch, args, device)
        
        loss_acc = self._calculate_loss_accelerator(batch, args, device)
        
        return [loss_DQN,loss_acc]
    
    def save(self):
        [self.models[key].save(self.args.save_model+'/{}.pts'.format(key)) for key in self.models]
    
    
    def to(self,device):
        
        [m.to(device) for m in self.models.values()]
        
    
    
    def _calculate_loss_model(self, target, batch, args,device):
    
        state, action, reward, next_state, terminal = batch
        
        # Moving data to gpu for network pass
        state = state.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        terminal = terminal.to(device)
        action = action.to(device)
        
        if self.args.accelerator:
            
            # Target value
            with t.no_grad():
                target = reward + terminal * args.gamma * target(next_state).max()
                
            # Network output    
            predicted = self.forward(state).gather(1,action)
    
        return f.smooth_l1_loss(predicted, target)
        

    def _calculate_loss_accelerator(self, batch, args, device):
    
        state, action, _, next_state, _ = batch
        
        # Moving data to gpu for network pass
        state = state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)
        
        predicted = self.models['accelerator'](state,action,args,device)
        
        if args.plot:
            plot_data(batch,predicted)
        
        loss = f.mse_loss(predicted,next_state)
    
        return loss
    
    def _discrete_to_continous_action(self,a):
        
        # delta orientation
        d_quat = np.array([0, 0, 0, 1])
        
        # delta position
        d_pos = np.zeros(3)
        
        if a == 6:
            # gripper state
            self.gripper_open = abs(self.gripper_open - 1)
        else:
            # For positive magnitude
            if(a%2==0):
                a = int(a/2)
                d_pos[a] = 0.02
                
            # For negative magnitude
            else:
                a = int((a-1)/2)
                d_pos[a] = -0.02
        
        # Forming action as expected by the environment
        action = np.concatenate([d_pos, d_quat, [self.gripper_open]])
        
        return action