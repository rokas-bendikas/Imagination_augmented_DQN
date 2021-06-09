import torch as t
import torch.nn.functional as f
from models.modules.dqn.DQN import DQN_model as DQN
from models.modules.a2c.A2C import A2C_model as A2C
from models.modules.model_accelerator import Accelerator 
from utils.utils import plot_data, as_tensor
import numpy as np
from utils.device import Device



class RLBench_models:
    def __init__(self,args):
        super(RLBench_models,self).__init__()
        
        # Saving args
        self.args = args
        
        # Models that are used
        self.models = dict()
        
        # Used optimisers
        self.optimisers = list()
        
        ###############################
        ############ DQN ##############
        ###############################
        
        # Initialise DQN model
        if self.args.model == "DQN":
            self._init_DQN()
        
        # Initialise A2C model
        elif self.args.model == "A2C":
            self._init_A2C()
            
        else:
            raise ValueError
        
        # Innitialise accelerator 
        if self.args.accelerator:
            self._init_Accelerator()
            
            

    # Forward function for both model and accelerator
    def __call__(self,x):
        
        return self.forward(x)
    
    
    # Converting network outputs to discrete actions:
    def get_action(self,state,itr,simulator,warmup_flag,writer):
        
        ###############################
        ############ DQN ##############
        ###############################
        
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
                action_discrete = as_tensor(np.random.RandomState().randint(simulator.n_actions()),t.long)
            else:
                action_discrete = self.forward(state).argmax()
    
              
            # Convert DQN discrete action to continuous
            action = self._DQN_discrete_to_continous_action(action_discrete)
            
            # Tensorboard logs for epsilon
            writer.add_scalar('Epsilon value',eps,itr)
            
            
        ###############################
        ############ A2C ##############
        ###############################
        
        # Epsilon-greedy for DQN
        elif self.args.model == "A2C":
            raise NotImplementedError
            
                
        return action,action_discrete
            
    
    
    def forward(self,x):
        
        # Adjust for single-input batch
        if(len(x.shape)==3):
            x = x.unsqueeze(0).permute(0,3,1,2)
        
        
        ###############################
        ############ DQN ##############
        ###############################
        
        # If using DQN model
        if self.args.model == "DQN":
        
            # If using accelerator module
            if self.args.accelerator:
                
                # Accelerator 
                rollout = self.models['accelerator'](x,self.args,Device.get_device())
                out = self.models['model'](x,rollout)
            
            else:
                out = self.models['model'](x) 
        
        
        ###############################
        ############ A2C ##############
        ###############################
        
        # Using A2C model
        elif self.args.model == "A2C":
            
            # If using accelerator module
            if self.args.accelerator:
                
                # Accelerator 
                rollout = self.models['accelerator'](x,self.args,Device.get_device())
                out = self.models['model'](x,rollout)
                          
        return out
    
    
    
    def calculate_loss(self,batches,target,device):
        
        losses = list()
        
        # Loss of the main model
        [losses.append(loss) for loss in self._calculate_loss_model(target, batches[0],device)]
        
        # Loss of the accelerator
        if self.args.accelerator:
            losses.append(self._calculate_loss_accelerator(batches[1],device))
        
        return losses
    
    def save(self):
        
        # Save all the model registered
        [self.models[key].save(self.args.save_model+'/{}.pts'.format(key)) for key in self.models]
    
    
    def to(self,device):
        
        # Moving all the models to the device selected
        [m.to(device) for m in self.models.values()]
        
    
    
    def _calculate_loss_model(self, target, batch,device):
    
        state, action, reward, next_state, terminal = batch
        
        # Moving data to gpu for network pass
        state = state.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        terminal = terminal.to(device)
        action = action.to(device)
        
        
        ###############################
        ############ DQN ##############
        ###############################
        
        if self.args.model == "DQN":
        
            # Target value
            with t.no_grad():
                target = reward + terminal * self.args.gamma * target(next_state).max()
                
            # Network output    
            predicted = self.forward(state).gather(1,action)
            
            out = [f.smooth_l1_loss(predicted, target)]
            
            
        ###############################
        ############ A2C ##############
        ###############################
        
        # Using A2C model
        elif self.args.model == "A2C":
            
            policy_state,value_state = self(state)
            policy_next_state,value_next_state = self(next_state)
            
            # Get critic loss
            advantage = reward + (1.0 - terminal) * self.args.gamma * value_next_state - value_state
            critic_loss = advantage.pow(2)
            
            # Get actor loss
            dist = t.distributions.Categorical(probs=policy_state)
            action = dist.sample()
            
            actor_loss = -dist.log_prob(action) * advantage.detach()
            
            out = [critic_loss,actor_loss]
            
            
            
    
        return out
        

    def _calculate_loss_accelerator(self, batch, device):
    
        state, action, next_state = zip(*batch)
        
        state = t.stack(state).permute(0,3,1,2)
        action = t.stack(action)
        next_state = t.stack(next_state).permute(0,3,1,2)
        
        
        predicted = self.models['accelerator'].predict_single_action(state,action,self.args,device)
        
        if self.args.plot:
            plot_data(batch,predicted)
        
        loss = f.mse_loss(predicted,next_state)
    
        return loss
    
    
    ##############################################################################################
    ###################################### DQN UTILS #############################################
    ##############################################################################################
    
    def _init_DQN(self):
        
        network = DQN(self.args)
        network.load(self.args.load_model)
        #network.to(self.device)
        self.models['model'] = network
        
        # Epsilon linear annealing
        #self.epsilon = self.args.eps
        self.epsilon_step = self.args.eps/self.args.num_episodes
        
        # Flags
        self.gripper_open = 1.0
        
        # Optimiser
        self.optimisers.append(t.optim.SGD(params=self.models['model'].parameters(), lr=self.args.lr,momentum=0.9))
        
    
    def _DQN_discrete_to_continous_action(self,a):
        
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
    
    
    ##############################################################################################
    ###################################### A2C UTILS #############################################
    ##############################################################################################
    
    def _init_A2C(self):
        
        network = A2C(self.args)
        network.load(self.args.load_model)
        self.models['model'] = network
        
        # Flags
        self.gripper_open = 1.0
        
        # Optimiser
        self.optimisers.append(t.optim.SGD(params=self.models['model'].parameters(), lr=self.args.lr,momentum=0.9))
        
    
    
    
    ##############################################################################################
    #################################### Accelerator UTILS #######################################
    ##############################################################################################
    
    
    def _init_Accelerator(self)->None:
        accelerator = Accelerator()
        accelerator.load(self.args.load_model_acc)
        self.models['accelerator'] = accelerator
        self.optimisers.append(t.optim.Adam(params=self.models['accelerator'].parameters(), lr=5e-5))