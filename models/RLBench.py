import torch as t
import torch.nn.functional as f
from models.modules.dqn.DQN import DQN_model as DQN
from models.modules.a2c.A2C import A2C_model as A2C
from models.modules.model_accelerator import Accelerator 
from utils.utils import plot_data,copy_weights
import numpy as np
from utils.device import Device
import getch



class RLBench_models:
    def __init__(self,args,testing=False):
        super(RLBench_models,self).__init__()
        
        self.testing = testing
        
        # Saving args
        self.args = args
        
        # Models that are used
        self.models = dict()
        
        if not testing:
            # Used optimisers
            self.optimisers = list()
        
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
    def get_action(self,state,itr=None,warmup_flag=None,writer=None):
        
        ###############################
        ############ DQN ##############
        ###############################
        
        # Epsilon-greedy for DQN
        if self.args.model == "DQN":
            
            if not self.testing:
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
                    
                # Tensorboard logs for epsilon
                writer.add_scalar('Epsilon value',eps,itr)
                    
                    
            else:
                eps = max(0,self.args.eps)
                
            
            # Epsilon-greedy policy
            if np.random.RandomState().rand() < eps:
                action_discrete = np.random.RandomState().randint(self.args.n_actions)
            else:
                action_discrete = self.forward(state).argmax().item()
    
            
            
                     
            """
            inp = getch.getch()
            
            
            action = 0
            
            if inp == 'w':
                action = 0
                
            if inp == 's':
                action = 1
            
            if inp == 'a':
                action = 2
                
            if inp == 'd':
                action = 3
                
            if inp == 'o':
                action = 4
                
            if inp == 'l':
                action = 5
            
            if inp == 'm':
                action = 6
                
            
             
            action_discrete = action
            """
            
            
            # Convert DQN discrete action to continuous
            action = self._action_discrete_to_continous(action_discrete)
            
            
            out = [action,action_discrete]
            
            
        ###############################
        ############ A2C ##############
        ###############################
        
        # Epsilon-greedy for DQN
        elif self.args.model == "A2C":
            
            value,policy = self.forward(state)
            
            value = value.detach().numpy()[0,0]
            dist = policy.detach().numpy() 

            action_discrete = np.random.choice(self.args.n_actions, p=np.squeeze(dist))
            log_prob = t.log(policy.squeeze(0)[action_discrete])
            entropy = -np.sum(np.mean(dist) * np.log(dist))
            
            # Convert discrete action to continuous
            action = self._action_discrete_to_continous(action_discrete)
            
        
            out = [action,action_discrete,log_prob,entropy,value]
               
        return out
            
    
    
    def forward(self,x):
        
        # Adjust for single-input batch
        if(len(x.shape)==3):
            x = x.unsqueeze(0).permute(0,3,1,2)
        
        
        # If using accelerator module
        if self.args.accelerator:
            
            # Accelerator 
            rollout = self.models['accelerator'](x,self.args,Device.get_device())
            out = self.models['model'](x,rollout)
        
        else:
            out = self.models['model'](x) 
        
                     
        return out
    
    
    
    def calculate_loss(self,batches,device,target=None):
        
        losses = list()
        
        # Loss of the main model
        [losses.append(loss) for loss in self._calculate_loss_model(batches[0],device,target)]
        
        # Loss of the accelerator
        if self.args.accelerator:
            losses.append(self._calculate_loss_accelerator(batches[1],device))
        
        return losses
    
    def calculate_loss_accelerator(self,batch,device):
        
        loss = self._calculate_loss_accelerator(batch,device)
        
        return loss
    
    def save(self):
        
        # Save all the model registered
        [self.models[key].save(self.args.save_model+'/{}.pts'.format(key)) for key in self.models]
    
    
    def to(self,device):
        
        # Moving all the models to the device selected
        [m.to(device) for m in self.models.values()]
        
    
    
    def _calculate_loss_model(self, batch,device,target=None):
    
        ###############################
        ############ DQN ##############
        ###############################
        
        if self.args.model == "DQN":
            
            state, action, reward, next_state, terminal = batch
        
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
            
            values,rewards,Qval,log_probs,entropy_term = batch
            
            # compute Q values
            Qvals = np.zeros_like(values)
            for i in reversed(range(len(rewards))):
                Qval = rewards[i] + self.args.gamma * Qval
                Qvals[i] = Qval
                
                
            #update actor critic
            values = t.FloatTensor(values)
            Qvals = t.FloatTensor(Qvals)
            log_probs = t.stack(log_probs)
            
            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term
            
            out = [ac_loss]
            
            
        return out
        

    def _calculate_loss_accelerator(self, batch, device):
        
    
        state, action, next_state = batch
        
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
        self.models['model'] = network.share_memory()
        
        # Flags
        self.gripper_open = 1.0
        
        if not self.testing:
            # Epsilon linear annealing
            self.epsilon_step = self.args.eps/self.args.num_episodes
            
            # Optimiser
            self.optimisers.append(t.optim.Adam(params=self.models['model'].parameters(), lr=self.args.lr))
        
    
    
    ##############################################################################################
    ###################################### A2C UTILS #############################################
    ##############################################################################################
    
    def _init_A2C(self):
        
        network = A2C(self.args)
        self.models['model'] = network.share_memory()
        
        # Flags
        self.gripper_open = 1.0
        
        if not self.testing:
        
            # Optimiser
            self.optimisers.append(t.optim.Adam(params=self.models['model'].parameters(), lr=self.args.lr))
            
    
    
    
    ##############################################################################################
    #################################### Accelerator UTILS #######################################
    ##############################################################################################
    
    
    def _init_Accelerator(self)->None:
        accelerator = Accelerator()
        self.models['accelerator'] = accelerator.share_memory()
        
        if not self.testing:
            self.optimisers.append(t.optim.Adam(params=self.models['accelerator'].parameters(), lr=5e-5))
        
        
        
        
    ##############################################################################################
    ###################################### General UTILS #########################################
    ##############################################################################################
        
    def load(self):
        
        if self.args.load_model != '':
        
            [self.models[model_name].load(self.args.load_model+model_name+'.pts') for model_name in self.models]
            print("Models were loaded successfully!")
            
    def eval(self):
        
        [self.models[model_name].eval() for model_name in self.models]
            
        
    def _action_discrete_to_continous(self,a):
        
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
    
    
    def copy_from_model(self,source_model):
        [copy_weights(target,source) for target,source in zip(self.models.values(),source_model.models.values())]