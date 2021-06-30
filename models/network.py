import torch as t
import torch.nn.functional as f
import torch.nn as nn
from models.modules.model_free.network import action_predictor
from models.modules.rollouts.rollout_module import rollout_engine
from models.modules.policy.policy_head import policy_head
from models.base import BaseModel
from utils.utils import plot_data,copy_weights
import numpy as np
from utils.device import Device
import getch



class I2A_model:
    def __init__(self,args,testing=False):
        super(I2A_model,self).__init__()

        # Flags
        self.testing = testing
        self.gripper_open = 1.0

        # Saving args
        self.args = args

        # Models to be used
        self.models = dict()

        # Dicts for modules
        if not self.testing:
            self.optimisers = dict()

        # Initialise model-free network
        self._init_models()


        if self.testing:
            self.eval()


    def __call__(self,x,device):
        out = self.forward(x,device)
        return out



    def forward(self,x,device):

        # Adjust for single-input batch
        if(len(x.shape)==3):
            x = x.unsqueeze(0).permute(0,3,1,2)

        # List of all rollouts
        rollouts = list()

        for action in range(self.args.n_actions):

            # Rollout
            rollouts.append(self.models['rollouts'](x,action,Device.get_device()))


        encoding = self._aggregate_rollouts(rollouts,device)

        Q_values = self.models['policy'](encoding)

        return Q_values

    # Converting network outputs to discrete actions:
    def get_action(self,state,device,itr=None,warmup_flag=None,writer=None,greedy=False):

        ###############################
        ############ DQN ##############
        ###############################

        if greedy:
            eps = max(0,self.args.eps)

        # Epsilon-greedy for DQN
        elif not self.testing:
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
            with t.no_grad():
                action_discrete = self(state,device).argmax().item()

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

        return out





    def get_losses(self,batches,target,device):

        losses = dict()

        # Loss of the policy networks
        for key, value in self._loss_policy_head(batches,target,device).items():
            losses[key] = value

        # Loss of the rollout modules
        for key, value in self._loss_rollout_module(batches,target,device).items():
            losses[key] = value

        # Loss of model-free path
        for key, value in self._loss_action_predictor(batches,device).items():
            losses[key] = value

        return losses



    def _loss_policy_head(self,batches,target,device):

        state, action, reward, next_state, terminal = batches[0]

        loss = dict()

        # Target value
        with t.no_grad():
            target = reward + terminal * self.args.gamma * target(next_state,device).max()

        # Network output
        predicted = self.forward(state,device).gather(1,action)

        loss['policy'] = f.smooth_l1_loss(predicted, target)

        return loss

    def _loss_rollout_module(self,batches,target,device):

        loss = dict()

        for key, value in self.models['rollouts'].get_loss(batches[1],device).items():
            loss[key] = value

        return loss



    def _loss_action_predictor(self,batches,device):

        state, action, _, next_state, _ = batches[0]

        actions = t.zeros_like(action)

        for i in range(actions.shape[0]):
            _, actions[i,0] = self.get_action(state,device,greedy=True)

        actions_d = self.models['action_predictor'](state)

        loss = {'action_predictor': f.cross_entropy(actions_d,actions.squeeze())}

        return loss


    ##############################################################################################
    ######################################## INITS ###############################################
    ##############################################################################################


    def _init_models(self)->None:

        self.models['policy'] = policy_head(self.args)
        self.models['action_predictor'] = action_predictor(self.args)
        self.models['rollouts'] = rollout_engine(self.models['action_predictor'],self.args)

        if not self.testing:

            # Action predictor optimiser
            self.optimisers['action_predictor'] = t.optim.Adam(self.models['action_predictor'].parameters(), lr=1e-6)

            # Envirnment model optimiser
            self.optimisers['environment_model'] = t.optim.Adam(self.models['rollouts'].env_model.parameters(), lr=5e-5)

            # State autoencoder
            self.optimisers['state_autoencoder'] = t.optim.Adam(self.models['rollouts'].state_encoder.parameters(), lr=5e-5)

            # Policy head and transformer head
            params = list(self.models['policy'].parameters()) + list(self.models['rollouts'].lstm.parameters())

            self.optimisers['policy_head'] = t.optim.Adam(self.models['policy'].parameters(), lr=1e-6)

            # Epsilon linear annealing
            self.epsilon_step = self.args.eps/self.args.num_episodes

    ##############################################################################################
    ###################################### General UTILS #########################################
    ##############################################################################################

    def save(self):

        # Save all the model registered
        [self.models[key].save(self.args.save_model+'/{}.pts'.format(key)) for key in self.models]


    def to(self,device):

        # Moving all the models to the device selected
        [m.to(device) for m in self.models.values()]


    def load(self):

        if self.args.load_model != '':

            [self.models[model_name].load(self.args.load_model+model_name+'.pts') for model_name in self.models]
            print("Models were loaded successfully!")

    def eval(self):
        [self.models[model_name].eval() for model_name in self.models]


    def _aggregate_rollouts(self,rollout_list,device):

        rollouts = t.zeros((self.args.batch_size,self.args.n_actions,384),device=device)

        for idx,val in enumerate(rollout_list):
            rollouts[:,idx,:] = val

        rollouts = rollouts.view(self.args.batch_size,-1)

        return rollouts

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


    def _copy_from_model(self,source_model):
        [copy_weights(target,source) for target,source in zip(self.models.values(),source_model.models.values())]
