import torch as t
import torch.nn.functional as f
import torch.nn as nn
from models.modules.model_free.network import action_predictor
from models.modules.rollouts.rollout_module import rollout_engine
from models.modules.policy.policy_head import policy_head
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


    # Forward function for both model and accelerator
    def __call__(self,x):

        return self.forward(x)


    # Converting network outputs to discrete actions:
    def get_action(self,state,itr=None,warmup_flag=None,writer=None):

        ###############################
        ############ DQN ##############
        ###############################

        # Epsilon-greedy for DQN
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

        return out



    def forward(self,x):

        # Adjust for single-input batch
        if(len(x.shape)==3):
            x = x.unsqueeze(0).permute(0,3,1,2)

        # List of all rollouts
        rollouts = list()

        for action in range(self.args.n_actions):

            # Rollout
            rollouts.append(self.models['rollouts'](x,action,Device.get_device()))


        encoding = self._aggregate_rollouts(rollouts)

        Q_values = self.policy_head(encoding)


        return Q_values



    def get_losses(self,batches,target,device):

        losses = dict()

        # Loss of the main model
        for key, value in self._loss_rollouts(batches,target).items():
            losses[key] = value

        # Loss of model-free path
        for key, value in self._loss_action_predictor(batches).items():
            losses[key] = value

        return losses


    def save(self):

        # Save all the model registered
        [self.models[key].save(self.args.save_model+'/{}.pts'.format(key)) for key in self.models]


    def to(self,device):

        # Moving all the models to the device selected
        [m.to(device) for m in self.models.values()]



    def _loss_rollouts(self,batches,target):

        state, action, reward, next_state, terminal = batches[0]

        # Target value
        with t.no_grad():
            target = reward + terminal * self.args.gamma * target(next_state).max()

        # Network output
        predicted = self.forward(state).gather(1,action)

        loss = {'policy': f.smooth_l1_loss(predicted, target)}

        loss['environment_model'] = self.models[rollouts].get_loss(batches[1],device)

        return out



    def _loss_action_predictor(self,batches):

        state, action, _, next_state, _ = batches[0]

        predicted = self.models['action_predictor'](state)

        if self.args.plot:
            plot_data(batch,predicted)

        loss = {'action_predictor': f.cross_entropy(predicted,next_state)}

        return loss


    ##############################################################################################
    ######################################## INITS ###############################################
    ##############################################################################################




    def _init_models(self)->None:

        self.models['policy'] = policy_head(self.args).share_memory()
        self.models['action_predictor'] = action_predictor(self.args).share_memory()

        self.models['rollouts'] = rollout_engine(self.models['action_predictor'],self.args)

        if not self.testing:

            # Action predictor optimiser
            self.optimisers['action_predictor'] = t.optim.Adam(params=self.models['action_predictor'].parameters(), lr=5e-5)

            # Envirnment model optimiser
            self.optimisers['environment_model'] = t.optim.Adam(self.models['rollouts'].env_model.parameters(), lr=5e-5)

            # Policy head and transformer head
            params = list(self.models['rollouts'].state_encoder.parameters()) + list(self.models['policy'].parameters())

            self.optimisers['policy_network'] = t.optim.Adam(params, lr=5e-5)

            # Epsilon linear annealing
            self.epsilon_step = self.args.eps/self.args.num_episodes


    ##############################################################################################
    ###################################### General UTILS #########################################
    ##############################################################################################

    def load(self):

        if self.args.load_model != '':

            [self.models[model_name].load(self.args.load_model+model_name+'.pts') for model_name in self.models]
            print("Models were loaded successfully!")

    def eval(self):
        [self.models[model_name].eval() for model_name in self.models]


    def _aggregate_rollouts(self,rollouts):

        print(rollouts.shape)

        encoding = "".join(rollouts)

        return encoding

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
