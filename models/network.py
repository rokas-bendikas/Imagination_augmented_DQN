import torch as t
import torch.nn.functional as f
from models.modules.dqn.DQN import DQN_model as DQN
from models.modules.model_accelerator import Accelerator
from models.modules.policy.policy_head import Policy_Head
from utils.utils import plot_data,copy_weights
import numpy as np
from utils.device import Device
import getch



class I2A_model:
    def __init__(self,args,testing=False):
        super(RLBench_models,self).__init__()

        # Tresting flag
        self.testing = testing

        # Saving args
        self.args = args

        # Models to be used
        self.models = dict()

        # Dicts for modules
        if not self.testing:
            self.optimisers = dict()

        # Initialise model-free network
        self._init_action_predictor()

        # Innitialise rollout engine with policy head
        self._init_rollout_engine()


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

        for action in self.args.n_actions:

            # Rollout
            rollouts.append(self.models['accelerator'](x,action,Device.get_device()))


        encoding = self._aggregate_rollouts(rollouts)

        Q_values = self.policy_head(encoding)


        return Q_values



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
        state, action, reward, next_state, terminal = batch

        # Target value
        with t.no_grad():
            target = reward + terminal * self.args.gamma * target(next_state).max()

        # Network output
        predicted = self.forward(state).gather(1,action)

        out = [f.smooth_l1_loss(predicted, target)]

        return out



    def _calculate_loss_accelerator(self, batch, device):

        state, action, next_state = batch

        predicted = self.models['accelerator'].predict_single_action(state,action,self.args,device)

        if self.args.plot:
            plot_data(batch,predicted)

        loss = f.mse_loss(predicted,next_state)

        return loss


    ##############################################################################################
    ######################################## INITS ###############################################
    ##############################################################################################

    def _init_action_predictor(self):

        self.models['model-free'] = DQN(self.args).share_memory()

        # Flags
        self.gripper_open = 1.0

        if not self.testing:
            # Epsilon linear annealing
            self.epsilon_step = self.args.eps/self.args.num_episodes

            # Optimiser
            self.optimisers['model-free'] = Adam(params=self.models['model-free'].parameters(), lr=self.args.lr))



    def _init_rollut_engine(self)->None:

        self.models['rollouts'] = rollout_module(self.model_free,self.args).share_memory()
        self.models['policy'] = Policy_Head(self.args).share_memory()

        if not self.testing:
            params = list(self.models['rollouts'].parameters()) + list(self.models['policy'].parameters())
            self.optimisers['rollouts'] = t.optim.Adam(params, lr=5e-5))


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

        encoding = "".join(rollouts)

        return encoding


    def copy_from_model(self,source_model):
        [copy_weights(target,source) for target,source in zip(self.models.values(),source_model.models.values())]
