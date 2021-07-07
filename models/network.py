import torch as t
import torch.nn.functional as f
import torch.nn as nn
from models.modules.encoders.state_encoder import StateEncoder
from models.modules.policy.distiller import ActionDistiller
from models.modules.imagination.rollout_module import RolloutEngine
from models.modules.policy.policy_head import PolicyHead
from models.base import BaseModel
from utils.utils import copy_weights
import numpy as np
from utils.device import Device
import getch



class I2A_model:
    def __init__(self,args,testing=False):
        super(I2A_model,self).__init__()

        # Flags
        self.testing = testing

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

        if(len(x.shape)!=4):
            print("Input shape is not correct!")
            raise TypeError

        state = self.models['encoder'].encode(x)

        # List of all rollouts
        rollouts = list()

        for action in range(self.args.n_actions):

            action_tensor = t.full((state.shape[0],1),action,device=device)

            # Rollout
            rollouts.append(self.models['rollouts'](state,action_tensor,Device.get_device()))


        encoding = self._aggregate_rollouts(rollouts,device)

        Q_values = self.models['policy'](encoding)

        return Q_values

    # Converting netwomse_rk outputs to discrete actions:
    def get_action(self,state,device,itr=None,warmup_flag=None,writer=None,greedy=False):

        ###############################
        ############ DQN ##############
        ###############################

        if ((state.shape[0] != 1) or (len(state.shape) != 4)):
            print("Action is produced per single state representation, shape of 1x9x96x96!")
            raise ValueError

        eps = self._get_eps(greedy,warmup_flag,itr)

        # Tensorboard logs for epsilon
        if itr is not None:
            writer.add_scalar('Epsilon value',eps,itr)

        # Epsilon-greedy policy
        if np.random.RandomState().rand() < eps:
            action_discrete = np.random.RandomState().randint(0,self.args.n_actions)
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

    def train_autoencoder(self,batch,writer,itr)->None:


        loss = self.models['encoder'].get_loss(batch)['encoder']

        self.optimisers['encoder'].zero_grad()

        loss.backward()

        self.optimisers['encoder'].step()

        writer.add_scalar('encoder', loss.item(),itr)



    def get_losses(self,batch,target,device):

        losses = dict()

        # Loss of the state encoder
        for key, value in self._loss_encoder(batch,target,device).items():
            losses[key] = value

        # Loss of the policy networks
        for key, value in self._loss_policy_head(batch,target,device).items():
            losses[key] = value

        # Loss of the rollout modules
        for key, value in self._loss_rollout_module(batch,target,device).items():
            losses[key] = value

        # Loss of model-free path
        for key, value in self._loss_distiller(batch,device).items():
            losses[key] = value

        return losses

    def _loss_encoder(self,batch,target,device):

        loss = dict()

        for key, value in self.models['encoder'].get_loss(batches).items():
            loss[key] = value

        return loss



    def _loss_policy_head(self,batch,target,device):

        state, action, reward, next_state, terminal = batch

        loss = dict()

        # Target value
        with t.no_grad():
            target = reward + terminal * self.args.gamma * target(next_state,device).max()

        # Network output
        predicted = self(state,device).gather(1,action)

        loss['policy'] = f.smooth_l1_loss(predicted, target)



        return loss

    def _loss_rollout_module(self,batch,target,device):

        loss = dict()

        state, action, _, next_state,_ = batch

        state = self.models['encoder'].encode(state)
        next_state = self.models['encoder'].encode(next_state)

        batch = (state,action,next_state)

        for key, value in self.models['rollouts'].get_loss(batch,device).items():
            loss[key] = value

        return loss



    def _loss_distiller(self,batch,device):

        state, action, _, _, _ = batch

        model_actions = t.zeros_like(action)

        for i in range(state.shape[0]):

            _,predicted_action = self.get_action(state[i,:,:,:].unsqueeze(0),device,greedy=True)

            model_actions[i] = predicted_action

        distiller_actions = self.models['distiller'](self.models['encoder'].encode(state))

        loss = {'distiller': f.cross_entropy(distiller_actions,model_actions.squeeze())}

        return loss


    ##############################################################################################
    ######################################## INITS ###############################################
    ##############################################################################################


    def _init_models(self)->None:

        self.models['encoder'] = StateEncoder(self.args)
        self.models['policy'] = PolicyHead(self.args)
        self.models['distiller'] = ActionDistiller(self.args)
        self.models['rollouts'] = RolloutEngine(self.models['distiller'],self.args)

        if not self.testing:

            # State autoencoder optimiser
            self.optimisers['encoder'] = t.optim.Adam(self.models['encoder'].parameters(), lr=1e-5)

            # Action predictor optimiser
            self.optimisers['distiller'] = t.optim.Adam(self.models['distiller'].parameters(), lr=1e-6)

            # Envirnment model optimiser
            self.optimisers['dynamics'] = t.optim.Adam(self.models['rollouts'].dynamics_model.parameters(), lr=5e-5)

            # Policy head and transformer head
            params = list(self.models['policy'].parameters()) + list(self.models['rollouts'].lstm.parameters())

            self.optimisers['policy'] = t.optim.Adam(self.models['policy'].parameters(), lr=1e-6)

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

        rollouts = t.zeros((self.args.batch_size,self.args.n_actions,256),device=device)

        for idx,val in enumerate(rollout_list):
            rollouts[:,idx,:] = val

        rollouts = rollouts.view(self.args.batch_size,-1)

        return rollouts

    def _action_discrete_to_continous(self,action_d):


        # delta orientation
        d_quat = np.array([0, 0, 0, 1])

        # delta position
        d_pos = np.zeros(3)

        # For positive magnitude
        if(action_d%2==0):
            a = int(action_d/2)
            d_pos[a] = 0.02

        # For negative magnitude
        else:
            a = int((action_d-1)/2)
            d_pos[a] = -0.02

        # Forming action as expected by the environment
        action_c = np.concatenate([d_pos, d_quat, [0]])

        return action_c


    def _copy_from_model(self,source_model):
        [copy_weights(target,source) for target,source in zip(self.models.values(),source_model.models.values())]

    def _get_eps(self,greedy,warmup_flag,itr):

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


        else:
            eps = max(0,self.args.eps)


        return eps
