import torch as t
import torch.nn.functional as f
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from utils.utils import copy_weights
import numpy as np
import getch
import math
from models.modules.encoders.state_encoder import StateEncoder
from models.modules.policy.DQN import DqnModel
from models.modules.imagination.rollout_module import RolloutEngine
from models.modules.policy.policy_head import PolicyHead
import time




class I2A_model:

    """
    A main class of I2A model.

    ...

    Attributes
    ----------
    testing : bool
        Flag to indicate testing stage.
    args : arg_parser
        user defined hyper-parameters.
    models : dict
        Dictionary of all the submodels used.
    optimisers : dict
        Dictionary of all the optimisers.


    Methods
    -------
    forward(x,target,device):
        Forward pass through the network.

    get_action(state,device,target,itr=None,warmup=None,writer=None,greedy=False):
        Returns continuous and discrete predicted action representations for a given state.

    train_warmup(batch,target_model,device):
        Train DQN and dynamics models during warmup.

    get_losses(batch,target,device):
        Returns all the submodule losses for optimisation.

    copy_from_model(source_model):
        Copies all the module weights from the identical source_model.

    share_memory():
        Sets all the modules to share_memory() mode.

    save():
        Saves the model.

    to(device):
        Moves the model to provided device.

    load():
        Loads the seperate modules from args.load_model parameter.

    eval():
        Sets all the modules to eval() mode.

    """


    def __init__(self,args,testing=False):
        super(I2A_model,self).__init__()

        """
        Constructor.

        Parameters
        ----------
            args : arg_parser
                User defined hyper-parameters.
            testing : bool
                Flag is testing mode used.
        """

        # Testing flag
        self.testing = testing

        # Storing args
        self.args = args

        # Dictionary for modules
        self.models = dict()

        # Dictionary for optimisers
        if not self.testing:
            self.optimisers = dict()

        # Initialise the modules
        self._init_models()

        # Set the network to eval() mode
        if self.testing:
            self.eval()

        else:
            self.train()

            # Epsilon single step
            self.eps_step = (1.0 - self.args.min_eps) / self.args.num_episodes




    def forward(self,x,target,device):

        """
        Forward network pass.

        Parameters
        ----------
            x : torch.tensor
                State representation in shape (batch_size,3,96,96)
            target : I2A_model
                Target network, identical to the current one.

        Returns
        -------
            Q_values : torch.tensor
                Q_values in shape (batch_size,num_actions)
        """

        if(len(x.shape)!=4):
            print("Input shape is not correct!")
            raise TypeError

        encoding = self.models['encoder'](x)

        # List of all rollouts
        rollouts = list()

        for action in range(self.args.n_actions):

            action_tensor = t.full((encoding.shape[0],1),action,device=device)

            # Rollout
            rollouts.append(self.models['rollouts'](encoding,target.models['DQN'],action_tensor,device))

        encoded_trajectories = self._aggregate_rollouts(rollouts,encoding,device)

        Q_values = self.models['policy'](encoded_trajectories)

        return Q_values


    def get_action(self,state,device,target,itr=None,warmup=None,writer=None,greedy=False)->list:

        """
        A function returning the best action given the state (works for single-state only).

        Parameters
        ----------
            state : torch.tensor
                State representation in shape (1,3,96,96).

            device : Device
                Device used to store data.

            target : I2A_model
                Target network, identical to the current one.

            itr : int32
                Current iteration value.

            warmup : bool
                Warmup flag.

            writer : writter
                Handle to the logger object.

            greedy : bool
                Greedy action flag.


        Returns
        -------
            [action,action_discrete] : list
                action format np.array(8), action_discrete format torch.tensor(1).
        """


        if ((state.shape[0] != 1) or (len(state.shape) != 4)):
            print("Action is produced per single state representation, shape of 1x3x96x96!")
            raise ValueError

        eps = self._get_eps(greedy,warmup,itr)

        # Tensorboard logs for epsilon
        if itr is not None:
            writer.add_scalar('Epsilon value',eps,itr)



        # Epsilon-greedy policy
        if np.random.rand() < eps:
            action_discrete = np.random.randint(0,self.args.n_actions)
        else:
            with t.no_grad():
                action_discrete = t.argmax(self(state,target,device)).item()



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



        action_discrete = action
        """


        # Convert DQN discrete action to continuous
        action = self._action_discrete_to_continous(action_discrete)

        return [action,action_discrete]


    def train_warmup(self,target_model,batch,writer,itr,device)->None:


        ############################################################
        #################### TRAINING DQN ##########################
        ############################################################

        self.optimisers['DQN'].zero_grad()

        for key, value in self._loss_DQN(batch,target_model).items():
            value.backward()
            writer.add_scalar(key, value.item(),itr)

        self.optimisers['DQN'].step()

        ############################################################
        ################## TRAINING DYNAMICS #######################
        ############################################################

        self.optimisers['rollouts'].zero_grad()

        for key, value in self._loss_rollout_module(batch,target_model,device).items():
            value.backward()
            writer.add_scalar(key, value.item(),itr)

        self.optimisers['rollouts'].step()



    def get_losses(self,batch,target,device)->dict:

        """
        Returns all the submodule losses for optimisation.

        Parameters
        ----------
            batch : list
                Sampled batch from the replay memory.

            target : I2A_model
                Target network, identical to the current one.

            device : Device
                Device used to store data.


        Returns
        -------
            losses : dict
                Dictionary of all the module losses.

        """

        losses = dict()

        # Loss of model-free path
        for key, value in self._loss_DQN(batch,target).items():
            losses[key] = value

        # Loss of the policy networks
        for key, value in self._loss_policy_head(batch,target,device).items():
            losses[key] = value

        # Loss of the rollout modules
        for key, value in self._loss_rollout_module(batch,target,device).items():
            losses[key] = value


        return losses



    ##############################################################################################
    ################################## HELPER FUNCTIONS ##########################################
    ##############################################################################################

    def copy_from_model(self,source_model,tau=1)->None:

        """
        Copies all the module weights from the identical source_model.

        Parameters
        ----------
            source_model : I2A_model
                Source model to copy from.


        Returns
        -------
            None

        """

        [copy_weights(target,source,tau) for target,source in zip(self.models.values(),source_model.models.values())]

    def share_memory(self)->None:

        """
        Sets all the modules to share_memory() mode.

        Parameters
        ----------
            None

        Returns
        -------
            None

        """

        [self.models[key].share_memory() for key in self.models]


    def save(self)->None:

        """
        Saves the model.

        Parameters
        ----------
            None


        Returns
        -------
            None

        """

        # Save all the model registered
        [self.models[key].save(self.args.save_model+'/{}.pts'.format(key)) for key in self.models]


    def to(self,device)->None:

        """
        Moves the model to provided device.

        Parameters
        ----------
            device : Device
                Device used to store data.

        Returns
        -------
            None

        """

        # Moving all the models to the device selected
        [m.to(device) for m in self.models.values()]


    def load(self)->None:

        """
        Loads the seperate modules from args.load_model parameter.

        Parameters
        ----------
            None

        Returns
        -------
            None

        """

        if self.args.load_model != '':

            for model_name in self.models:
                try:
                    self.models[model_name].load(self.args.load_model+model_name+'.pts')
                    print("{} was loaded successfully!".format(model_name))

                except Exception:
                    print("{} could not be loaded!".format(model_name))




    def eval(self)->None:

        """
        Sets all the modules to eval() mode.

        Parameters
        ----------
            None


        Returns
        -------
            None

        """

        [self.models[model_name].eval() for model_name in self.models]


    def train(self)->None:

        """
        Sets all the modules to eval() mode.

        Parameters
        ----------
            None


        Returns
        -------
            None

        """

        [self.models[model_name].train() for model_name in self.models]


    ##############################################################################################
    ######################################## INITS ###############################################
    ##############################################################################################


    def _init_models(self)->None:

        """
        Initialises all the submodules.

        Parameters
        ----------
            None


        Returns
        -------
            None

        """

        self.models['encoder'] = StateEncoder(self.args)
        self.models['DQN'] = DqnModel(self.args)
        self.models['rollouts'] = RolloutEngine(self.args)
        self.models['policy'] = PolicyHead(self.args)

        if not self.testing:

            # DQN optimiser
            self.optimisers['DQN'] = t.optim.RMSprop([
                {'params': self.models['encoder'].parameters()},
                {'params': self.models['DQN'].parameters()}],
                lr=1e-6)

            # Dynamics model optimiser
            self.optimisers['rollouts'] = t.optim.RMSprop(self.models['rollouts'].dynamics_model.parameters(),lr=1e-6)

            # Policy head optimiser
            self.optimisers['policy'] = t.optim.RMSprop([
                {'params': self.models['policy'].parameters()},
                {'params': self.models['rollouts'].lstm.parameters()}],
                lr=1e-6)


    ##############################################################################################
    ########################################## UTILS #############################################
    ##############################################################################################

    def _aggregate_rollouts(self,rollout_list,state,device):

        rollouts = t.zeros((rollout_list[0].shape[1],self.args.n_actions+1,512),device=device)

        for idx,val in enumerate(rollout_list):
            rollouts[:,idx,:] = val

        rollouts[:,-1,:] = state

        rollouts = rollouts.flatten(start_dim=1)

        return rollouts



    def _loss_DQN(self,batch,target_model)->dict:

        """
        Returns DQN loss handle.

        Parameters
        ----------
            batch : torch.tensor
                State representation in shape (batch_size,3,96,96).

            target_model : I2A_model
                Target network, identical to the current one.


        Returns
        -------
            loss : dict
                Loss in the dictionary structure.
        """

        state, action, reward, next_state, terminal, weights = batch

        state_encoded = self.models['encoder'](state)
        next_state_encoded = target_model.models['encoder'].encode(next_state)

        # Target value
        with t.no_grad():
            target = reward + (1 - terminal.int()) * self.args.gamma * target_model.models['DQN'](next_state_encoded).max(dim=1)[0].detach().unsqueeze(1)

        # Network output
        predicted = self.models['DQN'](state_encoded).gather(1,action)

        loss_DQN = f.smooth_l1_loss(predicted, target,reduction='none')

        loss = {'DQN': t.mean(loss_DQN*weights)}

        return loss

    def _loss_rollout_module(self,batch,target,device):

        loss = dict()

        state, action, _, next_state,_,_ = batch

        state = target.models['encoder'].encode(state)
        next_state = target.models['encoder'].encode(next_state)

        batch = (state,action,next_state)

        for key, value in self.models['rollouts'].get_loss(batch,device).items():
            loss[key] = value

        return loss


    def _loss_policy_head(self,batch,target_model,device):

        state, action, reward, next_state, terminal, weights = batch

        # Target value
        with t.no_grad():
            target = reward + (1 - terminal.int()) * self.args.gamma * target_model(next_state,target_model,device).max(dim=1)[0].detach().unsqueeze(1)

        # Network output
        predicted = self(state,target_model,device).gather(1,action)

        loss_policy = f.smooth_l1_loss(predicted,target,reduction='none').squeeze()

        loss  = {'policy': t.mean(loss_policy*weights)}

        return loss


    def __call__(self,x,target_model,device)->t.tensor:

        """
        Makes class callable to forward().

        Parameters
        ----------
            x : torch.tensor
                State representation in shape (batch_size,3,96,96).

            target : I2A_model
                Target network, identical to the current one.

            device : Device
                Device used to store data.


        Returns
        -------
            out : torch.tensor
                Q_values in shape (batch_size,num_actions)
        """

        out = self.forward(x,target_model,device)

        return out


    def _action_discrete_to_continous(self,action_d)->np.ndarray:

        """
        Transforms a discrete action from the network to continouos representation for simualtor.

        Parameters
        ----------
            action_d : t.tensor
                Discrete actions, shaped (batch_size,1)

            device : Device
                Device used to store data.


        Returns
        -------
            out : np.ndarray
                Action representation, format np.array([delta_orientation,delta_position,gripper_closed])

        """


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



    def _get_eps(self,greedy,warmup,itr)->float:

        """
        Acquire epsilon value for epsilon-greedy policy.

        Parameters
        ----------
            greedy : bool
                Greedy action flag.

            warmup : bool
                Warmup flag.

            itr : int
                Current iteration value.


        Returns
        -------
            eps : float
                Epsilon value.

        """

        # Greedy epsilon
        if greedy or self.testing:
            eps = max(0.0,self.args.eps)

        # Fully random
        elif warmup:
            eps = 1.0

        # Decaying epsilon
        else:
            # Updating decay parameters
            epsilon = 1.0 - (itr-self.args.warmup)*self.eps_step
            eps = max(epsilon, self.args.min_eps)

        return eps
