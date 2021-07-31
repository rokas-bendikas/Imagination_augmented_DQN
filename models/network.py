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



    def forward(self,x):

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

        Q_values = self.models['DQN'](encoding)

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
                self.eval()
                action_discrete = t.argmax(self(state)).item()
                self.train()



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

        return [action,action_discrete]


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

        return losses



    ##############################################################################################
    ################################## HELPER FUNCTIONS ##########################################
    ##############################################################################################

    def copy_from_model(self,source_model)->None:

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

        [copy_weights(target,source) for target,source in zip(self.models.values(),source_model.models.values())]

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

        if not self.testing:


            # DQN for the imagination core
            self.optimisers['DQN'] = t.optim.RMSprop([
                {'params': self.models['encoder'].parameters()},
                {'params': self.models['DQN'].parameters()}],
                lr=5e-5)


    ##############################################################################################
    ########################################## UTILS #############################################
    ##############################################################################################



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

        # Target value
        with t.no_grad():
            target = reward + (1 - terminal.int()) * self.args.gamma * target_model(next_state).max(dim=1)[0].detach().unsqueeze(1)

        # Network output
        predicted = self(state).gather(1,action)

        loss_DQN = (predicted - target)**2
        loss = {'DQN': t.mean(loss_DQN*weights)}

        return loss


    def __call__(self,x)->t.tensor:

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

        out = self.forward(x)
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
            epsilon = 1/math.exp((itr-self.args.warmup)/(self.args.num_episodes/3))
            eps = max(epsilon, self.args.min_eps)


        return eps
