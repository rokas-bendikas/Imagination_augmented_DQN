import torch as t
import torch.nn as nn
from models.modules.imagination.imagination_core import Imagination_Core
from models.modules.environment.EM import environment_model
from models.modules.rollouts.utils import state_autoencoder


class rollout_engine(nn.Module):
    def __init__(self,main_model,args):
        super().__init__()

        # System parameters
        self.args = args

        # Model-free network
        self.model_free = main_model

        # Environment model
        self.env_model = environment_model(args).share_memory()

        # Imagination core
        self.imagination_core = Imagination_Core(self.env_model,self.model_free).share_memory()

        # State encoder
        self.state_encoder = state_autoencoder(self.args)

        # LSTM
        self.lstm = nn.LSTM(384, 384)

        # Transformer
        #self.transformer = nn.Transformer(batch_first=True).share_memory()

    def forward(self,state,action,device):

        encodings = t.zeros((self.args.num_rollouts,self.args.batch_size,384),device=device)

        for i in range(self.args.num_rollouts):

            # Predicting successor state and reward
            if i == 0:
                next_state = self.imagination_core(state,self.args,device,action)
            else:
                next_state= self.imagination_core(state,self.args,device)

            # Encoding state representation
            encoding = self.state_encoder.encode(next_state)

            # Adding encoding to the encoding to the list
            encodings[i,:,:] = encoding

            # Setting state value for the next rollout
            state = next_state

        ############ Transformer ##############
        h0 = t.zeros((1,self.args.batch_size,384),device=device)
        c0 = t.zeros((1,self.args.batch_size,384),device=device)

        self.lstm.flatten_parameters()
        _, (hn,_) = self.lstm(encodings, (h0, c0))

        ########################################

        return hn

    def get_loss(self,batch,device):

        loss = dict()

        # Environmental model loss
        loss['environemnt_model'] = self.env_model.get_loss(batch,device)

        # State autoencoder loss
        loss['state_autoencoder'] = self.state_encoder.get_loss(batch)


        return loss
