import torch as t
import torch.nn as nn
import torch.nn.functional as f
from models.modules.encoders.utils import DoubleConv,Down,Up,OutConv
from utils.utils import plot_autoencoder
from models.base import BaseModel

class StateEncoder(BaseModel):
    def __init__(self,args):
        super().__init__()

        self.args = args

        self._init_network()

    def forward(self,state):

        encoding = self.encoder(state)
        batch_decoded = self.decoder(encoding)

        plot_autoencoder(state,batch_decoded)

        return batch_decoded

    def encode(self,batch):


        with t.no_grad():
            encoding = self.encoder(batch)

        return encoding

    def get_loss(self,batch):

        state,_,_,_,_ = batch

        state_autoencoded = self(state)

        loss = f.mse_loss(state_autoencoded,state)

        loss_dict = {'encoder': loss}

        return loss_dict

    def _init_network(self):

        self.encoder = nn.Sequential(
                # batch_size x 9 x 96 x 96
                DoubleConv(9,18),
                # batch_size x 18 x 96 x 96
                Down(18,36),
                # batch_size x 36 x 48 x 48
                Down(36,72),
                # batch_size x 72 x 24 x 24
                Down(72,144),
                # batch_size x 144 x 12 x 12
                Down(144,288),
                # batch_size x 288 x 6 x 6
                Down(288,576),
                # batch_size x 576 x 3 x 3
                Down(576,576),
                # batch_size x 576 x 1 x 1
                nn.Flatten(),
                # batch_size x 576
                nn.Linear(576,256),
                # batch_size x 256
                nn.ReLU())

        self.decoder = nn.Sequential(
                # batch_size x 256
                nn.Linear(256,576),
                # batch_size x 576
                nn.LeakyReLU(0.2),
                nn.Linear(576,1152),
                # batch_size x 1152
                nn.LeakyReLU(0.2),
                nn.Linear(1152,2304),
                # batch_size x 2304
                nn.LeakyReLU(0.2),
                nn.Linear(2304,5184),
                # batch_size x 5184
                nn.LeakyReLU(0.2),
                nn.Unflatten(dim=1,unflattened_size=(576,3,3)),
                # batch_size x 576 x 3 x 3
                Up(576,288),
                # batch_size x 288 x 6 x 6
                Up(288,144),
                # batch_size x 144 x 12 x 12
                Up(144,72),
                # batch_size x 72 x 24 x 24
                Up(72,36),
                # batch_size x 36 x 48 x 48
                Up(36,18),
                # batch_size x 18 x 96 x 96
                OutConv(18,9)
                # batch_size x 9 x 96 x 96
                )
