import torch as t
import torch.nn as nn
import torch.nn.functional as f
from models.modules.encoders.utils import DoubleConv,Down,Up
from models.base import BaseModel

class StateEncoder(BaseModel):

    """
    State representation encoder, used as first part of DQN.

    ...

    Attributes
    ----------
    args : arg_parser
        user defined hyper-parameters.

    encoder : nn.Sequential
        Encoder network.

    Methods
    -------
    forward(state):
        Forward pass through the network.

    encode(state):
        Encodes the batch of states.

    """

    def __init__(self,args):
        super().__init__()

        """
        Constructor.

        Parameters
        ----------
            args : arg_parser
                User defined hyper-parameters.

            encoder : nn.Sequential
                Encoder network.

        """

        # Hyper parameters
        self.args = args

        # The encoder network
        self.encoder = nn.Sequential(
                # batch_size x 4 x 128 x 128
                DoubleConv(4,8),
                # batch_size x 8 x 128 x 128
                Down(8,16),
                # batch_size x 16 x 64 x 64
                Down(16,32),
                # batch_size x 32 x 32 x 32
                Down(32,64),
                # batch_size x 64 x 16 x 16
                Down(64,128),
                # batch_size x 128 x 8 x 8
                Down(128,256),
                # batch_size x 256 x 4 x 4
                Down(256,512),
                # batch_size x 512 x 2 x 2
                nn.Conv2d(512, 1024, kernel_size=2, padding=0),
                # batch_size x 1024 x 1 x 1
                nn.Flatten())
                # batch_size x 1024




    def forward(self,state)->t.tensor:

        """
        Forward network pass.

        Parameters
        ----------
            state : torch.tensor
                Encoded state representation in shape (batch_size,256)

        Returns
        -------
            out : torch.tensor
                An encoded shape representation in shape (batch_size,256)
        """

        out = self.encoder(state)

        return out


    def encode(self,state)->t.tensor:

        """
        Forward network pass with no gradient collection.

        Parameters
        ----------
            state : torch.tensor
                Encoded state representation in shape (batch_size,256)

        Returns
        -------
            out : torch.tensor
                An encoded shape representation in shape (batch_size,256)
        """


        with t.no_grad():

            out = self.encoder(state)

        return out
