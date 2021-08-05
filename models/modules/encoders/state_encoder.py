import torch as t
import torch.nn as nn
import torch.nn.functional as f
from models.modules.encoders.utils import Down
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
                # batch_size x 4 x 64 x 64
                Down(4,32),
                # batch_size x 32 x 32 x 32
                Down(32,32),
                # batch_size x 32 x 16 x 16
                Down(32,32),
                # batch_size x 32 x 8 x 8
                Down(32,32),
                # batch_size x 32 x 4 x 4
                nn.Flatten())
                # batch_size x 512




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
