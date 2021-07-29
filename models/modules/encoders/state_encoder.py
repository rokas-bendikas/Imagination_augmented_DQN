import torch as t
import torch.nn as nn
import torch.nn.functional as f
from models.modules.encoders.utils import DoubleConv,Down,Up,OutConv
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
                # batch_size x 3 x 96 x 96
                DoubleConv(3,9),
                # batch_size x 9 x 96 x 96
                Down(9,18),
                # batch_size x 18 x 48 x 48
                Down(18,36),
                # batch_size x 36 x 24 x 24
                Down(36,72),
                # batch_size x 72 x 12 x 12
                Down(72,144),
                # batch_size x 144 x 6 x 6
                Down(144,288),
                # batch_size x 288 x 3 x 3
                Down(288,576),
                # batch_size x 576 x 1 x 1
                nn.Flatten(),
                # batch_size x 576
                nn.Linear(576,256))
                # batch_size x 256


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

        self.train()

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


        self.eval()

        with t.no_grad():

            out = self.encoder(state)

        return out
