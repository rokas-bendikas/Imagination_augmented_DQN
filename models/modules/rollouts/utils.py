import torch as t
import torch.nn as nn
import torch.nn.functional as f
from models.modules.environment.utils import DoubleConv,Down,Up,OutConv
from utils.utils import as_tensor



class state_autoencoder(nn.Module):
    def __init__(self,args,bilinear=True):
        super().__init__()

        self.args = args

        # Network settings
        self.bilinear = bilinear
        self.factor = 2 if bilinear else 1

        # Init network modules
        self._init_encoder()
        self._init_decoder()

    def forward(self,state):

        if(len(state.shape)==3):
            state = state.unsqueeze(0)

        self.train()

        x1 = self.inc(state)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)

        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        out = self.outc(x)

        return out


    # Predict the next state representation given a single action
    def encode(self,state):


        if(len(state.shape)==3):
            state = state.unsqueeze(0)

        self.eval()

        with t.no_grad():
            x1 = self.inc(state)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x7 = self.down6(x6)
            encoding = x7.flatten(start_dim=1)

        return encoding


    def get_loss(self,batch):

        state, _, _ = batch

        predicted = self(state)

        loss = f.mse_loss(predicted,state)

        return loss



    def _init_encoder(self):

        self.inc = DoubleConv(9,12)
        self.down1 = Down(12, 24)
        self.down2 = Down(24, 48)
        self.down3 = Down(48, 96)
        self.down4 = Down(96, 192)
        self.down5 = Down(192, 384)
        self.down6 = Down(384, 768 // self.factor)


    def _init_decoder(self):
        self.up1 = Up(768, 384 // self.factor, self.bilinear)
        self.up2 = Up(384, 192 // self.factor, self.bilinear)
        self.up3 = Up(192, 96 // self.factor, self.bilinear)
        self.up4 = Up(96, 48 // self.factor, self.bilinear)
        self.up5 = Up(48, 24 // self.factor, self.bilinear)
        self.up6 = Up(24, 12, self.bilinear)
        self.outc = OutConv(12, 9)
