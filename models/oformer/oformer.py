import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from .attention_module import pair, PreNorm, PostNorm,\
    StandardAttention, FeedForward, LinearAttention, ReLUFeedForward
from .cnn_module import PeriodicConv2d, PeriodicConv3d, UpBlock

from .encoder_module import Encoder1D
from .decoder_module import PointWiseDecoder1D


class OFormer1D(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self._ssl = False

    def forward(self, x, input_pos):
        h = self.encoder(x, input_pos)
        if(self._ssl):
            return h
        else:
            out = self.decoder(h, propagate_pos=input_pos, input_pos=input_pos) 
            #out = self.decoder.rollout(h, propagate_pos=input_pos, input_pos=input_pos, forward_steps=1) # This might be wrong.
            return out

    def get_loss(self, x, y, input_pos, loss_fn):
        y_pred = self.forward(x, input_pos)[...,0]
        return y_pred, loss_fn(y_pred, y)

    def ssl(self):
        self._ssl = True


class OFormer2D(nn.Module):
    def __init__(self, encoder, decoder, num_x=60, num_y=100):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.num_x = num_x
        self.num_y = num_y
        self._ssl = False

    def forward(self, x, input_pos):
        #x = x.flatten(1,2)
        #input_pos = input_pos.flatten(1,2)
        if(len(x.shape) == 4):
            x = x.flatten(1,2)
        if(len(input_pos.shape) == 4):
            input_pos = input_pos.flatten(1,2)

        h = self.encoder(x, input_pos)
        if(self._ssl):
            return h
        else:
            out, _ = self.decoder(h, propagate_pos=input_pos, input_pos=input_pos)
            #out = self.decoder.rollout(h, propagate_pos=input_pos, input_pos=input_pos, forward_steps=1)
            out = out.reshape(out.shape[0], self.num_x, self.num_y)
            if(len(out.shape) == 4):
                return out[...,0]
            else:
                return out

    def get_loss(self, x, y, input_pos, loss_fn):
        y_pred = self.forward(x, input_pos)
        #print(y_pred.shape)
        #print(y.shape)
        return y_pred, loss_fn(y_pred, y)

    def ssl(self):
        self._ssl = True


