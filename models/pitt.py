import torch
import torch.nn as nn
from torch import Tensor
import yaml
import h5py
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_
from einops import rearrange, repeat, reduce

from .fno import SpectralConv1d, FNO1d, SpectralConv2d_fast, FNO2d

import deepxde as dde
from deepxde.nn.pytorch.fnn import FNN
from deepxde.nn.pytorch.nn import NN
from deepxde.nn import activations

from .oformer import SpatialTemporalEncoder2D, PointWiseDecoder2D
from .oformer import Encoder1D, STDecoder1D, PointWiseDecoder1D
from .deeponet import DeepONet2D


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
            Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# New position encoding module
# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


class LinearAttention(nn.Module):
    """
    Contains following two types of attention, as discussed in "Choose a Transformer: Fourier or Galerkin"

    Galerkin type attention, with instance normalization on Key and Value
    Fourier type attention, with instance normalization on Query and Key
    """
    def __init__(self,
                 input_dim,
                 attn_type,                 # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',    # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1/64,             # 1/64 is for 64 x 64 ns2d,
                 cat_pos=False,
                 pos_dim=2,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inner_dim)
        self.attn_type = attn_type

        self.heads = heads
        self.dim_head = dim_head

        #print("\nINPUT DIM\n")
        self.to_q = nn.Linear(input_dim, inner_dim, bias = False)
        self.to_k = nn.Linear(input_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(input_dim, inner_dim, bias = False)

        if attn_type == 'galerkin':
            self.k_norm = nn.InstanceNorm1d(dim_head)
            self.v_norm = nn.InstanceNorm1d(dim_head)
        elif attn_type == 'fourier':
            self.q_norm = nn.InstanceNorm1d(dim_head)
            self.k_norm = nn.InstanceNorm1d(dim_head)
        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, input_dim),
                #nn.Linear(input_dim, input_dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                #nn.Linear(inner_dim + pos_dim*heads, input_dim),
                nn.Linear(inner_dim, input_dim),
                #nn.Linear(input_dim, input_dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / np.sqrt(dim_head)
            self.diagonal_weight = 1. / np.sqrt(dim_head)
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        #if init_params:
        #    self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            assert not cat_pos
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        #for param in self.to_qkv.parameters():
        for param in self.to_q.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for q
                    init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)

                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                torch.diag(torch.ones(
                                                                                    param.size(-1),
                                                                                    dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)


    def forward(self, queries, keys, values, pos=None, not_assoc=False, padding_mask=None):
        if pos is None and self.relative_emb:
            raise Exception('Must pass in coordinates when under relative position embedding mode')

        queries = self.to_q(queries)
        keys = self.to_k(keys)
        values = self.to_k(values)
        queries, keys, values = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (queries,keys,values))
        if padding_mask is None:
            if self.attn_type == 'galerkin':
                k = self.norm_wrt_domain(keys, self.k_norm)
                v = self.norm_wrt_domain(values, self.v_norm)
            else:  # fourier
                q = self.norm_wrt_domain(queries, self.q_norm)
                k = self.norm_wrt_domain(keys, self.k_norm)
        else:
            grid_size = torch.sum(padding_mask, dim=[-1, -2]).view(-1, 1, 1, 1)  # [b, 1, 1]

            padding_mask = repeat(padding_mask, 'b n d -> (b h) n d', h=self.heads)  # [b, n, 1]

            # currently only support instance norm
            if self.attn_type == 'galerkin':
                k = rearrange(k, 'b h n d -> (b h) n d')
                v = rearrange(v, 'b h n d -> (b h) n d')

                k = masked_instance_norm(k, padding_mask)
                v = masked_instance_norm(v, padding_mask)

                k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)
                v = rearrange(v, '(b h) n d -> b h n d', h=self.heads)
            else:  # fourier
                q = rearrange(q, 'b h n d -> (b h) n d')
                k = rearrange(k, 'b h n d -> (b h) n d')

                q = masked_instance_norm(q, padding_mask)
                k = masked_instance_norm(k, padding_mask)

                q = rearrange(q, '(b h) n d -> b h n d', h=self.heads)
                k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)

            padding_mask = rearrange(padding_mask, '(b h) n d -> b h n d', h=self.heads)  # [b, h, n, 1]


        if self.relative_emb:
            if self.relative_emb_dim == 2:
                freqs_x = self.emb_module.forward(pos[..., 0], x.device)
                freqs_y = self.emb_module.forward(pos[..., 1], x.device)
                freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=q.shape[1])
                freqs_y = repeat(freqs_y, 'b n d -> b h n d', h=q.shape[1])

                q = apply_2d_rotary_pos_emb(q, freqs_x, freqs_y)
                k = apply_2d_rotary_pos_emb(k, freqs_x, freqs_y)
            elif self.relative_emb_dim == 1:
                assert pos.shape[-1] == 1
                freqs = self.emb_module.forward(pos[..., 0], x.device)
                freqs = repeat(freqs, 'b n d -> b h n d', h=q.shape[1])
                q = apply_rotary_pos_emb(q, freqs)
                k = apply_rotary_pos_emb(k, freqs)
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')

        elif self.cat_pos:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.heads, 1, 1])
            q, k, v = [torch.cat([pos, x], dim=-1) for x in (q, k, v)]

        if not_assoc:
            # this is more efficient when n<<c
            score = torch.matmul(q, k.transpose(-1, -2))
            if padding_mask is not None:
                padding_mask = ~padding_mask
                padding_mask_arr = torch.matmul(padding_mask, padding_mask.transpose(-1, -2))  # [b, h, n, n]
                mask_value = 0.
                score = score.masked_fill(padding_mask_arr, mask_value)
                out = torch.matmul(score, v) * (1./grid_size)
            else:
                out = torch.matmul(score, v) * (1./q.shape[2])
        else:
            if padding_mask is not None:
                q = q.masked_fill(~padding_mask, 0)
                k = k.masked_fill(~padding_mask, 0)
                v = v.masked_fill(~padding_mask, 0)
                dots = torch.matmul(k.transpose(-1, -2), v)
                out = torch.matmul(q, dots) * (1. / grid_size)
            else:
                #print(k.shape)
                #print(v.shape)
                #print(queries.shape)
                dots = torch.matmul(keys.transpose(-1, -2), values)
                out = torch.matmul(queries, dots) * (1./queries.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        #print(out.shape)
        #print(self.to_out)
        #return self.to_out(out), None
        #return self.to_out(out), dots
        #return self.to_out(out), dots
        #print(self.to_k.weight)
        return self.to_out(out), self.to_q.weight


class PhysicsInformedTokenTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, neural_operator, dropout=0.1):
        super().__init__()

        self.temp = nn.Linear(100, 100, bias=False)
        self.temp.weight.data.copy_(torch.eye(100))

        # Get input processing
        self.k_embedding_layer = nn.Linear(input_dim, 100, bias=False)
        self.embedding_layer1 = nn.Linear(1, hidden_dim, bias=False)
        self.embedding_layer2 = nn.Linear(1, hidden_dim, bias=False)
        self.embedding_layer3 = nn.Linear(1, hidden_dim, bias=False)

        self.kh1_embedding = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kh2_embedding = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Query and value processing
        self.q_embedding_layer = nn.Linear(1, hidden_dim, bias=False)
        self.v_embedding_layer = nn.Linear(1, hidden_dim, bias=False)
        self.v1_embedding_layer = nn.Linear(1, hidden_dim, bias=False)
        self.v2_embedding_layer = nn.Linear(1, hidden_dim, bias=False)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Internal Physics Model
        self.neural_operator = neural_operator

        # Layers and dropout
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        # Make more layers? Would probably lose any hope of interpretability
        self.mhls = torch.nn.ModuleList()
        self.mhls.append(nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True))

        self.feval_mhls = torch.nn.ModuleList()
        self.norms1 = torch.nn.ModuleList()
        self.norms2 = torch.nn.ModuleList()
        self.t_embeddings = torch.nn.ModuleList()
        self.updates_h = torch.nn.ModuleList()
        for l in range(self.num_layers):

            # For updating state
            self.feval_mhls.append(LinearAttention(input_dim=hidden_dim, attn_type='galerkin',
                                      heads=num_heads, dim_head=hidden_dim, dropout=dropout,
                                      relative_emb=False,
                                      init_method='xavier',
                                      init_gain=1.
            ))

            # Trainable layer norm
            self.norms1.append(nn.LayerNorm(hidden_dim))
            self.norms2.append(nn.LayerNorm(hidden_dim))

            # Embedding time
            self.t_embeddings.append(torch.nn.Linear(1, hidden_dim))

            # NN Update
            self.updates_h.append(nn.Sequential(
                                       nn.Linear(101, hidden_dim),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, 100)
            ))

        self.project_in = nn.Linear(100, 1)

        # Output decoding layer
        self.output_layers = nn.Sequential(
                     nn.Linear(hidden_dim, hidden_dim),
                     nn.GELU(),
                     nn.Dropout(dropout),
                     nn.Linear(hidden_dim, hidden_dim),
                     nn.GELU(),
                     nn.Dropout(dropout),
                     nn.Linear(hidden_dim, output_dim)
        )
        self.act = nn.SiLU()


    def forward(self, queries, keys, values, t):#, mask):

        # Physics Model Forward
        x = torch.swapaxes(values.clone(), 1, 2)
        x = self.neural_operator(x, queries.unsqueeze(-1))[...,0]

        # Get difference so tokens can learn how to update it
        if(len(x.shape) == 2):
            x = x.unsqueeze(-1)
        dx = x - values[:,-1,:].unsqueeze(-1)

        # Scale and shift the keys
        keys = (keys - keys.max())/keys.max()

        # Keys
        keys = self.k_embedding_layer(keys)
        kh1 = self.embedding_layer1(keys.unsqueeze(-1)) #* self.factor
        kh2 = self.embedding_layer2(keys.unsqueeze(-1)) #* self.factor
        kh3 = self.embedding_layer3(keys.unsqueeze(-1)) #* self.factor

        # Process equation
        for l in range(self.num_layers):

            # Attention on just equation tokens
            ah, _ = self.mhls[l](kh1, kh2, kh3)

            #ah = self.norms1[l](ah)
            ah = self.dropout(ah)

            # Copy hidden state to input for next layer
            kh1 = ah.clone()
            kh2 = ah.clone()
            kh3 = ah.clone()

        # Embed Values
        vh = self.v_embedding_layer(dx)

        # Use FNO embedding
        vh_old = vh.clone()

        # Embed time
        t = t.unsqueeze(1).unsqueeze(1)
        t_frac = t/self.num_layers

        # Embed learned tokens
        kh1 = self.kh1_embedding(kh1)
        kh2 = self.kh2_embedding(kh2)

        for l in range(self.num_layers):

            # Calculate update from learned embedding of tokenized equations
            update, _ = self.feval_mhls[l](kh1, kh2, vh_old)
            update = self.dropout(update)

            # Match time embedding with update
            t_h = self.t_embeddings[l](t_frac)
            up_t = torch.swapaxes(torch.cat((update, t_h), dim=1), 1, 2)

            # Calculate time-dependent update
            up_th = torch.swapaxes(self.updates_h[l](up_t), 1, 2)

            # Apply update like numerical method
            vh = vh_old + up_th
            vh_old = vh.clone()

            # Update target time if we're doing multiple layers
            t_frac = t_frac + t/self.num_layers

        out = self.output_layers(vh)[...,0]
        return x[...,0] + out 


class StandardPhysicsInformedTokenTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim, neural_operator, dropout=0.1):
        super().__init__()

        self.temp = nn.Linear(100, 100, bias=False)
        self.temp.weight.data.copy_(torch.eye(100))
        self.hidden_dim = hidden_dim

        # Get input processing
        self.k_embedding_layer = nn.Linear(input_dim, 100, bias=False)
        self.embedding_layer1 = nn.Linear(1, hidden_dim, bias=False)
        self.embedding_layer2 = nn.Linear(1, hidden_dim, bias=False)
        self.embedding_layer3 = nn.Linear(1, hidden_dim, bias=False)

        self.kh1_embedding = nn.Linear(500, 100, bias=False)
        self.kh2_embedding = nn.Linear(500, 100, bias=False)

        # Query and value processing
        self.q_embedding_layer = nn.Linear(1, hidden_dim, bias=False)
        self.v_embedding_layer = nn.Linear(1, hidden_dim, bias=False)
        self.v1_embedding_layer = nn.Linear(1, hidden_dim, bias=False)
        self.v2_embedding_layer = nn.Linear(1, hidden_dim, bias=False)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Encoding/embedding
        self.embedding = torch.nn.Embedding(500, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)

        # Internal Physics Model
        self.neural_operator = neural_operator

        # Layers and dropout
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        # Make more layers? Would probably lose any hope of interpretability
        self.mhls = torch.nn.ModuleList()
        self.mhls.append(nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True))

        self.feval_mhls = torch.nn.ModuleList()
        self.norms1 = torch.nn.ModuleList()
        self.norms2 = torch.nn.ModuleList()
        self.t_embeddings = torch.nn.ModuleList()
        self.updates_h = torch.nn.ModuleList()
        for l in range(self.num_layers):

            # For updating state
            self.feval_mhls.append(LinearAttention(input_dim=hidden_dim, attn_type='galerkin',
                                      heads=num_heads, dim_head=hidden_dim, dropout=dropout,
                                      relative_emb=False,
                                      #relative_emb=True,
                                      init_method='xavier',
                                      init_gain=1.
            ))

            # Trainable layer norm
            self.norms1.append(nn.LayerNorm(hidden_dim))
            self.norms2.append(nn.LayerNorm(hidden_dim))

            # Embedding time
            self.t_embeddings.append(torch.nn.Linear(1, hidden_dim))

            # NN Update
            self.updates_h.append(nn.Sequential(
                                       nn.Linear(101, hidden_dim),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, 100)
            ))

        self.project_in = nn.Linear(100, 1)

        # Output decoding layer
        self.output_layers = nn.Sequential(
                     nn.Linear(hidden_dim, hidden_dim),
                     nn.GELU(),
                     nn.Dropout(dropout),
                     nn.Linear(hidden_dim, hidden_dim),
                     nn.GELU(),
                     nn.Dropout(dropout),
                     nn.Linear(hidden_dim, output_dim)
        )
        self.act = nn.SiLU()


    def forward(self, queries, keys, values, t):#, mask):

        # Physics Model Forward
        x = torch.swapaxes(values.clone(), 1, 2)
        x = self.neural_operator(x, queries.unsqueeze(-1))[...,0]

        # Get difference so tokens can learn how to update it
        if(len(x.shape) == 2):
            x = x.unsqueeze(-1)
        dx = x - values[:,-1,:].unsqueeze(-1)

        # Encoding/Embedding
        keys = self.embedding(keys.long()) * np.sqrt(self.hidden_dim)
        keys = self.pos_encoding(keys)
        kh1 = keys.clone()
        kh2 = keys.clone()
        kh3 = keys.clone()

        # Process equation
        for l in range(self.num_layers):

            # Attention on just equation tokens
            ah, _ = self.mhls[l](kh1, kh2, kh3)

            #ah = self.norms1[l](ah)
            ah = self.dropout(ah)

            # Copy hidden state to input for next layer
            kh1 = ah.clone()
            kh2 = ah.clone()
            kh3 = ah.clone()

        # Embed Values
        vh = self.v_embedding_layer(dx)

        # Use FNO embedding
        vh_old = vh.clone()

        # Embed time
        t = t.unsqueeze(1).unsqueeze(1)
        t_frac = t/self.num_layers

        # Embed learned tokens
        kh1 = torch.swapaxes(self.kh1_embedding(torch.swapaxes(kh1,1,2)),1,2)
        kh2 = torch.swapaxes(self.kh2_embedding(torch.swapaxes(kh2,1,2)),1,2)

        for l in range(self.num_layers):

            # Calculate update from learned embedding of tokenized equations
            update, _ = self.feval_mhls[l](kh1, kh2, vh_old)
            update = self.dropout(update)

            # Match time embedding with update
            t_h = self.t_embeddings[l](t_frac)
            up_t = torch.swapaxes(torch.cat((update, t_h), dim=1), 1, 2)

            # Calculate time-dependent update
            up_th = torch.swapaxes(self.updates_h[l](up_t), 1, 2)

            # Apply update like numerical method
            vh = vh_old + up_th
            vh_old = vh.clone()

            # Update target time if we're doing multiple layers
            t_frac = t_frac + t/self.num_layers

        out = self.output_layers(vh)[...,0]
        return x[...,0] + out


class PhysicsInformedTokenTransformer2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim1, output_dim2, neural_operator, dropout=0.1):
        super().__init__()

        self.temp = nn.Linear(100, 100, bias=False)
        self.temp.weight.data.copy_(torch.eye(100))
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2

        # Get input processing
        self.k_embedding_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.embedding_layer1 = nn.Linear(1, hidden_dim, bias=False)
        self.embedding_layer2 = nn.Linear(1, hidden_dim, bias=False)
        self.embedding_layer3 = nn.Linear(1, hidden_dim, bias=False)

        self.kh1_embedding = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kh2_embedding = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Query and value processing
        self.q_embedding_layer = nn.Linear(1, hidden_dim, bias=False)
        self.v_embedding_layer = nn.Linear(1, hidden_dim, bias=False)
        self.vh_embedding_layer = nn.Linear(output_dim1*output_dim2, hidden_dim, bias=False)
        self.vh_unembedding_layer = nn.Linear(hidden_dim, output_dim1*output_dim2, bias=False)

        self.output_layer = nn.Linear(hidden_dim, output_dim1)

        # Internal Physics Model
        self.neural_operator = neural_operator

        # 
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        # Maybe give the option for multiple layers
        self.mhls = torch.nn.ModuleList()
        self.mhls.append(nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True))

        self.feval_mhls = torch.nn.ModuleList()
        self.t_embeddings = torch.nn.ModuleList()
        self.updates = torch.nn.ModuleList()
        self.updates_h = torch.nn.ModuleList()
        for l in range(self.num_layers):

            # For updating state
            self.feval_mhls.append(LinearAttention(input_dim=hidden_dim, attn_type='galerkin',
                                      heads=num_heads, dim_head=hidden_dim, dropout=dropout,
                                      relative_emb=False,
                                      init_method='xavier',
                                      init_gain=1.
            ))

            # Embedding time
            self.t_embeddings.append(torch.nn.Linear(1, hidden_dim))

            # NN Update
            self.updates.append(nn.Sequential(
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.SiLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.SiLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, hidden_dim)
            ))

            # NN Update
            self.updates_h.append(nn.Sequential(
                                       nn.Linear(hidden_dim+1, hidden_dim),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, hidden_dim)
            ))

        self.project_in = nn.Linear(100, 1)

        # Output decoding layer
        self.output_layers = nn.Sequential(
                     nn.Linear(hidden_dim, hidden_dim),
                     nn.GELU(),
                     nn.Dropout(dropout),
                     nn.Linear(hidden_dim, hidden_dim),
                     nn.GELU(),
                     nn.Dropout(dropout),
                     nn.Linear(hidden_dim, 1)
        )

        # This needs to vary based on other model. Might take out for default training.
        self.act = nn.SiLU()


    def forward(self, queries, keys, values, t):#, mask):

        # Physics Model Forward
        if(isinstance(self.neural_operator, (DeepONet2D, FNO2d))):
            x = self.neural_operator(values, queries)
        else:
            #x = self.neural_operator(values.flatten(1,2), queries.flatten(1,2))
            x = self.neural_operator(values, queries)
        x = x.reshape(x.shape[0], self.output_dim1, self.output_dim2)

        # Get difference between physics model output and input
        dx = x - values[...,-1]
        dx = dx.unsqueeze(-1)

        # Scale and shift the keys
        keys = (keys - keys.max())/keys.max()

        # Keys
        keys = self.k_embedding_layer(keys)
        h1 = self.embedding_layer1(keys.unsqueeze(-1)) 
        h2 = self.embedding_layer2(keys.unsqueeze(-1)) 
        h3 = self.embedding_layer3(keys.unsqueeze(-1))

        # Positional encoding
        kh1 = h1 
        kh2 = h2 
        kh3 = h3 

        # Process equation
        for l in range(1):

            # Attention on just equation tokens
            ah, _ = self.mhls[l](kh1, kh2, kh3)
            ah = self.dropout(ah)

            # Copy hidden state to input for next layer
            kh1 = ah.clone()
            kh2 = ah.clone()
            kh3 = ah.clone()

        # Embed Values
        dx = dx.flatten(1,2)[...,0]
        vh = self.vh_embedding_layer(dx).unsqueeze(-1)
        vh = self.v_embedding_layer(vh)

        # Use FNO embedding
        vh_old = vh.clone()

        # Embed time
        t = t.unsqueeze(1).unsqueeze(1)
        t_frac = t/self.num_layers

        kh1 = self.kh1_embedding(kh1)
        kh2 = self.kh2_embedding(kh2)

        for l in range(self.num_layers):

            update, _ = self.feval_mhls[l](kh1, kh2, vh_old)
            update = self.dropout(update)

            t_h = self.t_embeddings[l](t_frac)
            up_t = torch.swapaxes(torch.cat((update, t_h), dim=1), 1, 2)
            up_th = torch.swapaxes(self.updates_h[l](up_t), 1, 2)

            vh = vh_old + up_th

            vh_old = vh.clone()

            t_frac = t_frac + t/self.num_layers

        vh = torch.swapaxes(self.vh_unembedding_layer(torch.swapaxes(vh, 1, 2)), 1, 2)
        out = self.output_layers(vh)[...,0].reshape((x.shape[0], x.shape[1], x.shape[2]))
        return x + out 


class StandardPhysicsInformedTokenTransformer2D(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim1, output_dim2, neural_operator, dropout=0.1):
        super().__init__()

        self.temp = nn.Linear(100, 100, bias=False)
        self.temp.weight.data.copy_(torch.eye(100))
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.hidden_dim = hidden_dim

        self.embedding = torch.nn.Embedding(100, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)

        # Get input processing
        self.k_embedding_layer = nn.Linear(input_dim, hidden_dim, bias=False)
        self.embedding_layer1 = nn.Linear(1, hidden_dim, bias=False)
        self.embedding_layer2 = nn.Linear(1, hidden_dim, bias=False)
        self.embedding_layer3 = nn.Linear(1, hidden_dim, bias=False)

        self.kh1_embedding = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kh2_embedding = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Query and value processing
        self.q_embedding_layer = nn.Linear(1, hidden_dim, bias=False)
        self.v_embedding_layer = nn.Linear(1, hidden_dim, bias=False)
        self.vh_embedding_layer = nn.Linear(output_dim1*output_dim2, 100, bias=False)
        self.vh_unembedding_layer = nn.Linear(100, output_dim1*output_dim2, bias=False)

        self.output_layer = nn.Linear(hidden_dim, output_dim1)

        # Internal Physics Model
        self.neural_operator = neural_operator

        # Dropout and layer number specification
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

        # Maybe give the option for multiple layers
        self.mhls = torch.nn.ModuleList()
        self.mhls.append(nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True))

        self.feval_mhls = torch.nn.ModuleList()
        self.t_embeddings = torch.nn.ModuleList()
        self.updates = torch.nn.ModuleList()
        self.updates_h = torch.nn.ModuleList()
        for l in range(self.num_layers):

            # For updating state
            self.feval_mhls.append(LinearAttention(input_dim=hidden_dim, attn_type='galerkin',
                                      heads=num_heads, dim_head=hidden_dim, dropout=dropout,
                                      relative_emb=False,
                                      init_method='xavier',
                                      init_gain=1.
            ))

            # Embedding time
            self.t_embeddings.append(torch.nn.Linear(1, hidden_dim))

            # NN Update
            self.updates.append(nn.Sequential(
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.SiLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.SiLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_dim, hidden_dim)
            ))

            # NN Update
            self.updates_h.append(nn.Sequential(
                                       nn.Linear(100+1, 100),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(100, 100),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(100, 100)
            ))

        self.project_in = nn.Linear(100, 1)

        # Output decoding layer
        self.output_layers = nn.Sequential(
                     nn.Linear(hidden_dim, hidden_dim),
                     nn.GELU(),
                     nn.Dropout(dropout),
                     nn.Linear(hidden_dim, hidden_dim),
                     nn.GELU(),
                     nn.Dropout(dropout),
                     nn.Linear(hidden_dim, 1)
        )

        # This needs to vary based on other model. Might take out for default training.
        self.act = nn.SiLU()


    def forward(self, queries, keys, values, t):#, mask):

        # Physics Model Forward
        if(isinstance(self.neural_operator, DeepONet2D)):
            x = self.neural_operator(values, queries)
        else:
            #x = self.neural_operator(values.flatten(1,2), queries.flatten(1,2))
            if(isinstance(self.neural_operator, FNO2d)):
                x = self.neural_operator(values, queries)
            else:
                x = self.neural_operator(values, queries)
                #x = self.neural_operator(values.flatten(1,2), queries.flatten(1,2))
        x = x.reshape(x.shape[0], self.output_dim1, self.output_dim2)

        # Get difference between physics model output and input
        dx = x - values[...,-1]
        dx = dx.unsqueeze(-1)

        # Embedding
        #print(keys.shape)
        keys = self.embedding(keys.long()) * np.sqrt(self.hidden_dim)
        keys = self.pos_encoding(keys)

        ## Scale and shift the keys
        #keys = (keys - keys.max())/keys.max()

        ## Keys
        kh1 = keys.clone()
        kh2 = keys.clone()
        kh3 = keys.clone()

        # Process equation
        for l in range(1):

            # Attention on just equation tokens
            ah, _ = self.mhls[l](kh1, kh2, kh3)
            #return _
            ah = self.dropout(ah)

            # Copy hidden state to input for next layer
            kh1 = ah.clone()
            kh2 = ah.clone()
            kh3 = ah.clone()

        # Embed Values
        #print(dx.shape)
        dx = dx.flatten(1,2)[...,0]
        #print(dx.shape)
        vh = self.vh_embedding_layer(dx).unsqueeze(-1)
        #print(vh.shape)
        vh = self.v_embedding_layer(vh)

        # Use FNO embedding
        vh_old = vh.clone()

        # Embed time
        t = t.unsqueeze(1).unsqueeze(1)
        t_frac = t/self.num_layers

        kh1 = self.kh1_embedding(kh1)
        kh2 = self.kh2_embedding(kh2)

        for l in range(self.num_layers):

            update, _ = self.feval_mhls[l](kh1, kh2, vh_old)
            update = self.dropout(update)

            t_h = self.t_embeddings[l](t_frac)
            up_t = torch.swapaxes(torch.cat((update, t_h), dim=1), 1, 2)
            #print(up_t.shape)
            up_th = torch.swapaxes(self.updates_h[l](up_t), 1, 2)

            #print(vh_old.shape, up_th.shape)
            vh = vh_old + up_th

            vh_old = vh.clone()

            t_frac = t_frac + t/self.num_layers

        vh = torch.swapaxes(self.vh_unembedding_layer(torch.swapaxes(vh, 1, 2)), 1, 2)
        out = self.output_layers(vh)[...,0].reshape((x.shape[0], x.shape[1], x.shape[2]))
        return x + out

