import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import numpy as np
from .attention_module import PreNorm, PostNorm, LinearAttention, CrossLinearAttention,\
    FeedForward, GeGELU, ProjDotProduct
from .cnn_module import UpBlock, PeriodicConv2d
from torch.nn.init import xavier_uniform_, orthogonal_
from copy import deepcopy

def copy_weights(model1,
                 model2
                 ):
    # copy weights of model2 to model1
    # two models need to be exactly the same
    for param_1, param_2 in zip(model1.parameters(), model2.parameters()):
        param_1.data = deepcopy(param_2.data)


def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)



class AttentionPropagator2D(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 attn_type,         # ['none', 'galerkin', 'fourier']
                 mlp_dim,
                 scale,
                 use_ln=True,
                 dropout=0.):
        super().__init__()
        assert attn_type in ['none', 'galerkin', 'fourier']
        self.layers = nn.ModuleList([])

        self.attn_type = attn_type
        self.use_ln = use_ln
        for d in range(depth):
            attn_module = LinearAttention(dim, attn_type,
                                          heads=heads, dim_head=dim_head, dropout=dropout,
                                          relative_emb=True, scale=scale,
                                          relative_emb_dim=2,
                                          min_freq=1/64,
                                          init_method='orthogonal'
                                          )
            if use_ln:
                self.layers.append(
                    nn.ModuleList([
                        nn.LayerNorm(dim),
                        attn_module,
                        nn.LayerNorm(dim),
                        nn.Linear(dim+2, dim),
                        FeedForward(dim, mlp_dim, dropout=dropout)
                    ]),
                )
            else:
                self.layers.append(
                    nn.ModuleList([
                        attn_module,
                        nn.Linear(dim + 2, dim),
                        FeedForward(dim, mlp_dim, dropout=dropout)
                    ]),
                )

    def forward(self, x, pos):
        for layer_no, attn_layer in enumerate(self.layers):
            if self.use_ln:
                [ln1, attn, ln2, proj, ffn] = attn_layer
                x = attn(ln1(x), pos) + x
                x = ffn(
                    proj(torch.cat((ln2(x), pos), dim=-1))
                        ) + x
            else:
                [attn, proj, ffn] = attn_layer
                x = attn(x, pos) + x
                x = ffn(
                    proj(torch.cat((x, pos), dim=-1))
                        ) + x
        return x


class AttentionPropagator1D(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 attn_type,         # ['none', 'galerkin', 'fourier']
                 mlp_dim,
                 scale,
                 res,
                 dropout=0.):
        super().__init__()
        assert attn_type in ['none', 'galerkin', 'fourier']
        self.layers = nn.ModuleList([])

        self.attn_type = attn_type

        for d in range(depth):
            attn_module = LinearAttention(dim, attn_type,
                                          heads=heads, dim_head=dim_head, dropout=dropout,
                                          relative_emb=True,
                                          scale=scale,
                                          relative_emb_dim=1,
                                          min_freq=1 / res,
                                          )
            self.layers.append(
                nn.ModuleList([
                    attn_module,
                    FeedForward(dim, mlp_dim, dropout=dropout)
                ]),
            )

    def forward(self, x, pos):
        for layer_no, attn_layer in enumerate(self.layers):
            [attn, ffn] = attn_layer

            x = attn(x, pos) + x
            x = ffn(x) + x
        return x


class FourierPropagator(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 mode):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.latent_channels = dim

        for d in range(depth):
            self.layers.append(nn.Sequential(FourierConv2d(self.latent_channels, self.latent_channels,
                                                           mode, mode), nn.GELU()))

    def forward(self, z):
        for layer, f_conv in enumerate(self.layers):
            z = f_conv(z) + z
        return z


# sin activation

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

# siren layer

class Siren(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 w0=1.,
                 c=6.,
                 is_first=False,
                 use_bias=True,
                 activation=None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (np.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out

# siren network
class SirenNet(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_hidden, dim_out, num_layers,
                 w0=1.,
                 w0_initial=30.,
                 use_bias=True, final_activation=None,
                 normalize_input=True):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.normalize_input = normalize_input

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first,
            ))

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in=dim_hidden,
                                dim_out=dim_out,
                                w0=w0,
                                use_bias=use_bias,
                                activation=final_activation)

        # self.last_layer = nn.Linear(dim_hidden, dim_out)
        # init last layer orthogonally
        # nn.init.orthogonal_(self.last_layer.weight, gain=1/dim_out)

    def in_norm(self, x):
        return (2 * x - torch.min(x, dim=1, keepdim=True)[0] - torch.max(x, dim=1, keepdim=True)[0]) /\
            (torch.max(x, dim=1, keepdim=True)[0] - torch.min(x, dim=1, keepdim=True)[0])

    def forward(self, x, mods=None):
        if self.normalize_input:
            x = self.in_norm(x)
        # x = (x - 0.5) * 2

        for layer in self.layers:
            x = layer(x)
        if mods is not None:
            x *= mods
        x = self.last_layer(x)
        # x = self.final_activation(x)
        return x


class MLPPropagator(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.latent_channels = dim

        for d in range(depth):
            layer = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(dim),
            )
            self.layers.append(layer)

    def forward(self, z):
        for layer, ffn in enumerate(self.layers):
            z = ffn(z) + z
        return z


class PointWiseMLPPropagator(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.latent_channels = dim

        for d in range(depth):
            if d == 0:
                layer = nn.Sequential(
                    nn.InstanceNorm1d(dim + 2),
                    nn.Linear(dim + 2, dim, bias=False),  # for position
                    nn.GELU(),
                    nn.Linear(dim, dim, bias=False),
                    nn.GELU(),
                    nn.Linear(dim, dim, bias=False),
                )
            else:
                layer = nn.Sequential(
                    nn.InstanceNorm1d(dim),
                    nn.Linear(dim, dim, bias=False),
                    nn.GELU(),
                    nn.Linear(dim, dim, bias=False),
                    nn.GELU(),
                    nn.Linear(dim, dim, bias=False),
                )
            self.layers.append(layer)

    def forward(self, z, pos):
        for layer, ffn in enumerate(self.layers):
            if layer == 0:
                z = ffn(torch.cat((z, pos), dim=-1)) + z
            else:
                z = ffn(z) + z
        return z


# code copied from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
# author: Nic Dahlquist
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):

        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class CrossFormer(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=False,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 dropout=0.,
                 cat_pos=False,
                 ):
        super().__init__()

        self.cross_attn_module = CrossLinearAttention(dim, attn_type,
                                                       heads=heads, dim_head=dim_head, dropout=dropout,
                                                       relative_emb=relative_emb,
                                                       scale=scale,

                                                       relative_emb_dim=relative_emb_dim,
                                                       min_freq=min_freq,
                                                       init_method='orthogonal',
                                                       cat_pos=cat_pos,
                                                       pos_dim=relative_emb_dim,
                                                       project_query=False
                                                  )
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn

        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        if self.use_ln:
            z = self.ln1(z)
            if self.residual:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos)) + x
            else:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos))
        else:
            if self.residual:
                x = self.cross_attn_module(x, z, x_pos, z_pos) + x
            else:
                x = self.cross_attn_module(x, z, x_pos, z_pos)

        if self.use_ffn:
            x = self.ffn(x) + x

        return x


class BranchTrunkNet(nn.Module):
    def __init__(self,
                 dim,
                 branch_size,
                 branchnet_dim,
                 ):
        super().__init__()
        self.proj = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(branch_size, branchnet_dim),
            nn.ReLU(),
            nn.Linear(branchnet_dim//2, branchnet_dim//2),
            nn.ReLU(),
            nn.Linear(branchnet_dim//2, 1),

        )
        self.net = ProjDotProduct(dim, dim, dim)

    def forward(self, x, z):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        z = self.proj(z).squeeze(-1)
        return self.net(x, z)


class PointWiseDecoder2D(nn.Module):
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,  # 10
                 propagator_depth,
                 scale=8,
                 dropout=0.,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        # self.coordinate_projection = nn.Sequential(
        #     GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
        #     nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        #     nn.GELU(),
        #     nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        # )
        self.coordinate_projection = SirenNet(2, self.latent_channels, self.latent_channels*4, 4)

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=16.,
                                                relative_emb_dim=2,
                                                min_freq=1/64,
                                                residual=False)

        # self.expand_feat = nn.Linear(self.latent_channels//2, self.latent_channels)

        # self.propagator = nn.ModuleList([
        #        nn.ModuleList([nn.LayerNorm(self.latent_channels),
        #        nn.Sequential(
        #             nn.Linear(self.latent_channels + 2, self.latent_channels, bias=False),
        #             nn.GELU(),
        #             nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        #             nn.GELU(),
        #             nn.Linear(self.latent_channels, self.latent_channels, bias=False))])
        #     for _ in range(propagator_depth)])

        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels * self.out_steps, bias=True))

    def propagate(self, z, pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos), dim=-1)) + z
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def get_embedding(self,
                      z,  # [b, n c]
                      propagate_pos,  # [b, n, 2]
                      input_pos
                      ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos,   # [b, n, 2]
                input_pos,
                ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        u = self.decode(z)
        u = rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return u, z                # [b c_out t n], [b c_latent t n]

    # def rollout(self,
    #             z,  # [b, n c]
    #             propagate_pos,  # [b, n, 2]
    #             forward_steps,
    #             input_pos):
    #     history = []
    #     x = self.coordinate_projection.forward(propagate_pos)
    #     z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
    #     z = self.expand_feat(z)

    #     # forward the dynamics in the latent space
    #     for step in range(forward_steps//self.out_steps):
    #         z = self.propagate(z, propagate_pos)
    #         u = self.decode(z)
    #         history.append(rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps))
    #     history = torch.cat(history, dim=-2)  # concatenate in temporal dimension
    #     return history  # [b, length_of_history*c, n]



class PointWiseDecoder1D(nn.Module):
    # for Burgers equation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 decoding_depth,  # 4?
                 scale=8,
                 res=2048,
                 **kwargs,
                  ):
         super().__init__()
         self.layers = nn.ModuleList([])
         self.out_channels = out_channels
         self.latent_channels = latent_channels
 
         #self.coordinate_projection = SirenNet(1, self.latent_channels, self.latent_channels*8, 4)
         self.coordinate_projection = SirenNet(1, self.latent_channels, self.latent_channels*8, 4)
 
 
         self.decoding_transformer = CrossFormer(self.latent_channels, 'fourier', 8,
                                                 self.latent_channels, self.latent_channels,
                                                 relative_emb=True,
                                                 scale=1,
                                                 relative_emb_dim=1,
                                                 min_freq=1/res, residual=False)
 
         self.propagator = nn.ModuleList([
             nn.Sequential(
                 nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                 nn.GELU(),
                 nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                 nn.GELU(),
                 nn.Linear(self.latent_channels, self.latent_channels, bias=False),)
             for _ in range(decoding_depth)])
 
         self.init_propagator_params()
         self.to_out = nn.Sequential(
             nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
             nn.GELU(),
             nn.Linear(self.latent_channels//2, self.out_channels, bias=True))
 
    def propagate(self, z):
        for num_l, layer in enumerate(self.propagator):
            z = z + layer(z)
        return z
 
    def decode(self, z):
        z = self.to_out(z)
        return z
 
    def init_propagator_params(self):
        for block in self.propagator:
            for layers in block:
                    for param in layers.parameters():
                        if param.ndim > 1:
                            in_c = param.size(-1)
                            orthogonal_(param[:in_c], gain=1/in_c)
                            param.data[:in_c] += 1/in_c * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))
                            if param.size(-2) != param.size(-1):
                                orthogonal_(param[in_c:], gain=1/in_c)
                                param.data[in_c:] += 1/in_c * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))
 
    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):
 
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
 
        z = self.propagate(z)
        z = self.decode(z)
        return z  # [b, n, c]



#class PointWiseDecoder1D(nn.Module):
#    # for Burgers equation
#    def __init__(self,
#                 latent_channels,  # 256??
#                 out_channels,  # 1 or 2?
#                 decoding_depth,  # 4?
#                 scale=8,
#                 res=2048,
#                 **kwargs,
#                 ):
#        super().__init__()
#        self.layers = nn.ModuleList([])
#        self.out_channels = out_channels
#        self.latent_channels = latent_channels
#
#        self.coordinate_projection = nn.Sequential(
#            GaussianFourierFeatureTransform(1, self.latent_channels, scale=scale),
#            nn.GELU(),
#            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
#        )
#
#        self.decoding_transformer = CrossFormer(self.latent_channels, 'fourier', 8,
#                                                self.latent_channels, self.latent_channels,
#                                                relative_emb=True,
#                                                scale=1,
#                                                relative_emb_dim=1,
#                                                min_freq=1/res)
#
#        self.propagator = nn.ModuleList([
#            nn.Sequential(
#                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
#                nn.GELU(),
#                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
#                nn.GELU(),
#                nn.Linear(self.latent_channels, self.latent_channels, bias=False),)
#            for _ in range(decoding_depth)])
#
#        self.init_propagator_params()
#        self.to_out = nn.Sequential(
#            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
#            nn.GELU(),
#            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))
#
#    def propagate(self, z):
#        for num_l, layer in enumerate(self.propagator):
#            z = z + layer(z)
#        return z
#
#    def decode(self, z):
#        z = self.to_out(z)
#        return z
#
#    def init_propagator_params(self):
#        for block in self.propagator:
#            for layers in block:
#                    for param in layers.parameters():
#                        if param.ndim > 1:
#                            in_c = param.size(-1)
#                            orthogonal_(param[:in_c], gain=1/in_c)
#                            param.data[:in_c] += 1/in_c * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))
#                            if param.size(-2) != param.size(-1):
#                                orthogonal_(param[in_c:], gain=1/in_c)
#                                param.data[in_c:] += 1/in_c * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))
#
#    def forward(self,
#                z,  # [b, n c]
#                propagate_pos,  # [b, n, 1]
#                input_pos=None,
#                ):
#
#        #print()
#        #print()
#        #print(propagate_pos.shape)
#        x = self.coordinate_projection.forward(propagate_pos)
#        #print(x.shape)
#        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
#        #print(z.shape)
#
#        z = self.propagate(z)
#        #print(z.shape)
#        z = self.decode(z)
#        #print(z.shape)
#        #print()
#        #print()
#        return z  # [b, n, c]


class PointWiseDecoder2DSimple(nn.Module):
    # for Darcy equation
    def __init__(self,
                 latent_channels,  # 256k
                 out_channels,  # 1 or 2?
                 res=211,
                 scale=0.5,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            # nn.Linear(2, self.latent_channels, bias=False),
            # nn.GELU(),
            # nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
            # nn.GELU(),
            # nn.Linear(self.latent_channels * 2, self.latent_channels, bias=False),
            nn.Dropout(0.05),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                use_ln=False,
                                                residual=True,
                                                relative_emb=True,
                                                scale=16,
                                                relative_emb_dim=2,
                                                min_freq=1/res)

        # self.init_propagator_params()
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels+2, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)

        z = self.decode(torch.cat((z, propagate_pos), dim=-1))
        return z  # [b, n, c]


class STPointWiseDecoder2D(nn.Module):
    # no time marching
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,
                 scale=8,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(3, self.latent_channels//2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 1,
                                                self.latent_channels, self.latent_channels,
                                                residual=False,
                                                use_ffn=False,
                                                relative_emb=True,
                                                scale=1.,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos,  # [b, tn, 3]
                input_pos,      # [b, n, 2]
                ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos[:, :, :-1], input_pos)
        z = self.decode(z)
        z = rearrange(z, 'b (t n) c -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return z


class BCDecoder1D(nn.Module):
    # for Burgers equation, using DeepONet formulation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 decoding_depth,  # 4?
                 scale=8,
                 res=2048,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(1, self.latent_channels, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.decoding_transformer = BranchTrunkNet(latent_channels,
                                                   res)

    def forward(self,
                z,  # [b, n, c]
                propagate_pos,  # [b, n, 1]
                ):
        propagate_pos = propagate_pos[0]
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z)

        return z  # [b, n, c]


class PieceWiseDecoder2DSimple(nn.Module):
    # for Darcy flow inverse problem
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 res=141,
                 scale=0.5,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            # nn.Linear(2, self.latent_channels, bias=False),
            # nn.GELU(),
            # nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
            # nn.GELU(),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.Dropout(0.05),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                use_ln=False,
                                                residual=True,
                                                use_ffn=False,
                                                relative_emb=True,
                                                scale=16,
                                                relative_emb_dim=2,
                                                min_freq=1/res)

        # self.init_propagator_params()
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels+2, self.latent_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)

        z = self.decode(torch.cat((z, propagate_pos), dim=-1))
        return z  # [b, n, c]


class NoRelPointWiseDecoder2D(nn.Module):
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,  # 10
                 propagator_depth,
                 scale=8,
                 dropout=0.,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels//2, 'galerkin', 4,
                                                self.latent_channels//2, self.latent_channels//2,
                                                relative_emb=False,
                                                cat_pos=True,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.expand_feat = nn.Linear(self.latent_channels//2, self.latent_channels)

        self.propagator = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(self.latent_channels),
               nn.Sequential(
                    nn.Linear(self.latent_channels + 2, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False))])
            for _ in range(propagator_depth)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels * self.out_steps, bias=True))

    def propagate(self, z, pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos), dim=-1)) + z
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def get_embedding(self,
                      z,  # [b, n c]
                      propagate_pos,  # [b, n, 2]
                      input_pos
                      ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos   # [b, n, 2]
                ):
        z = self.propagate(z, propagate_pos)
        u = self.decode(z)
        u = rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return u, z                # [b c_out t n], [b c_latent t n]

    def rollout(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos):
        history = []
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)

        # forward the dynamics in the latent space
        for step in range(forward_steps//self.out_steps):
            z = self.propagate(z, propagate_pos)
            u = self.decode(z)
            history.append(rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps))
        history = torch.cat(history, dim=-2)  # concatenate in temporal dimension
        return history  # [b, length_of_history*c, n]


class PointWiseDecoder2DTemporal(nn.Module):
    # for 2d convection diffusion equation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 propagator_depth,
                 res=64,
                 scale=2,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            # nn.Linear(2, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            # nn.GELU(),
            # nn.Linear(self.latent_channels * 2, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                use_ln=False,
                                                residual=True,
                                                relative_emb=True,
                                                scale=16,
                                                relative_emb_dim=2,
                                                min_freq=1/res)

        self.propagator = nn.ModuleList([
                           nn.Sequential(
                               nn.LayerNorm(self.latent_channels),
                               nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels, self.latent_channels, bias=False))
            for _ in range(propagator_depth)])
        # self.init_propagator_params()
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def init_propagator_params(self):
        for block in self.propagator:
            for layers in block:
                    for param in layers.parameters():
                        if param.ndim > 1:
                            in_c = param.size(-2)
                            orthogonal_(param, gain=1/in_c)
                            param.data[:, :in_c] += 1/in_c * torch.diag(torch.ones(in_c, dtype=torch.float32))


    def propagate(self, z, pos):
        for num_l, layer in enumerate(self.propagator):
            ffn = layer
           #  z = ffn(torch.cat((z, pos), dim=-1)) + z
            z = ffn(z) + z
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.propagate(z, propagate_pos)
        z = self.decode(z)
        return z  # [b, n, c]


class STDecoder1D(nn.Module):
    # for time-marching Burgers equation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 scale=8,
                 res=2048,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(1, self.latent_channels, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'fourier', 4,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=1,
                                                relative_emb_dim=1,
                                                min_freq=1/res)

        self.propagator = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(self.latent_channels),
                           nn.Sequential(
                               nn.Linear(self.latent_channels + 1, self.latent_channels, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels, self.latent_channels, bias=False))])
                                        ])

        self.init_propagator_params()
        self.to_out = nn.ModuleList([
            LinearAttention(self.latent_channels, 'fourier',
                            heads=1, dim_head=self.latent_channels,
                            pos_dim=1,
                            relative_emb=True,
                            relative_emb_dim=1,
                            min_freq=1/res,
                            ),
            nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))]
        )

    def propagate(self, z, pos):
        for num_l, layer in enumerate(self.propagator):
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos/16.), dim=-1)) + z
        return z

    def decode(self, z, pos):
        attn, ffn = self.to_out[0], self.to_out[1]
        z = attn(z, pos) + z
        z = ffn(z)
        return z

    def init_propagator_params(self):
        for layers in self.propagator[0][1]:
                for param in layers.parameters():
                    if param.ndim > 1:
                        in_c = param.size(-2)
                        orthogonal_(param, gain=1 / in_c)
                        param.data[:, :in_c] += 1 / in_c * torch.diag(torch.ones(in_c, dtype=torch.float32))

    def get_embedding(self,
                      z,  # [b, n c]
                      propagate_pos,  # [b, n, 2]
                      input_pos
                      ):
        x = self.coordinate_projection.forward(propagate_pos/16.)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        return z

    def forward(self,
                z,  # [b, n, c]
                propagate_pos  # [b, n, 2]
                ):
        x = self.coordinate_projection.forward(propagate_pos/16.)
        z = self.decoding_transformer.forward(x, z, propagate_pos)

        z = self.propagate(z, propagate_pos)
        u = self.decode(z, propagate_pos)
        u = rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return u, z  # [b c_out t n], [b c_latent t n]

    def rollout(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos):
        history = []
        x = self.coordinate_projection.forward(propagate_pos/16.)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z_init = z

        # forward the dynamics in the latent space
        for step in range(forward_steps):
            z = self.propagate(z, propagate_pos)
            u = self.decode(z, propagate_pos)
            history.append(rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=1))
        history = torch.cat(history, dim=-2)  # concatenate in temporal dimension
        return z_init, history  # [b, length_of_history*c, n]


class STDecoder2D(nn.Module):
    # For Markovian Navier-Stokes / Reaction-diffusion
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 scale=8,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=16.,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.propagator = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(self.latent_channels),
               nn.Sequential(
                    nn.Linear(self.latent_channels + 2, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False))])
            ])

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def load_pretrained_weights(self, pretrained_decoder):
        copy_weights(self.coordinate_projection, pretrained_decoder.coordinate_projection)
        copy_weights(self.decoding_transformer, pretrained_decoder.decoding_transformer)


    def propagate(self, z, pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos), dim=-1)) + z
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos   # [b, n, 2]
                ):
        z = self.propagate(z, propagate_pos)
        u = self.decode(z)
        if self.out_channels == 1:
            u = rearrange(u, 'b n t-> b t n')
        else:
            u = rearrange(u, 'b n (t c) -> b t c n', c=self.out_channels)
        return u, z                # [b c_out t n], [b c_latent t n]

    def rollout(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos):
        history = []
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)

        # forward the dynamics in the latent space
        for step in range(forward_steps):
            z = self.propagate(z, propagate_pos)
            u = self.decode(z)

            if self.out_channels == 1:
                u = rearrange(u, 'b n t-> b t n')
            else:
                u = rearrange(u, 'b n (t c) -> b t n c', c=self.out_channels)

            history.append(u)
        history = torch.cat(history, dim=1)  # concatenate in temporal dimension
        return history


class ReconSTDecoder2D(nn.Module):
    # For Navier-Stokes MAE
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 scale=8,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=16.,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        # simple reconstruction head
        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def forward(self,
                z,              # [b, n, c]
                propagate_pos,   # [b, n, 2]
                input_pos,
                ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.to_out(z)
        return z                # [b n c_out]


class SpatialDecoder2D(nn.Module):
    # for Darcy equation or Heat-cavity
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 res=50,
                 scale=0.1,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                use_ln=False,
                                                residual=True,
                                                relative_emb=True,
                                                scale=8,
                                                relative_emb_dim=2,
                                                min_freq=1/res)

        self.ln = nn.LayerNorm(self.latent_channels)

        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels+1, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos,
                param   # [b,]
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.ln(z)
        param = param.view(-1, 1, 1).repeat([1, z.shape[0], 1])
        z = torch.cat((z, param), dim=-1)
        z = self.decode(z)
        return z  # [b, n, c]


class STDecoder3D(nn.Module):
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 scale=8,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.time_projection = nn.Sequential(
            nn.Linear(1, self.latent_channels, bias=False),
            nn.Tanh(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False))

        self.combine = nn.Linear(self.latent_channels*2, self.latent_channels, bias=False)


        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=16.,
                                                relative_emb_dim=2,
                                                use_ffn=False,
                                                min_freq=1/64)

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.Tanh(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.Tanh(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.Tanh(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.Tanh(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def forward(self,
                z,              # [b, n, c]
                propagate_pos_x,   # [b, (t) n, 1]  2d spatial
                propagate_pos_y,  # [b, (t) n, 1]  2d spatial
                propagate_time,  # [b, (t) n, 1]
                input_pos,
                ):
        pos_enc = self.coordinate_projection.forward(torch.cat((propagate_pos_x, propagate_pos_y), dim=-1))
        time_enc = self.time_projection.forward(propagate_time)
        x = self.combine(torch.cat((pos_enc, time_enc), dim=-1))
        propagate_pos = torch.cat((propagate_pos_x, propagate_pos_y), dim=-1)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)

        u = self.to_out(z)     # b (t n) 2

        return u





