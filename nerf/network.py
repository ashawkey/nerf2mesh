import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 specular_dim=3,
                 ):

        super().__init__(opt)

        # sigma and feature network
        
        self.encoder, self.in_dim_density = get_encoder("hashgrid", level_dim=1, desired_resolution=2048 * self.bound, interpolation='smoothstep')
        self.encoder_color, self.in_dim_color = get_encoder("hashgrid", level_dim=2, desired_resolution=2048 * self.bound, interpolation='linear')

        self.sigma_net = MLP(self.in_dim_density, 1, 32, 2, bias=False)

        # color network
        self.encoder_dir, self.in_dim_dir = get_encoder("None")
        self.color_net = MLP(self.in_dim_color + self.individual_dim, 3 + specular_dim, 64, 3, bias=False)
        self.specular_net = MLP(specular_dim + self.in_dim_dir, 3, 32, 2, bias=False)


    def forward(self, x, d, c=None, shading='full'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # c: [1/N, individual_dim]

        sigma = self.density(x)['sigma']
        color, specular = self.rgb(x, d, c, shading)

        return sigma, color, specular


    def density(self, x):

        # sigma
        h = self.encoder(x, bound=self.bound)
        h = self.sigma_net(h)

        sigma = trunc_exp(h[..., 0])

        return {
            'sigma': sigma,
        }
    

    def geo_feat(self, x, c=None):

        h = self.encoder_color(x, bound=self.bound)
        if c is not None:
            h = torch.cat([h, c.repeat(x.shape[0], 1) if c.shape[0] == 1 else c], dim=-1)
        h = self.color_net(h)
        geo_feat = torch.sigmoid(h)

        return geo_feat


    def rgb(self, x, d, c=None, shading='full'):

        # color
        geo_feat = self.geo_feat(x, c)
        diffuse = geo_feat[..., :3]

        d = self.encoder_dir(d)
        if shading == 'diffuse':
            color = diffuse
            specular = None
        else: 
            specular = self.specular_net(torch.cat([d, geo_feat[..., 3:]], dim=-1))
            specular = torch.sigmoid(specular)
            if shading == 'specular':
                color = specular
            else: # full
                color = (specular + diffuse).clamp(0, 1) # specular + albedo

        return color, specular


    # optimizer utils
    def get_params(self, lr):

        params = super().get_params(lr)

        params.extend([
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.encoder_color.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
            {'params': self.specular_net.parameters(), 'lr': lr}, 
        ])

        return params