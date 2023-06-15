import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, geom_init=False, weight_norm=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.geom_init = geom_init

        net = []
        for l in range(num_layers):

            in_dim = self.dim_in if l == 0 else self.dim_hidden
            out_dim = self.dim_out if l == num_layers - 1 else self.dim_hidden

            net.append(nn.Linear(in_dim, out_dim, bias=bias))
        
            if geom_init:
                if l == num_layers - 1:
                    torch.nn.init.normal_(net[l].weight, mean=math.sqrt(math.pi) / math.sqrt(in_dim), std=1e-4)
                    if bias: torch.nn.init.constant_(net[l].bias, -0.5) # sphere init (very important for hashgrid encoding!)

                elif l == 0:
                    torch.nn.init.normal_(net[l].weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(out_dim))
                    torch.nn.init.constant_(net[l].weight[:, 3:], 0.0)
                    if bias: torch.nn.init.constant_(net[l].bias, 0.0)

                else:
                    torch.nn.init.normal_(net[l].weight, 0.0, math.sqrt(2) / math.sqrt(out_dim))
                    if bias: torch.nn.init.constant_(net[l].bias, 0.0)
            
            if weight_norm:
                net[l] = nn.utils.weight_norm(net[l])

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                if self.geom_init:
                    x = F.softplus(x, beta=100)
                else:
                    x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 specular_dim=3,
                 ):

        super().__init__(opt)

        # density network
        self.encoder, self.in_dim_density = get_encoder("hashgrid_tcnn" if self.opt.tcnn else "hashgrid", level_dim=1, desired_resolution=2048 * self.bound, interpolation='linear')
        # self.sigma_net = MLP(3 + self.in_dim_density, 1, 32, 2, bias=self.opt.sdf, geom_init=self.opt.sdf, weight_norm=self.opt.sdf)
        self.sigma_net = MLP(3 + self.in_dim_density, 1, 32, 2, bias=False)

        # color network
        self.encoder_color, self.in_dim_color = get_encoder("hashgrid_tcnn" if self.opt.tcnn else "hashgrid", level_dim=2, desired_resolution=2048 * self.bound, interpolation='linear')
        self.color_net = MLP(3 + self.in_dim_color + self.individual_dim, 3 + specular_dim, 64, 3, bias=False)

        self.encoder_dir, self.in_dim_dir = get_encoder("None")
        self.specular_net = MLP(specular_dim + self.in_dim_dir, 3, 32, 2, bias=False)

        # sdf
        if self.opt.sdf:
            self.register_parameter('variance', nn.Parameter(torch.tensor(0.3, dtype=torch.float32)))

    def forward(self, x, d, c=None, shading='full'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # c: [1/N, individual_dim]

        sigma = self.density(x)['sigma']
        color, specular = self.rgb(x, d, c, shading)

        return sigma, color, specular


    def density(self, x):

        # sigma
        h = self.encoder(x, bound=self.bound, max_level=self.max_level)
        h = torch.cat([x, h], dim=-1)
        h = self.sigma_net(h)

        results = {}

        if self.opt.sdf:
            sigma = h[..., 0].float() # sdf
        else:
            sigma = trunc_exp(h[..., 0])

        results['sigma'] = sigma

        return results

    # init the sdf to two spheres by pretraining, assume view cameras fall between the spheres
    def init_double_sphere(self, r1=0.5, r2=1.5, iters=8192, batch_size=8192):
        # sphere init is only for sdf mode!
        if not self.opt.sdf:
            return
        # import kiui
        import tqdm
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)
        pbar = tqdm.trange(iters, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for _ in range(iters):
            # random points inside [-b, b]^3
            xyzs = torch.rand(batch_size, 3, device='cuda') * 2 * self.bound - self.bound
            d = torch.norm(xyzs, p=2, dim=-1)
            gt_sdf = torch.where(d < (r1 + r2) / 2, d - r1, r2 - d)
            # kiui.lo(xyzs, gt_sdf)
            pred_sdf = self.density(xyzs)['sigma']
            loss = loss_fn(pred_sdf, gt_sdf)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f'pretrain sdf loss={loss.item():.8f}')
            pbar.update(1)
    
    # finite difference
    def normal(self, x, epsilon=1e-4):

        if self.opt.tcnn:
            with torch.enable_grad():
                x.requires_grad_(True)
                sigma = self.density(x)['sigma']
                normal = torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]
        else:
            dx_pos = self.density((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dx_neg = self.density((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dy_pos = self.density((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dy_neg = self.density((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dz_pos = self.density((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dz_neg = self.density((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            
            normal = torch.stack([
                0.5 * (dx_pos - dx_neg) / epsilon, 
                0.5 * (dy_pos - dy_neg) / epsilon, 
                0.5 * (dz_pos - dz_neg) / epsilon
            ], dim=-1)

        return normal
    

    def geo_feat(self, x, c=None):

        h = self.encoder_color(x, bound=self.bound, max_level=self.max_level)
        h = torch.cat([x, h], dim=-1)
        if c is not None:
            h = torch.cat([h, c.repeat(x.shape[0], 1) if c.shape[0] == 1 else c], dim=-1)
        h = self.color_net(h)
        geo_feat = torch.sigmoid(h)

        return geo_feat


    def rgb(self, x, d, c=None, shading='full'):

        # color
        geo_feat = self.geo_feat(x, c)
        diffuse = geo_feat[..., :3]

        if shading == 'diffuse':
            color = diffuse
            specular = None
        else: 
            d = self.encoder_dir(d)
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

        if self.opt.sdf:
            params.append({'params': self.variance, 'lr': lr * 0.1})

        return params