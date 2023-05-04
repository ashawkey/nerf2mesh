import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class FreqEncoder_torch(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
    
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)


        return out

class TCNN_hashgrid(nn.Module):
    def __init__(self, num_levels, level_dim, log2_hashmap_size, base_resolution, desired_resolution, interpolation, **kwargs):
        super().__init__()
        import tinycudann as tcnn
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": np.exp2(np.log2(desired_resolution / num_levels) / (num_levels - 1)),
                "interpolation": "Smoothstep" if interpolation == 'smoothstep' else "Linear",
            },
            dtype=torch.float32,
        )
        self.output_dim = self.encoder.n_output_dims # patch
    
    def forward(self, x, bound=1, **kwargs):
        return self.encoder((x + bound) / (2 * bound))


def get_encoder(encoding, input_dim=3, 
                output_dim=1, resolution=300, mode='bilinear', # dense grid
                multires=6, # freq
                degree=4, # SH
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, # hash/tiled grid
                align_corners=False, interpolation='linear', # grid
                **kwargs):

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim

    elif encoding == 'frequency_torch':
        encoder = FreqEncoder_torch(input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)

    elif encoding == 'frequency':
        from freqencoder import FreqEncoder
        encoder = FreqEncoder(input_dim=input_dim, degree=multires)

    elif encoding == 'sh':
        from shencoder import SHEncoder
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == 'hashgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners, interpolation=interpolation)
    
    elif encoding == 'hashgrid_tcnn':
        encoder = TCNN_hashgrid(num_levels=num_levels, level_dim=level_dim, log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution, desired_resolution=desired_resolution, interpolation=interpolation)
    
    elif encoding == 'tiledgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners, interpolation=interpolation)
    
    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sh, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim