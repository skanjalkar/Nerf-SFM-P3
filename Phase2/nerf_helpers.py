import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Embedder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires,
        'num_freqs': multires-1,
        'log_sampling': True,
    }

    embedderObj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedderObj: eo.embed(x)
    return embed, embedderObj.out_dim

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_channel_views=3, output_channel=4, skips=[4], use_viewdirs=False) -> None:
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_channel_views = input_channel_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]+[nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )

        self.view_linears = nn.ModuleList([nn.Linear(input_channel_views+W, W//2)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)

    def forward(self, x):
        # print("IN FORWARD ------------------")
        # print(x.shape)
        # print(self.input_ch)
        # print(self.input_channel_views)
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_channel_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.view_linears):
            h = self.view_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)

        return outputs


def get_rays(H, W, K, pose):
    # same as custom nerf
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H)) # ij indexing
    i, j = i.t(), j.t()

    directions = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1) # directions x, -y, -z
    # direction vector aligned with pose, same as dot product
    rays_d = torch.sum(directions[..., None, :]*pose[:3, :3], -1)

    # Origin of all rays is same
    rays_o = pose[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d