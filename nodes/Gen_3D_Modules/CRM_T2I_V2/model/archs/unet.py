'''
Codes are from:
https://github.com/jaxony/unet-pytorch/blob/master/model.py
'''

import torch
import torch.nn as nn
from diffusers import UNet2DModel
import einops
class UNetPP(nn.Module):
    '''
        Wrapper for UNet in diffusers
    '''
    def __init__(self, in_channels):
        super(UNetPP, self).__init__()
        self.in_channels = in_channels
        self.unet = UNet2DModel(
                sample_size=[256, 256*3],
                in_channels=in_channels,
                out_channels=32,
                layers_per_block=2,
                block_out_channels=(64, 128, 128, 128*2, 128*2, 128*4, 128*4),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
             
        # xformers is deliberately not a dependency of this pack (see
        # nodes/comfy-env.toml, and _try_enable_xformers in nodes.py which
        # already guards the calls it owns). diffusers raises
        # ModuleNotFoundError from attention_processor.py:405 when xformers is
        # absent, which took the whole model build down:
        #     ModuleNotFoundError: Refer to https://github.com/.../xformers ...
        #
        # Skipping is a backend change, not a lost optimisation. On torch 2.x
        # diffusers already defaults to AttnProcessor2_0 -- PyTorch SDPA, which
        # dispatches to flash kernels. This call was switching that default OFF.
        try:
            self.unet.enable_xformers_memory_efficient_attention()
        except (ModuleNotFoundError, ImportError, ValueError):
            pass
        if in_channels > 12:
            self.learned_plane = torch.nn.parameter.Parameter(torch.zeros([1,in_channels-12,256,256*3]))

    def forward(self, x, t=256):
        learned_plane = self.learned_plane
        if x.shape[1] < self.in_channels:
            learned_plane = einops.repeat(learned_plane, '1 C H W -> B C H W', B=x.shape[0]).to(x.device)
            x = torch.cat([x, learned_plane], dim = 1)
        return self.unet(x, t).sample

