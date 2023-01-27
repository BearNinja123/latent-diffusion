import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class ResBlock(nn.Module):
    def __init__(self, in_c: int, nc: int, temb_c: int = None):
        '''
        in_c: number of input channels
        nc: number of output channels
        temb_c: number of t (time?) embedding input channels (or None if no time embedding)
        '''
        super().__init__()
        self.in_c = in_c
        self.nc = nc
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, nc, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, nc)
        self.conv2 = nn.Conv2d(nc, nc, 3, padding=1)
        if temb_c is not None:
            self.temb_proj = nn.Linear(temb_c, nc)
        self.skip_conv = nn.Conv2d(in_c, nc, 1)
    
    def forward(self, x, temb): # temb = t (time) embedding
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = F.silu(h)

        if temb is not None:
            h = h + self.temb_proj[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        x = self.skip_conv(x)

        return x + h

class Downsample(nn.Module):
    def __init__(self, nc: int, with_conv: bool = True):
        '''
        nc: number of input and output channels
        with_conv: whether or not to downsample with a strided conv
        '''
        super().__init__()
        self.nc = nc
        self.with_conv = with_conv
        self.layer = nn.Conv2d(nc, nc, 3, 2) if with_conv else nn.AvgPool2d((2, 2))

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (0, 1, 0, 1))
        return self.layer(x)

class Upsample(nn.Module):
    def __init__(self, nc: int, with_conv: bool = True):
        '''
        nc: number of input and output channels
        with_conv: whether or not to finish upsample with a conv
        '''
        super().__init__()
        self.nc = nc
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(nc, nc, 3, padding=1)

    def forward(self, x):
        B, H, W, C = x.shape
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x

class Attn2d(nn.Module):
    def __init__(self, nc: int):
        '''
        nc: number of input and output channels
        '''
        super().__init__()
        self.nc = nc
        self.norm = nn.GroupNorm(8, self.nc)
        self.attn = nn.MultiheadAttention(self.nc, 1)
        self.conv = nn.Conv2d(self.nc, self.nc, 1)

    def forward(self, x):
        h = x
        B, C, H, W = h.shape
        h = self.norm(h)
        h = h.reshape(B, C, H*W)
        h = h.permute(0, 2, 1) # B, H*W, C
        h, _attn_weights = self.attn(h, h, h)
        h = h.permute(0, 2, 1) # B, C, H*W
        h = h.reshape(B, C, H, W)
        h = self.conv(h)

        return x + h
