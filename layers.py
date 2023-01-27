import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, math

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
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]

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

class CrossAttention(nn.Module):
    def __init__(self, d: int, query_dim: int, context_dim: int = None, nheads: int = 8):
        super().__init__()
        self.d = d
        self.query_dim = query_dim
        context_dim = context_dim or query_dim 
        self.context_dim = context_dim
        self.nheads = nheads

        self.scale = (d // nheads) ** -0.25

        self.to_q = nn.Linear(query_dim, d, bias=False)
        self.to_k = nn.Linear(context_dim, d, bias=False)
        self.to_v = nn.Linear(context_dim, d, bias=False)
        self.to_out = nn.Linear(d, query_dim)

    def split_heads(self, q):
        B, N, D = q.shape
        d_head = D // self.nheads
        q = q.reshape(B, N, self.nheads, d_head) # (B, N, H, d_k)
        q = q.permute(0, 2, 1, 3) # (B, H, N, d_k)
        q = q.reshape(B*self.nheads, N, d_head) # (B*H, N, d_k)
        return q

    def merge_heads(self, x):
        BH, N, Dv = x.shape
        b = BH // self.nheads
        x = x.reshape(b, self.nheads, N, Dv) # (B, H, N, d_v)
        x = x.permute(0, 2, 1, 3) # (B, N, H, d_v)
        x = x.reshape(b, N, self.nheads*Dv) # (B, N, D)
        return x

    def forward(self, x, context=None):
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(self.split_heads, (q, k, v))

        kT = k.permute(0, 2, 1)
        sim = torch.bmm(q*self.scale, kT*self.scale)
        #sim = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.bmm(attn, v)
        #out = torch.einsum('bij,bjd->bid', attn, v)
        out = self.merge_heads(out)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    '''
    Attention between UNet feature maps and text embeddings.
    '''
    def __init__(self, in_c: int, d: int, d_emb: int, nheads: int=8):
        '''
        in_c: number of image input channels
        d: dimensionality of attention
        d_emb: dimensionality of conditioning embedding (like text conditioning)
        nheads: number of heads
        '''
        super().__init__()
        self.in_c = in_c
        self.d = d
        self.d_emb = d_emb
        self.nheads = nheads

        self.norm_in = nn.GroupNorm(8, in_c)
        self.proj_in = nn.Conv2d(in_c, d, 1)

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.attn1 = CrossAttention(d, d, d, nheads)
        self.attn2 = CrossAttention(d, d, d_emb, nheads)
        self.ff = nn.Sequential(
                nn.Linear(d, 4*d),
                nn.SiLU(),
                nn.Linear(4*d, d),
                )

        self.proj_out = nn.Conv2d(d, in_c, 1)

    def forward(self, x, context):
        skip = x
        x = self.proj_in(self.norm_in(x))
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)
        x = x.permute(0, 2, 1) # (B, H*W, C)

        h = self.norm1(x)
        x = self.attn1(h) + x
        x = self.attn2(self.norm2(x), context) + x
        x = h + x
        x = self.ff(self.norm3(x)) + x

        x = x.permute(0, 2, 1) # (B, H*W, d)
        x = x.reshape(B, self.d, H, W)
        x = self.proj_out(x)

        return x + skip

class TimeEmbedding(nn.Module):
    '''
    Sinusoidal time embedding with a feed forward network.
    '''
    def __init__(self, embed_c: int, out_c: int, max_period: int = 10000):
        '''
        embed_c: dimensionality of sinusoidal time embedding
        out_c: dimensionality of projected (output) embedding
        max_period: controls the minimum frequency of the embeddings
        '''
        super().__init__()
        self.embed_c = embed_c
        self.out_c = out_c
        self.max_period = max_period
        half = embed_c // 2
        self.freqs = torch.exp(-math.log(max_period) * torch.linspace(0, 1, half))
        self.ff = nn.Sequential(
                nn.Linear(embed_c, out_c),
                nn.SiLU(),
                nn.Linear(out_c, out_c),
            )

    def forward(self, timesteps):
        t_freqs = torch.outer(timesteps, self.freqs.to(timesteps.device))
        emb = torch.cat([t_freqs.cos(), t_freqs.sin()], dim=-1)
        emb = self.ff(emb)
        return emb
