from ptflops import get_model_complexity_info
from torch import nn
import torch

from models import FrozenCLIPEmbedder
from layers import ResBlock, Downsample, Upsample, Attn2d, TransformerBlock, TimeEmbedding
from transformers import CLIPTokenizerFast, CLIPTextModel, logging
logging.set_verbosity_error()

from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

class VAEEncoder(nn.Module):
    def __init__(self,
        nc: int,
        ch_mults: tuple,
        nlayers_per_res: int,
        nz: int
    ):
        '''
        nc: base number of channels
        ch_mults: channel multiplier per resolution
        nz: number of latent output channels
        '''
        super().__init__()
        self.nc = nc
        self.ch_mults = ch_mults
        self.nlayers_per_res = nlayers_per_res
        self.nz = nz

        self.first_conv = nn.Conv2d(3, nc, 3, padding=1)
        block_out_c = self.nc

        res_blocks = []
        downsamples = []
        for ch_mult in ch_mults:
            block_in_c, block_out_c = block_out_c, self.nc * ch_mult
            res_blocks.append(nn.ModuleList([
                ResBlock(block_in_c, block_out_c), *[ResBlock(block_out_c, block_out_c) for _ in range(nlayers_per_res-1)]
            ]))
            downsamples.append(Downsample(block_out_c))
        self.res_blocks = nn.ModuleList(res_blocks)
        self.downsamples = nn.ModuleList(downsamples)

        self.mid_block1 = ResBlock(block_out_c, block_out_c)
        self.mid_attn = Attn2d(block_out_c)
        self.mid_block2 = ResBlock(block_out_c, block_out_c)

        self.norm = nn.GroupNorm(8, block_out_c)
        self.last_conv = nn.Conv2d(block_out_c, 2*nz, 3, padding=1)
    
    def __call__(self, x):
        temb_placeholder = None

        x = self.first_conv(x)
        for block_idx, (res_block, downsample) in enumerate(zip(self.res_blocks, self.downsamples)):
            for res_layer in res_block:
                x = res_layer(x, temb_placeholder)
            x = downsample(x)
            
        x = self.mid_block1(x, temb_placeholder)
        x = self.mid_attn(x)
        x = self.mid_block2(x, temb_placeholder)
        
        x = self.norm(x)
        x = F.silu(x)
        x = self.last_conv(x) # nz*2 for the mean, stdev
        return x

class VAEDecoder(nn.Module):
    def __init__(self,
        nc: int,
        ch_mults: tuple,
        nlayers_per_res: int,
        nz: int
    ):
        '''
        nc: base number of channels
        ch_mults: channel multiplier per resolution
        nz: number of latent output channels
        '''
        super().__init__()
        self.nc = nc
        self.ch_mults = ch_mults
        self.nlayers_per_res = nlayers_per_res
        self.nz = nz

        block_out_c = self.nc * self.ch_mults[-1] 
        self.first_conv = nn.Conv2d(nz, block_out_c, 3, padding=1)

        self.mid_block1 = ResBlock(block_out_c, block_out_c)
        self.mid_attn = Attn2d(block_out_c)
        self.mid_block2 = ResBlock(block_out_c, block_out_c)

        res_blocks = []
        upsamples = []
        for ch_mult in reversed(ch_mults):
            block_in_c, block_out_c = block_out_c, self.nc * ch_mult
            res_blocks.append(nn.ModuleList([
                ResBlock(block_in_c, block_out_c), *[ResBlock(block_out_c, block_out_c) for _ in range(nlayers_per_res-1)]
            ]))
            upsamples.append(Upsample(block_out_c))
        self.res_blocks = nn.ModuleList(res_blocks)
        self.upsamples = nn.ModuleList(upsamples)

        self.norm = nn.GroupNorm(8, block_out_c)
        self.last_conv = nn.Conv2d(block_out_c, 3, 3, padding=1)
    
    def __call__(self, x):
        temb_placeholder = None

        x = self.first_conv(x)
        x = self.mid_block1(x, temb_placeholder)
        x = self.mid_attn(x)
        x = self.mid_block2(x, temb_placeholder)
        for res_block, upsample in zip(self.res_blocks, self.upsamples):
            for res_layer in res_block:
                x = res_layer(x, temb_placeholder)
            x = upsample(x)

        x = self.norm(x)
        x = F.silu(x)
        x = self.last_conv(x) # nz*2 for the mean, stdev
        return x
    
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.nc = 64
        self.ch_mults = (1, 2, 2, 4)
        self.nlayers_per_res = 2
        self.nz = 16
        self.encoder = VAEEncoder(nc=self.nc, ch_mults=self.ch_mults, nlayers_per_res=self.nlayers_per_res, nz=self.nz)
        self.decoder = VAEDecoder(nc=self.nc, ch_mults=self.ch_mults, nlayers_per_res=self.nlayers_per_res, nz=self.nz)

    def __call__(self, x):
        z_params = self.encoder(x) #.clamp(-30.0, 20.0)
        mean, log_var = torch.split(z_params, self.nz, dim=1)
        z = self.sample(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var

    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        return mean + eps * std

class TimestepEmbedSequential(nn.Sequential):
    '''
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    '''
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, TransformerBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(self,
        in_c: int = 16,
        nc: int = 256,
        ch_mults: list = [1, 2, 4],
        nlayers_per_res: int = 2,
        context_dim: int = 512,
    ):
        super().__init__()
        self.in_c = in_c
        self.nc = nc
        self.ch_mults = ch_mults
        self.nlayers_per_res = nlayers_per_res
        self.context_dim = context_dim
        self.temb_c = 4 * nc

        self.time_embed = TimeEmbedding(nc, self.temb_c)
        self.first_conv = nn.Conv2d(in_c, nc, 3, padding='same')

        self.downs = nn.ModuleList([])
        down_out_cs = []
        
        block_out_c = self.nc
        for block_idx, ch_mult in enumerate(ch_mults):
            block = []
            block_in_c, block_out_c = block_out_c, nc * ch_mult
            down_out_cs.append(block_out_c)

            layer_out_c = block_in_c
            for _ in range(nlayers_per_res):
                layer_in_c, layer_out_c = layer_out_c, block_out_c
                block.append(ResBlock(layer_in_c, layer_out_c, self.temb_c))
                block.append(TransformerBlock(layer_out_c, layer_out_c, context_dim))
            if block_idx != len(ch_mults) - 1:
                block.append(Downsample(block_out_c))

            block = TimestepEmbedSequential(*block)
            self.downs.append(block)

        self.mid_block = TimestepEmbedSequential(
                ResBlock(block_out_c, block_out_c, self.temb_c),
                TransformerBlock(block_out_c, block_out_c, context_dim),
                ResBlock(block_out_c, block_out_c, self.temb_c),
            )

        self.ups = nn.ModuleList([])
        for block_idx, ch_mult in enumerate(reversed(ch_mults)):
            block = []
            block_in_c, block_out_c = block_out_c, nc * ch_mult
            down_c = down_out_cs.pop()

            layer_out_c = block_in_c + down_c
            for _ in range(nlayers_per_res):
                layer_in_c, layer_out_c = layer_out_c, block_out_c
                block.append(ResBlock(layer_in_c, layer_out_c, self.temb_c))
                block.append(TransformerBlock(layer_out_c, layer_out_c, context_dim))
            if block_idx != 0:
                block.append(Upsample(block_out_c))

            block = TimestepEmbedSequential(*block)
            self.ups.append(block)

        self.out = nn.Sequential(
                nn.GroupNorm(8, block_out_c + nc),
                nn.SiLU(),
                nn.Conv2d(block_out_c + nc, in_c, 3, padding='same')
            )

    def forward(self, x):
        timesteps = torch.zeros((x.shape[0],))
        context = torch.zeros((x.shape[0],77,self.context_dim))
        temb = self.time_embed(timesteps)
        x = self.first_conv(x)
        skip = x
        downs = []
        for block in self.downs:
            x = block(x, temb, context)
            downs.append(x)
        x = self.mid_block(x, temb, context)
        for block in self.ups:
            x = torch.cat([x, downs.pop()], dim=1)
            x = block(x, temb, context)
        x = torch.cat([x, skip], dim=1)
        x = self.out(x)

        return x

def get_gmacs(macs_str):
    macs, base = macs_str.split()
    macs = float(macs)
    if 'kmac' in base.lower():
        return macs / 1000 / 1000
    elif 'mmac' in base.lower():
        return macs / 1000
    elif 'gmac' in base.lower():
        return macs

# base: 178.7 gflops per forward + backward pass
# big: 243.5 gflops per forward + backward pass

net = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
#net = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
clip_input_constructor = lambda shape: {'input_ids': torch.ones((1, *shape), dtype=torch.long)}
clip_macs, params = get_model_complexity_info(net, (77,), input_constructor=clip_input_constructor, as_strings=True, print_per_layer_stat=False, verbose=False)
print('CLIP')
print('{:<30}  {:<8}'.format('Computational complexity: ', clip_macs))
print('{:<30}  {:<8}'.format('N params: ', params))

net = VAEEncoder(64,[1,2,2,4],2,16)
vae_macs, params = get_model_complexity_info(net, (3,256,256), as_strings=True, print_per_layer_stat=False, verbose=False)
print('VAE Encoder')
print('{:<30}  {:<8}'.format('Computational complexity: ', vae_macs))
print('{:<30}  {:<8}'.format('N params: ', params))

net = UNet(nc=256, context_dim=768)
#net = UNet(nc=64, context_dim=512)
unet_macs, params = get_model_complexity_info(net, (net.in_c,16,16), as_strings=True, print_per_layer_stat=False, verbose=False)
print('UNet')
print('{:<30}  {:<8}'.format('Computational complexity: ', unet_macs))
print('{:<30}  {:<8}'.format('N params: ', params))

total_gmacs = sum(get_gmacs(i) for i in (clip_macs, vae_macs, unet_macs))
print('Total GMacs:', total_gmacs)
print('Total GFLOPs per forward pass:', 2*total_gmacs)
print('Total GFLOPs per forward and backward pass:', 3*2*total_gmacs)
