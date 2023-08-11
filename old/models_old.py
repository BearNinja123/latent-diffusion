from layers_old import ResBlock, Downsample, Upsample, Attn2d, TransformerBlock, TimeEmbedding
from transformers import CLIPTokenizerFast, CLIPTextModel
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
    
    def forward(self, x):
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
        x = F.relu(x)
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
    
    def forward(self, x):
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
        x = F.relu(x)
        x = self.last_conv(x) # nz*2 for the mean, stdev
        return x

class VAE(nn.Module):
    def __init__(
        self,
        nc: int = 128,
        ch_mults: list = [1, 1, 2, 2, 4],
        nlayers_per_res: int = 2,
        nz: int = 16
    ):
        super().__init__()
        # put input args as class fields
        locals_ = locals().copy()
        locals_.pop('self')
        for k, v in locals_.items():
            setattr(self, k, v)

        self.encoder = VAEEncoder(nc=self.nc, ch_mults=self.ch_mults, nlayers_per_res=self.nlayers_per_res, nz=self.nz)
        self.decoder = VAEDecoder(nc=self.nc, ch_mults=self.ch_mults, nlayers_per_res=self.nlayers_per_res, nz=self.nz)

    def forward(self, x):
        z_params = self.encoder(x)
        mean, log_var = torch.split(z_params, self.nz, dim=1)
        z = self.sample(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var

    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(log_var)
        return mean + eps * std

class PercepModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.vgg = vgg16(pretrained=True)
        self.fmap_layers = (3, 8, 15, 22, 29) # layer indexes which contain perceptual feature map info
        
        # expects inputs to be between [-1, 1]
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

        blocks = []
        e_i = 0
        for layer_idx in self.fmap_layers:
            s_i, e_i = e_i, layer_idx
            blocks.append(self.vgg.features[s_i:e_i])
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        out = []
        x = (x - self.shift) / self.scale
        for block in self.blocks:
            x = block(x)
            out.append(x)
        return out

class Discriminator(nn.Module):
    '''
    just a normal PatchGAN discriminator model - copied from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
    '''
    def __init__(self, input_nc: int = 3, ndf: int = 64, n_layers: int = 3):
        '''
        input_nc: the number of channels in input images
        ndf: the number of filters in the last conv layer
        n_layers: the number of conv layers in the discriminator
        norm_layer: normalization layer
        '''
        super().__init__()
        use_bias = False # because batch norm after conv nullifies bias

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.GroupNorm(8, ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            nn.GroupNorm(8, ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, x):
        return self.main(x)

class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-base-patch32", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizerFast.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version).to(device)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

class TimestepEmbedSequential(nn.Sequential):
    '''
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    '''
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, (ResBlock)):
                x = layer(x, emb)
            elif isinstance(layer, TransformerBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(self,
        #in_c: int = 16,
        #nc: int = 256,
        #ch_mults: list = [1, 2, 4],
        in_c: int = 4,
        nc: int = 320,
        ch_mults: list = [1, 2, 4, 4],
        nlayers_per_res: int = 2,
        context_dim: int = 1024
    ):
        super().__init__()
        self.in_c = in_c
        self.nc = nc
        self.ch_mults = ch_mults
        self.nlayers_per_res = nlayers_per_res
        self.context_dim = context_dim
        self.temb_c = 4 * nc

        self.time_embed = TimeEmbedding(nc, self.temb_c)
        self.first_conv = nn.Conv2d(in_c, nc, 3, padding=1)

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
                #block.append(TransformerBlock(layer_out_c, layer_out_c, context_dim))
                block.append(TransformerBlock(layer_out_c, context_dim))
            if block_idx != len(ch_mults) - 1:
                block.append(Downsample(block_out_c))

            block = TimestepEmbedSequential(*block)
            self.downs.append(block)

        self.mid_block = TimestepEmbedSequential(
                ResBlock(block_out_c, block_out_c, self.temb_c),
                #TransformerBlock(block_out_c, block_out_c, context_dim),
                TransformerBlock(block_out_c, context_dim),
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
                #print('RB:', layer_in_c, layer_out_c)
                block.append(ResBlock(layer_in_c, layer_out_c, self.temb_c))
                #block.append(TransformerBlock(layer_out_c, layer_out_c, context_dim))
                block.append(TransformerBlock(layer_out_c, context_dim))
            if block_idx != 0:
                block.append(Upsample(block_out_c))

            block = TimestepEmbedSequential(*block)
            self.ups.append(block)

        self.out = nn.Sequential(
                nn.GroupNorm(8, block_out_c + nc),
                nn.ReLU(),
                nn.Conv2d(block_out_c + nc, in_c, 3, padding=1)
            )

    def forward(self, x, timesteps, context=None):
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
