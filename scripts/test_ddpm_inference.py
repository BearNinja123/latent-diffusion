import sys
sys.path.append('.') # for python ./scripts/file.py
sys.path.append('..') # for python file.py within ./scripts dir

from modules.models import VAE, UNet, FrozenCLIPEmbedder
from modules.samplers import NoiseSchedule, DDIMSampler
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch, time, os

def load_img(img_fname, size=None):
    img = Image.open(img_fname)
    if size is not None:
        img = img.resize(size)
    x = np.array(img)
    x = (torch.from_numpy(x).permute(2, 0, 1)[None] / 127.5 - 1.0)
    #print(f'mean: {x.mean()}, std: {x.std()}')
    return x

def tn(tensor, idx=0):
    if idx is None:
        return tensor.float().cpu().permute(0, 2, 3, 1).numpy() * 0.5 + 0.5
    return tensor.float().cpu().clamp(-1, 1).permute(0, 2, 3, 1).numpy()[idx] * 0.5 + 0.5

def get_loss_curve(img, ddpm, timesteps): # track DM's MSE loss of an image through time
    mean, log_var = torch.split(vae.encoder(img.to(device)), vae.nz, dim=1)
    z = vae.sample(mean, log_var)
    z_std = z.std()
    z /= z_std
    noise = torch.randn_like(z)
    losses = []
    for t_idx, t in enumerate(timesteps):
        noised = z * nsr.signal_stds[t] + noise * nsr.noise_stds[t]

        noise_pred = ddpm(noised, torch.tensor([t], dtype=torch.long, device=device))
        loss = F.mse_loss(noise_pred, noise).item()
        losses.append(loss)

        #fig, ax = plt.subplots(2, 2)
        #ax[0][0].imshow(noised.cpu().permute(0, 2, 3, 1).numpy()[0, :, :, :3]*0.225+0.5)
        #ax[0][1].imshow(noise.cpu().permute(0, 2, 3, 1).numpy()[0, :, :, :3]*0.225+0.5)
        #ax[1][0].imshow(vae.decoder(noised*z_std).cpu().permute(0, 2, 3, 1)[0].float()*0.5+0.5)
        #pred_x0 = (noised - (1 - nsr.signal_stds[t])**0.5 * noise_pred) / (nsr.signal_stds[t])**0.5
        #ax[1][1].imshow(vae.decoder(pred_x0*z_std).cpu().permute(0, 2, 3, 1)[0].float()*0.5+0.5)
        #ax[0][0].set_title(f'T: {t}, signal: {nsr.signal_stds[t].item()}, noise: {nsr.noise_stds[t].item()}')
        #plt.show()
        #plt.close('all')

    return losses

get_params = lambda model: sum(p.numel() for p in model.parameters())

N_ROWS, N_COLS = 2, 4
device = 'cuda'
with torch.no_grad():
    with torch.amp.autocast(device):
        homedir = os.path.expanduser('~')
        img_fname = f'{homedir}/Projects/test.jpg'
        x1 = load_img(img_fname).to(device)

        sd = torch.load('../trained_models/vae_0029_stable_norm.pth')['vae_state_dict']
        vae = VAE()
        print(get_params(vae))
        vae.load_state_dict(sd, strict=True)
        vae = vae.to(device)
        vae.eval()

        ddpm = UNet()
        sd = torch.load(f'../trained_models/ddpm_0044.pth')['ddpm_state_dict']
        ddpm.load_state_dict(sd, strict=True)
        ddpm = ddpm.to(device)
        ddpm.eval()
        
        nsr = NoiseSchedule().to(device)
        sampler = DDIMSampler(ddpm, nsr, tau_dim=20)

        samp_models = [44]
        opacities = np.linspace(0, 1, len(samp_models)+1)[1:]
        img_fnames = ['test.jpg', 'deer256.jpg', 'cake.jpg']
        colors = ['blue', 'red', 'green', 'orange', 'pink', 'black'][:len(img_fnames)]
        for mv, opacity in zip(samp_models, opacities):
            #sd = torch.load(f'../trained_models/ddpm_{mv:0>4}_fixed.pth')['ddpm_state_dict']
            #ddpm.load_state_dict(sd, strict=True)
            #ddpm = ddpm.to(device)
            #ddpm.eval()
            #ddpm = torch.compile(ddpm)

            nsr = NoiseSchedule().to(device)
            sampler = DDIMSampler(ddpm, nsr, tau_dim=50)

            for img_fname, color in zip(img_fnames, colors):
                losses = get_loss_curve(load_img(f'{homedir}/Projects/{img_fname}'), ddpm, sampler.tau);
                ts = range(len(losses))
                plt.plot(ts, losses, label=f'{mv}_{img_fname}', color=color, alpha=opacity)
        plt.legend()
        plt.show()

        raise

        text_embedder = FrozenCLIPEmbedder()

        while True:
            prompt = input('Enter prompt: ')
            initial_x = torch.randn((N_ROWS * N_COLS, 4, 32, 32), device=device)

            tic = time.time()
            text_embedding = text_embedder([prompt])
            print(f'Time for CLIP inference: {time.time()-tic:.4f} s')

            tic = time.time()
            z = sampler.get_samples(initial_x=initial_x, context=text_embedding, cfg_weight=5) * 1.2
            print(f'Time for DM inference of {initial_x.shape} shape tensor: {time.time()-tic:.4f} s')

            tic = time.time()
            ypred = vae.decoder(z)
            torch.cuda.synchronize()
            print(f'Time for VAE decoder inference of {z.shape} shape tensor: {time.time()-tic:.4f} s')
            #Image.fromarray((255*tn(ypred)).astype(np.uint8)).save('/tmp/ldm_img.png')

            ypred_show = tn(ypred, idx=None)
            fig, ax = plt.subplots(N_ROWS, N_COLS)
            for i in range(N_ROWS * N_COLS):
                ax[i // N_COLS][i % N_COLS].imshow(ypred_show[i])
            plt.show()
