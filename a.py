from models import VAE
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch, time

def load_img(img_fname, size=None):
    img = Image.open(img_fname)
    if size is not None:
        img = img.resize(size)
    x = np.array(img)
    x = torch.from_numpy(x).permute(2, 0, 1).float()[None] / 127.5 - 1.0
    return x

def tn(tensor, idx=0):
    if idx is None:
        return tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    return tensor.detach().cpu().clamp(-1, 1).permute(0, 2, 3, 1).numpy()[idx] * 0.5 + 0.5

device = 'cpu'
#img_fname = '../test.jpg'
img_fname = 'dataset/imgs/000312516.jpg'
#x1, x2 = load_img(img_fname), load_img(img_fname2)
#x1 = load_img(img_fname, size=(1632, 1224)).to(device)
x1 = load_img(img_fname).to(device)

sd = torch.load('checkpoints/vae_0100.pth')['vae_state_dict']
vae = VAE().to(device)
vae.load_state_dict(sd, strict=True)
vae.eval()

def show_latent_fmaps(mean):
    fig, ax = plt.subplots(4,4, figsize=(30, 30))
    mean = mean.detach().cpu().numpy()[0]
    for i in range(4):
        for j in range(4):
            ax[i][j].imshow(mean[4*i+j])
    plt.show()

def show_random_sample(mean, log_var):
    std = torch.exp(0.5 * log_var).to(device)
    eps = torch.randn_like(log_var).to(device)
    z = mean + eps * std
    out = vae.decoder(z)
    plt.imshow(tn(out))
    plt.show()

ss = time.time()
mean, log_var = torch.split(vae.encoder(x1), 16, dim=1)
z = vae.sample(mean, log_var)
print(z.std())
#ypred, mean, log_var = vae(x1)
print(f'Time for VAE inference of {x1.shape} shape tensor: {time.time()-ss:.4f} s')
plt.imshow(tn(ypred)); plt.show()

#show_random_sample(torch.zeros((1, 16, 2, 2)), 0*torch.ones((1, 16, 2, 2)))
