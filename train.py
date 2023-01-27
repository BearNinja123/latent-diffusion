from losses import kl_divergence, adv_loss_fn, PercepLoss
from models import VAE, Discriminator
from data import ImgTextDataset
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from torch.optim import Adam
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, glob, os

EPOCHS = 10
STEPS_PER_EPOCH = 16
RESUME_EPOCH_IDX = RESUME_GLOBAL_STEP_IDX = 0
BATCH_SIZE = 1
CROP_SIZE = 128
IMG_FOLDER_PATH = 'dataset/imgs'
CAPTION_FILE = 'dataset/id_to_text.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPT_DIR = 'checkpoints'
# save dirs may be different from load dirs if load dirs are unwritable (e.g. /kaggle/input)
MODEL_SAVE_DIR = 'pretrained'
CKPT_SAVE_DIR = 'checkpoints'
# get most recent VAE checkpoint for resuming training
CKPT_PATH = sorted(glob.glob(f'{CKPT_DIR}/vae_*.pth'))
if CKPT_PATH != []:
    RESUME = True
    CKPT_PATH = CKPT_PATH[-1]
else:
    RESUME = False
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(CKPT_SAVE_DIR, exist_ok=True)

LEARNING_RATE = 4.5e-6 * BATCH_SIZE
ADV_STEP_THRESHOLD = 10000 # number of steps of VAE training before adversarial training

vae = VAE().to(device)
disc = Discriminator().to(device)
p_loss_model = PercepLoss().to(device)
vae_opt = Adam(vae.parameters(), LEARNING_RATE, (0.5, 0.9))
disc_opt = Adam(disc.parameters(), LEARNING_RATE, (0.5, 0.9))
if RESUME:
    print(f'Loading checkpoint from {CKPT_PATH}')
    ckpt = torch.load(CKPT_PATH)
    vae.load_state_dict(ckpt['vae_state_dict'])
    disc.load_state_dict(ckpt['disc_state_dict'])
    vae_opt.load_state_dict(ckpt['vae_opt_state_dict'])
    disc_opt.load_state_dict(ckpt['disc_opt_state_dict'])
    RESUME_EPOCH_IDX = ckpt['epoch_idx']
    RESUME_GLOBAL_STEP_IDX = ckpt['global_step_idx']

dataset = ImgTextDataset(IMG_FOLDER_PATH, CAPTION_FILE, crop_size=CROP_SIZE)
dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=cpu_count())

def dr(tensor, decimals=4): # detach and round
    if isinstance(tensor, torch.Tensor):
        return round(float(tensor.detach().cpu().numpy()), decimals)
    return round(tensor, decimals)

def display_vae_predictions(model=vae, dataset=dataset, n_imgs=5,
        model_input=None, display=True, save_path=None):
    if model_input is None:
        # get random images from the training dataset
        img_idxs = np.random.randint(0, len(dataset), (n_imgs,))
        imgs = []
        for idx in img_idxs:
            img, _caption = dataset[idx]
            imgs.append(img)
        model_input = torch.tensor(np.array(imgs)).permute(0, 3, 1, 2).to(device)

    model.eval()
    preds, _mean, _log_var = model(model_input)
    preds = preds.permute(0, 2, 3, 1).detach().cpu().clamp(0, 1)
    y = model_input.permute(0, 2, 3, 1).cpu()

    fig, ax = plt.subplots(2, n_imgs, figsize=(20, 10))
    for i in range(n_imgs):
        if n_imgs == 1:
            ax[0].set_ylabel('Input')
            ax[1].set_ylabel('VAE Reconstruction')
            ax[0].imshow(y[i])
            ax[1].imshow(preds[i])
        else:
            ax[0][0].set_ylabel('Input')
            ax[1][0].set_ylabel('VAE Reconstruction')
            ax[0][i].imshow(y[i])
            ax[1][i].imshow(preds[i])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)

    if display:
        plt.show()

def save_models(epoch_idx=0, global_step_idx=0):
    model_save_path = f'{MODEL_SAVE_DIR}/vae_{epoch_idx:0>4}.pth'
    ckpt_save_path = f'{CKPT_SAVE_DIR}/vae_{epoch_idx:0>4}.pth'
    torch.save(vae.state_dict(), model_save_path)
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'vae_opt_state_dict': vae_opt.state_dict(),
        'disc_opt_state_dict': disc_opt.state_dict(),
        'epoch_idx': epoch_idx,
        'global_step_idx': global_step_idx,
        }, ckpt_save_path)

def train_loop(vae=vae, disc=disc, p_loss_model=p_loss_model,
        dataloader=dataloader,
        vae_opt=vae_opt, disc_opt=disc_opt,
        epochs=1, steps_per_epoch=-1,
        save_every_n_epochs=1,
        resume_epoch_idx=0, resume_global_step_idx=0):

    kl_weight, l1_weight, p_weight, adv_weight = 1e-6, 1.0, 1.0, 0.5

    global_step_idx = resume_global_step_idx
    img, _caption = dataset[0]
    batch = torch.tensor(img[None]).permute(0, 3, 1, 2).to(device)
    for epoch_idx in range(resume_epoch_idx, epochs):
        pbar = tqdm(enumerate(dataloader), total=steps_per_epoch, position=0)
        pbar2 = tqdm(total=steps_per_epoch, position=1)
        for step_idx, (imgs, _captions) in pbar:
            if step_idx == steps_per_epoch:
                break

            imgs = imgs.to(device).permute(0, 3, 1, 2)
            vae_opt.zero_grad()
            disc_opt.zero_grad()

            # G update
            preds, mean, log_var = vae(imgs)

            kl_loss = kl_weight * kl_divergence(mean, log_var)
            l1_loss, p_loss = p_loss_model(preds, imgs)
            l1_loss = l1_weight * l1_loss
            p_loss = p_weight * p_loss
            rec_loss = l1_loss + p_loss

            if global_step_idx < ADV_STEP_THRESHOLD: # introduce discriminator in VAE training later
                g_adv_loss = 0.0
            else:
                logits_fake = disc(preds)
                g_adv_loss = -torch.mean(logits_fake)
                last_layer = vae.decoder.last_conv.parameters()
                reconstruction_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
                last_layer = vae.decoder.last_conv.parameters()
                adv_grad = torch.autograd.grad(g_adv_loss, last_layer, retain_graph=True)[0]
                adv_adaptive_weight = torch.norm(reconstruction_grad) / (torch.norm(adv_grad) + 1e-6)
                adv_adaptive_weight = torch.clamp(adv_adaptive_weight, 0.0, 1e4).detach()
                g_adv_loss *= adv_weight * adv_adaptive_weight

            loss = kl_loss + g_adv_loss + rec_loss
            loss.backward()
            vae_opt.step()

            global_step_idx += 1

            # D update (if generator finishes initial training)
            if global_step_idx < ADV_STEP_THRESHOLD:
                pbar.set_description(f'Epoch {epoch_idx+1}/{epochs}, {step_idx+1}/{steps_per_epoch}: percep: {dr(p_loss)}, l1: {dr(l1_loss)}, kl: {dr(kl_loss)}, adv: {dr(g_adv_loss)}')
                continue
            logits_real = disc(imgs)
            logits_fake = disc(preds.detach())
            d_adv_loss = adv_loss_fn(logits_real, logits_fake)
            d_adv_loss.backward()
            disc_opt.step()
            pbar.set_description(f'Epoch {epoch_idx+1}/{epochs}, {step_idx+1}/{steps_per_epoch}: percep: {dr(p_loss)}, l1: {dr(l1_loss)}, kl: {dr(kl_loss)}, g_adv: {dr(g_adv_loss)}')
            pbar2.set_description(f'd_adv: {dr(d_adv_loss)}, logits_real: {dr(logits_real.mean())}, logits_fake: {dr(logits_fake.mean())}')

        if (epoch_idx + 1) % save_every_n_epochs == 0:
            save_models(epoch_idx=epoch_idx+1, global_step_idx=global_step_idx)
            display_vae_predictions(save_path=f'predictions/out_{epoch_idx+1:0>4}.jpg')

display_vae_predictions(save_path=f'predictions/out_initial.jpg')
train_loop(epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH,
        resume_epoch_idx=RESUME_EPOCH_IDX, resume_global_step_idx=RESUME_GLOBAL_STEP_IDX)
