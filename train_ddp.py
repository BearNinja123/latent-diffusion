from losses import kl_divergence, adv_loss_fn, PercepLoss
from models import VAE, Discriminator
from data import ImgTextDataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from torch.optim import Adam
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, glob, os

EPOCHS = 10
STEPS_PER_EPOCH = 16
BATCH_SIZE = 1
CROP_SIZE = 64
IMG_FOLDER_PATH = 'dataset/imgs'
CAPTION_FILE = 'dataset/id_to_text.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CKPT_DIR = 'checkpoints'
# save dirs may be different from load dirs if load dirs are unwritable (e.g. /kaggle/input)
CKPT_SAVE_DIR = 'checkpoints'
# get most recent VAE checkpoint for resuming training
CKPT_PATH = sorted(glob.glob(f'{CKPT_DIR}/vae_*.pth'))
if CKPT_PATH != []:
    CKPT_PATH = CKPT_PATH[-1]
else:
    CKPT_PATH = None
os.makedirs(CKPT_SAVE_DIR, exist_ok=True)

LEARNING_RATE = 4.5e-6 * BATCH_SIZE
ADV_STEP_THRESHOLD = 16 # number of steps of VAE training before adversarial training

def load_models(rank=0, lr=1e-4, ckpt_path=None):
    vae = VAE()
    disc = Discriminator()
    p_loss_model = PercepLoss()
    vae_opt = Adam(vae.parameters(), lr, (0.5, 0.9))
    disc_opt = Adam(disc.parameters(), lr, (0.5, 0.9))
    resume_epoch_idx = 0
    resume_global_step_idx = 0
    if ckpt_path is not None:
        if rank == 0:
            print(f'Loading checkpoint from {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        vae.load_state_dict(ckpt['vae_state_dict'])
        disc.load_state_dict(ckpt['disc_state_dict'])
        vae_opt.load_state_dict(ckpt['vae_opt_state_dict'])
        disc_opt.load_state_dict(ckpt['disc_opt_state_dict'])
        resume_epoch_idx = ckpt['epoch_idx']
        resume_global_step_idx = ckpt['global_step_idx']

    return vae, disc, p_loss_model, vae_opt, disc_opt, resume_epoch_idx, resume_global_step_idx

class AutoencoderTrainer:
    def __init__(
        self,
        vae, disc, p_loss_model,
        dataset, dataloader, vae_opt, disc_opt,
        adv_step_threshold, rank):
        self.rank = rank
        self.vae = DDP(vae.to(rank), device_ids=[rank])
        self.disc = DDP(disc.to(rank), device_ids=[rank])
        self.p_loss_model = p_loss_model.to(rank)
        self.dataset = dataset
        self.dataloader = dataloader
        self.vae_opt = vae_opt
        self._opt_to_device(vae_opt)
        self.disc_opt = disc_opt
        self._opt_to_device(disc_opt)
        self.adv_step_threshold = adv_step_threshold

    def _opt_to_device(self, optimizer): # resolve https://github.com/pytorch/pytorch/issues/2830
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.rank)
    
    def dr(self, tensor, decimals=4): # detach and round
        if isinstance(tensor, torch.Tensor):
            return round(float(tensor.detach().cpu().numpy()), decimals)
        return round(tensor, decimals)

    def display_vae_predictions(self, n_imgs=5,
            model_input=None, display=True, save_path=None):
        if model_input is None:
            # get random images from the training dataset
            img_idxs = np.random.randint(0, len(self.dataset), (n_imgs,))
            imgs = []
            for idx in img_idxs:
                img, _caption = self.dataset[idx]
                imgs.append(img)
            model_input = torch.tensor(np.array(imgs)).permute(0, 3, 1, 2).to(self.rank)

        toggle = self.vae.training
        if toggle:
            self.vae.eval()
        preds, _mean, _log_var = self.vae(model_input)
        if toggle:
            self.vae.train()

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

    def save_models(self, epoch_idx=0, global_step_idx=0):
        ckpt_save_path = f'{CKPT_SAVE_DIR}/vae_{epoch_idx:0>4}.pth'
        torch.save({
            'vae_state_dict': self.vae.module.state_dict(),
            'disc_state_dict': self.disc.module.state_dict(),
            'vae_opt_state_dict': self.vae_opt.state_dict(),
            'disc_opt_state_dict': self.disc_opt.state_dict(),
            'epoch_idx': epoch_idx,
            'global_step_idx': global_step_idx,
            }, ckpt_save_path)

    def train_loop(self, epochs=1, steps_per_epoch=-1,
            save_every_n_epochs=1, resume_epoch_idx=0,
            resume_global_step_idx=0):
        dr = self.dr
        kl_weight, l1_weight, p_weight, adv_weight = 1e-6, 1.0, 1.0, 0.5

        global_step_idx = resume_global_step_idx
        img, _caption = self.dataset[0]
        batch = torch.tensor(img[None]).permute(0, 3, 1, 2).to(self.rank)
        for epoch_idx in range(resume_epoch_idx, epochs):
            self.dataloader.sampler.set_epoch(epoch_idx)
            pbar = enumerate(self.dataloader)
            pbar2 = None
            if self.rank == 0:
                pbar = tqdm(pbar, total=steps_per_epoch, position=0)
                pbar2 = tqdm(total=steps_per_epoch, position=1)

            for step_idx, (imgs, _captions) in pbar:
                if step_idx == steps_per_epoch:
                    break

                imgs = imgs.to(self.rank).permute(0, 3, 1, 2)
                self.vae_opt.zero_grad()
                self.disc_opt.zero_grad()

                #with torch.autocast('cuda', torch.float16):
                if True:
                    # G update
                    preds, mean, log_var = self.vae(imgs)

                    kl_loss = kl_weight * kl_divergence(mean, log_var)
                    l1_loss, p_loss = self.p_loss_model(preds, imgs)
                    l1_loss = l1_weight * l1_loss
                    p_loss = p_weight * p_loss
                    rec_loss = l1_loss + p_loss

                    if global_step_idx < self.adv_step_threshold: # introduce discriminator in VAE training later
                        g_adv_loss = 0.0
                    else:
                        logits_fake = self.disc(preds)
                        g_adv_loss = -torch.mean(logits_fake)
                        last_layer = self.vae.module.decoder.last_conv.parameters()
                        reconstruction_grad = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
                        last_layer = self.vae.module.decoder.last_conv.parameters()
                        adv_grad = torch.autograd.grad(g_adv_loss, last_layer, retain_graph=True)[0]
                        adv_adaptive_weight = torch.norm(reconstruction_grad) / (torch.norm(adv_grad) + 1e-6)
                        adv_adaptive_weight = torch.clamp(adv_adaptive_weight, 0.0, 1e4).detach()
                        g_adv_loss *= adv_weight * adv_adaptive_weight

                    loss = kl_loss + g_adv_loss + rec_loss
                    loss.backward()
                    self.vae_opt.step()

                    global_step_idx += 1

                    # D update (if generator finishes initial training)
                    if global_step_idx < self.adv_step_threshold:
                        if self.rank == 0:
                            pbar.set_description(f'Epoch {epoch_idx+1}/{epochs}, {step_idx+1}/{steps_per_epoch}: percep: {dr(p_loss)}, l1: {dr(l1_loss)}, kl: {dr(kl_loss)}, adv: {dr(g_adv_loss)}')
                        continue
                    logits_real = self.disc(imgs)
                    logits_fake = self.disc(preds.detach())
                    d_adv_loss = adv_loss_fn(logits_real, logits_fake)
                    d_adv_loss.backward()
                    self.disc_opt.step()

                    if self.rank == 0:
                        pbar.set_description(f'Epoch {epoch_idx+1}/{epochs}, {step_idx+1}/{steps_per_epoch}: percep: {dr(p_loss)}, l1: {dr(l1_loss)}, kl: {dr(kl_loss)}, g_adv: {dr(g_adv_loss)}')
                        pbar2.set_description(f'd_adv: {dr(d_adv_loss)}, logits_real: {dr(logits_real.mean())}, logits_fake: {dr(logits_fake.mean())}')

            if (epoch_idx + 1) % save_every_n_epochs == 0 and self.rank == 0:
                self.save_models(epoch_idx=epoch_idx+1, global_step_idx=global_step_idx)
                self.display_vae_predictions(save_path=f'predictions/out_{epoch_idx+1:0>4}.jpg')

def ddp_setup(rank: int, world_size: int):
    '''
    rank: GPU ID
    world_size: number of GPUs
    '''
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train(
    rank: int, world_size: int,
    epochs: int, steps_per_epoch: int,
    crop_size: int, batch_size: int,
    learning_rate: float, adv_step_threshold: int,
    img_folder_path: str, caption_file: str,
    ckpt_path: str = None):
    ddp_setup(rank, world_size)
    vae, disc, p_loss_model, vae_opt, disc_opt, resume_epoch_idx, resume_global_step_idx = load_models(rank=rank, lr=learning_rate, ckpt_path=ckpt_path)
    # persistent_workers - see https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778/7
    dataset = ImgTextDataset(img_folder_path, caption_file, crop_size)
    dataloader = DataLoader(dataset, batch_size, shuffle=False,
            pin_memory=True, num_workers=cpu_count(),
            persistent_workers=cpu_count(), sampler=DistributedSampler(dataset))

    trainer = AutoencoderTrainer(vae, disc, p_loss_model,
            dataset, dataloader, vae_opt, disc_opt, adv_step_threshold, rank)
    trainer.display_vae_predictions(save_path=f'predictions/out_initial.jpg')
    trainer.train_loop(epochs=epochs, steps_per_epoch=steps_per_epoch,
        resume_epoch_idx=resume_epoch_idx, resume_global_step_idx=resume_global_step_idx)
    destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    train_args = (
        world_size, EPOCHS, STEPS_PER_EPOCH,
        CROP_SIZE, BATCH_SIZE,
        LEARNING_RATE, ADV_STEP_THRESHOLD,
        IMG_FOLDER_PATH, CAPTION_FILE, CKPT_PATH
    )

    mp.spawn(train, args=train_args, nprocs=world_size)
