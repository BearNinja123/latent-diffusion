from losses import kl_divergence, adv_loss_fn, PercepLoss
from models import VAE, FrozenCLIPEmbedder, UNet
from samplers import NoiseSchedule, DDIMSampler
from data import ImgTextDataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from multiprocessing import cpu_count
from transformers import logging
from torch.optim import Adam
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch, glob, os, gc, wandb

torch.backends.cudnn.benchmark = True
logging.set_verbosity_error() # disables the "Some weights failed to load" for the CLIP text model (non-issue, these failed weights are for the ViT and aren't used anyway)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
USE_WANDB = False

EPOCHS = 150
STEPS_PER_EPOCH = 1024
CROP_SIZE = 64
BATCH_SIZE = 1
USE_MIXED = False
IMG_FOLDER_PATH = 'dataset/imgs'
CAPTION_FILE = 'dataset/id_to_text.txt'

WANDB_RUN_PATH = WANDB_MODEL_FNAME = None
#WANDB_VAE_RUN_PATH = WANDB_VAE_MODEL_FNAME = None
WANDB_VAE_RUN_PATH = 'tiewa_enguin/ldm_vae/17jvvczf'
WANDB_MODEL_VAE_FNAME = 'vae_0100.pth'

#CKPT_DIR = 'checkpoints'
CKPT_DIR = ''
# save dirs may be different from load dirs if load dirs are unwritable (e.g. /kaggle/input)
CKPT_SAVE_DIR = 'checkpoints'

LEARNING_RATE = 4.5e-3 * BATCH_SIZE

def load_models(rank=0, lr=1e-4, ckpt_path=None, vae_ckpt_path=None):
    ddpm = UNet()
    vae = VAE().eval()
    text_embedder = FrozenCLIPEmbedder(device=rank)
    noise_schedule = NoiseSchedule()
    sampler = DDIMSampler(ddpm, noise_schedule=noise_schedule, tau_dim=50)
    opt = Adam(ddpm.parameters(), lr, (0.5, 0.9))
    scaler = GradScaler()
    resume_epoch_idx = 0
    resume_global_step_idx = 0

    if vae_ckpt_path is not None:
        if rank == 0:
            print(f'Loading VAE checkpoint from {vae_ckpt_path}')
        vae_ckpt = torch.load(vae_ckpt_path)
        vae.load_state_dict(vae_ckpt['vae_state_dict'])

    if ckpt_path is not None:
        if rank == 0:
            print(f'Loading checkpoint from {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        ddpm.load_state_dict(ckpt['ddpm_state_dict'])
        opt.load_state_dict(ckpt['opt_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        resume_epoch_idx = ckpt['epoch_idx']
        resume_global_step_idx = ckpt['global_step_idx']

    return (
            ddpm, vae, text_embedder,
            noise_schedule, sampler,
            opt, scaler,
            resume_epoch_idx, resume_global_step_idx
            )

class DDPMTrainer:
    def __init__(
        self,
        ddpm, vae, text_embedder,
        noise_schedule, sampler,
        dataset, dataloader,
        opt, use_mixed, scaler,
        img_size, ckpt_save_dir, rank,
        wandb_run
    ):
        self.rank = rank
        self.ddpm = DDP(ddpm.to(rank), device_ids=[rank])
        self.vae = vae.to(rank)
        self.text_embedder = text_embedder.to(rank)
        self.noise_schedule = noise_schedule.to(rank)
        self.sampler = sampler
        self.dataset = dataset
        self.dataloader = dataloader
        self.opt = opt
        self._opt_to_device(opt)
        self.use_mixed = use_mixed
        self.scaler = scaler
        self.ckpt_save_dir = ckpt_save_dir
        self.wandb_run = wandb_run
        self.z_std = None

        vae_downscale_factor = np.prod(self.vae.ch_mults)
        if isinstance(img_size, int):
            z_spatial = img_size // vae_downscale_factor
            self.z_shape = (self.vae.nz, z_spatial, z_spatial)
        else:
            z_h = img_size[0] // vae_downscale_factor
            z_w = img_size[1] // vae_downscale_factor
            self.z_shape = (self.vae.nz, z_h, z_w)

    def _opt_to_device(self, optimizer): # resolve https://github.com/pytorch/pytorch/issues/2830
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.rank)
    
    def dr(self, tensor, decimals=4): # detach and round
        if isinstance(tensor, torch.Tensor):
            return round(float(tensor.detach().cpu().numpy()), decimals)
        return round(tensor, decimals)

    def display_predictions(self, n_rows=1, n_cols=1,
            captions=None, display=True, save_path=None):
        n_imgs = n_rows * n_cols
        if captions is None:
            # get random captions from the training dataset
            img_idxs = np.random.randint(0, len(self.dataset), (n_imgs,))
            captions = []
            for idx in img_idxs:
                _img, caption = self.dataset[idx]
                captions.append(caption)
        text_embeddings = self.text_embedder(captions)

        z = torch.randn((n_imgs, *self.z_shape), device=self.rank)

        with torch.no_grad():
            with torch.autocast('cuda', enabled=self.use_mixed):
                z_denoised = self.sampler.get_samples(initial_x=z, context=text_embeddings)
                preds = self.vae.decoder(z_denoised)

        preds = preds.permute(0, 2, 3, 1).detach().float().cpu().clamp(-1, 1) * 0.5 + 0.5
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 10))

        for i in range(n_imgs):
            row, col = i // n_cols, i % n_cols
            if n_rows == n_cols == 1:
                ax.title.set_text(captions[i])
                ax.imshow(preds[i])
            elif n_rows == 1 or n_cols == 1:
                ax[i].title.set_text(captions[i])
                ax[i].imshow(preds[i])
            else:
                ax[row][col].title.set_text(captions[i])
                ax[row][col].imshow(preds[i])

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path)

        if self.wandb_run is not None:
            wandb.log({'preds': wandb.Image(fig)})

        if display:
            plt.show()

        plt.close('all')

    def save_models(self, epoch_idx=0, global_step_idx=0):
        ckpt_save_path = f'{self.ckpt_save_dir}/ddpm_{epoch_idx:0>4}.pth'
        torch.save({
            'ddpm_state_dict': self.ddpm.module.state_dict(),
            'opt_state_dict': self.opt.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'epoch_idx': epoch_idx,
            'global_step_idx': global_step_idx,
            }, ckpt_save_path)

        if self.wandb_run is not None:
            wandb.save(ckpt_save_path)

    def train_loop(self, epochs=1, steps_per_epoch=-1,
            save_every_n_epochs=1, resume_epoch_idx=0,
            resume_global_step_idx=0):
        dr = self.dr

        global_step_idx = resume_global_step_idx
        for epoch_idx in range(resume_epoch_idx, epochs):
            gc.collect()

            self.dataloader.sampler.set_epoch(epoch_idx)
            pbar = enumerate(self.dataloader)
            if self.rank == 0:
                pbar = tqdm(pbar, total=steps_per_epoch, position=0)

            for step_idx, (imgs, captions) in pbar:
                if step_idx == steps_per_epoch:
                    break

                imgs = imgs.to(self.rank).permute(0, 3, 1, 2).float() / 127.5 - 1.0

                self.opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', enabled=self.use_mixed):
                    z_params = self.vae.encoder(imgs)

                    mean, _log_var = torch.split(z_params, self.vae.nz, dim=1)
                    if self.z_std is None:
                        self.z_std = mean.std()
                    mean = mean / self.z_std

                    noised, timesteps, noise = self.noise_schedule(mean)
                    text_embeddings = self.text_embedder(captions)

                    noise_preds = self.ddpm(noised, timesteps, text_embeddings)
                    loss = F.mse_loss(noise_preds, noise)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()

                global_step_idx += 1

                if self.rank == 0:
                    pbar.set_description(f'Epoch {epoch_idx+1}/{epochs}, {step_idx+1}/{steps_per_epoch}: noise_loss: {dr(loss)}')
                if self.wandb_run is not None:
                    wandb.log(
                        {
                            'noise_loss': loss,
                            'loss_scale': self.scaler.get_scale(),
                        },
                        step=global_step_idx
                    )

            if (epoch_idx + 1) % save_every_n_epochs == 0 and self.rank == 0:
                self.save_models(epoch_idx=epoch_idx+1, global_step_idx=global_step_idx)
                self.display_predictions(save_path=f'predictions/out_{epoch_idx+1:0>4}.jpg')

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
    crop_size: int, batch_size: int, use_mixed: bool,
    learning_rate: float, img_folder_path: str, caption_file: str,
    ckpt_save_dir: str, ckpt_path: str = None,
    vae_ckpt_path: str = None
):
    ddp_setup(rank, world_size)
    load_model_return_vals = load_models(
            rank=rank, lr=learning_rate,
            ckpt_path=ckpt_path, vae_ckpt_path=vae_ckpt_path)
    ddpm, vae, text_embedder = load_model_return_vals[0:3]
    noise_schedule, sampler = load_model_return_vals[3:5]
    opt, scaler = load_model_return_vals[5:7]
    resume_epoch_idx, resume_global_step_idx = load_model_return_vals[7:9]

    # persistent_workers - see https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778/7
    dataset = ImgTextDataset(img_folder_path, caption_file, crop_size)
    num_dataloader_workers = cpu_count() // torch.cuda.device_count()
    dataloader = DataLoader(dataset, batch_size // world_size, shuffle=False,
            pin_memory=True, num_workers=num_dataloader_workers,
            persistent_workers=num_dataloader_workers, sampler=DistributedSampler(dataset))

    wandb_run = None
    if rank == 0 and USE_WANDB:
        wandb_run = wandb.init(project='ldm_vae')

    trainer = DDPMTrainer(
        ddpm, vae, text_embedder,
        noise_schedule, sampler,
        dataset, dataloader,
        opt, use_mixed, scaler,
        crop_size, ckpt_save_dir, rank,
        wandb_run
    )

    trainer.display_predictions(save_path=f'predictions/out_initial.jpg')
    trainer.train_loop(epochs=epochs, steps_per_epoch=steps_per_epoch,
        resume_epoch_idx=resume_epoch_idx, resume_global_step_idx=resume_global_step_idx)

    if rank == 0 and USE_WANDB:
        wandb.finish()

    destroy_process_group()

if __name__ == '__main__':
    if USE_WANDB and WANDB_RUN_PATH is not None: # load DDPM checkpoint from wandb
        wandb.restore(WANDB_MODEL_FNAME, run_path=WANDB_RUN_PATH, root=CKPT_DIR)
    if USE_WANDB and WANDB_VAE_RUN_PATH is not None: # load VAE checkpoint from wandb
        wandb.restore(WANDB_VAE_MODEL_FNAME, run_path=WANDB_VAE_RUN_PATH, root=CKPT_DIR)

    # get most recent VAE checkpoint for resuming training
    VAE_CKPT_PATH = sorted(glob.glob(f'{CKPT_DIR}/vae_*.pth'))
    if VAE_CKPT_PATH != []:
        VAE_CKPT_PATH = VAE_CKPT_PATH[-1]
    else:
        VAE_CKPT_PATH = None
        print('Warning: no pretrained VAE found.')

    # get most recent DDPM checkpoint for resuming training
    CKPT_PATH = sorted(glob.glob(f'{CKPT_DIR}/ddpm_*.pth'))
    if CKPT_PATH != []:
        CKPT_PATH = CKPT_PATH[-1]
    else:
        CKPT_PATH = None
    os.makedirs(CKPT_SAVE_DIR, exist_ok=True)

    world_size = torch.cuda.device_count()
    train_args = (
        world_size,
        EPOCHS, STEPS_PER_EPOCH,
        CROP_SIZE, BATCH_SIZE, USE_MIXED,
        LEARNING_RATE, IMG_FOLDER_PATH, CAPTION_FILE,
        CKPT_SAVE_DIR, CKPT_PATH,
        VAE_CKPT_PATH
    )

    mp.spawn(train, args=train_args, nprocs=world_size)
