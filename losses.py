from models import PercepModel, Discriminator
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

@torch.jit.script
def kl_divergence(mean, logvar):
    return 0.5 * torch.sum(mean.pow(2) + logvar.exp() - logvar - 1) / mean.shape[0]

@torch.jit.script
def adv_loss_fn(logits_real, logits_fake):
    loss_real = F.relu(1. - logits_real)
    loss_fake = F.relu(1. + logits_fake)
    d_loss = 0.5 * (loss_real + loss_fake).mean()
    return d_loss

class PercepLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.percep_model = PercepModel()
        self.percep_model.requires_grad_(False)
        self.percep_model.eval()

    def forward(self, y_pred, y):
        y_pred_fmaps = self.percep_model(y_pred)
        y_fmaps = self.percep_model(y)

        # pixelwise loss
        l1_loss = F.l1_loss(y_pred, y)

        # perceptual loss
        p_loss = torch.zeros((), device=y.device)
        for y_pred_fmap, y_fmap in zip(y_pred_fmaps, y_fmaps):
            p_loss += F.mse_loss(self.normalize_tensor(y_pred_fmap), self.normalize_tensor(y_fmap))

        return l1_loss, p_loss

    def normalize_tensor(self, x, eps=1e-6):
        norm_factor = torch.norm(x, dim=1, keepdim=True).detach()
        return x / (norm_factor + eps)
