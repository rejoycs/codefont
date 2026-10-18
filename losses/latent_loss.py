import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL

class LatentLoss(nn.Module):

    def __init__(
        self,
        model_name="stabilityai/sd-vae-ft-mse",
        device="cuda"
    ):
        super().__init__()

        self.vae = AutoencoderKL.from_pretrained(
            model_name
        ).to(device)

        self.vae.eval()

        for p in self.vae.parameters():
            p.requires_grad = False

        self.scale = self.vae.config.scaling_factor

    def encode_latent(self, x):

        z = self.vae.encode(x).latent_dist.mean
        z = z * self.scale

        return z

    def forward(self, fake, real):

        z_fake = self.encode_latent(fake)

        with torch.no_grad():
            z_real = self.encode_latent(real)

        loss = F.l1_loss(z_fake, z_real)
        return loss