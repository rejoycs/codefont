import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --------------------------------------------------
# Perceptual Loss
# --------------------------------------------------
class PerceptualLoss(nn.Module):
    def __init__(self, mode='L1'):
        super().__init__()

        vgg = models.vgg19(weights="IMAGENET1K_V1").features

        self.blocks = nn.ModuleList([
            vgg[:4],    # relu1_2
            vgg[4:9],   # relu2_2
            vgg[9:18],  # relu3_4
            vgg[18:27]  # relu4_4
        ])

        for b in self.blocks:
            for p in b.parameters():
                p.requires_grad = False

        self.mode = mode

        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])

        self.register_buffer("mean", mean.view(1,3,1,1))
        self.register_buffer("std", std.view(1,3,1,1))

        self.weights = [1.0, 0.75, 0.5, 0.25]

    def forward(self, x, y):
        x = (x + 1) * 0.5
        y = (y + 1) * 0.5

        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        loss = 0.0

        for w, block in zip(self.weights, self.blocks):
            x = block(x)
            y = block(y)

            if self.mode == "L1":
                loss += w * F.l1_loss(x, y)
            else:
                loss += w * F.mse_loss(x, y)

        return loss
