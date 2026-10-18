import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ======================================================
class NetD(nn.Module):
    """
    Improved Multi-Task PatchGAN Discriminator

    Upgrades:
    ✔ Spectral normalization
    ✔ No BatchNorm leakage
    ✔ Stronger classification heads
    ✔ Stabilized adversarial head
    """

    def __init__(self,
                 in_channels=3,
                 ndf=64,
                 num_styles=500,
                 num_contents=30,
                 num_blocks=4):

        super().__init__()

        # --------------------------------------------------
        # Shared trunk
        # --------------------------------------------------
        layers = [
            spectral_norm(nn.Conv2d(in_channels, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf = ndf
        for i in range(num_blocks):

            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 2 if i < num_blocks - 1 else 1

            layers.append(
                spectral_norm(nn.Conv2d(nf_prev, nf, 4, stride, 1))
            )
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.backbone = nn.Sequential(*layers)

        # --------------------------------------------------
        # Adversarial head
        # --------------------------------------------------
        self.adv_head = spectral_norm(
            nn.Conv2d(nf, 1, 3, 1, 1)
        )

        # --------------------------------------------------
        # Projection heads for classification
        # --------------------------------------------------
        hidden = max(nf // 2, 128)

        self.style_head = nn.Sequential(
            spectral_norm(nn.Linear(nf, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden, num_styles))
        )

        self.content_head = nn.Sequential(
            spectral_norm(nn.Linear(nf, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden, num_contents))
        )


    # --------------------------------------------------
    def forward(self, x):

        feat = self.backbone(x)

        adv_out = self.adv_head(feat)

        pooled = torch.mean(feat, dim=[2,3])

        style_logits = self.style_head(pooled)
        content_logits = self.content_head(pooled)

        return adv_out, style_logits, content_logits
