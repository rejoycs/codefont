import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops


# ======================================================
# Deformable Conv
# ======================================================
class DeformableConvNet(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()

        self.weight_conv = nn.Conv2d(
            in_c, out_c, k, s, p
        )

        self.offset_conv = nn.Conv2d(
            in_c, 2 * k * k, k, s, p
        )

        self.s = s
        self.p = p

    def forward(self, x):

        offset = self.offset_conv(x)
        offset = torch.clamp(offset, -2.0, 2.0)

        return ops.deform_conv2d(
            x,
            offset,
            self.weight_conv.weight,
            self.weight_conv.bias,
            stride=self.s,
            padding=self.p
        )


# ======================================================
# Residual Block
# ======================================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(16, channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(16, channels),
        )

    def forward(self, x):
        return F.relu(self.block(x) + x, inplace=True)


# ======================================================
# Backbone
# ======================================================
class MultiScaleBackbone(nn.Module):
    def __init__(self, in_channels=3, ngf=64):
        super().__init__()

        self.l1 = nn.Sequential(
            DeformableConvNet(in_channels, ngf),
            nn.GroupNorm(8, ngf),
            nn.ReLU(inplace=True)
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, 4, 2, 1),
            nn.GroupNorm(16, ngf*2),
            nn.ReLU(inplace=True),
            ResidualBlock(ngf*2)
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, 4, 2, 1),
            nn.GroupNorm(16, ngf*4),
            nn.ReLU(inplace=True),
            ResidualBlock(ngf*4)
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(ngf*4, ngf*8, 4, 2, 1),
            nn.GroupNorm(16, ngf*8),
            nn.ReLU(inplace=True),
            ResidualBlock(ngf*8)
        )

        self.l5 = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*16, 4, 2, 1),
            nn.GroupNorm(16, ngf*16),
            nn.ReLU(inplace=True),
            ResidualBlock(ngf*16)
        )

    def forward(self, x):

        x = self.l1(x)
        x = self.l2(x)

        f3 = self.l3(x)
        f4 = self.l4(f3)
        f5 = self.l5(f4)

        return [f3, f4, f5]


# ======================================================
# GeM Pooling
# ======================================================
class GeMPool(nn.Module):
    def __init__(self, p=3.0):
        super().__init__()

        self.p = nn.Parameter(
            torch.ones(1) * p
        )

    def forward(self, x):

        p = torch.clamp(self.p, min=1.0)

        return (
            x.clamp(min=1e-6)
             .pow(p)
             .mean(dim=[2, 3])
             .pow(1.0 / p)
        )


# ======================================================
# Content Head
# ======================================================
class ContentHead(nn.Module):
    def __init__(self, in_channels, dim=256):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, dim, 1),
            nn.GroupNorm(16, dim),
            nn.ReLU(inplace=True)
        )

        self.pool = GeMPool()

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def forward(self, f):

        f = self.proj(f)

        emb = self.pool(f)

        emb = self.mlp(emb)

        return F.normalize(emb, dim=1)


# ======================================================
# Content Encoder
# ======================================================
class ContentEncoder(nn.Module):

    def __init__(
        self,
        in_channels=3,
        ngf=64,
        dim=256
    ):
        super().__init__()

        self.backbone = MultiScaleBackbone(
            in_channels,
            ngf
        )

        self.content_heads = nn.ModuleList([
            ContentHead(ngf * 4, dim),
            ContentHead(ngf * 8, dim),
            ContentHead(ngf * 16, dim)
        ])

        # Learnable scale fusion
        self.scale_weights = nn.Parameter(
            torch.ones(3)
        )

    def forward(self, x):

        feats = self.backbone(x)

        content_embs = []

        for f, head in zip(
            feats,
            self.content_heads
        ):
            content_embs.append(head(f))

        content_embs = torch.stack(
            content_embs,
            dim=1
        )                       # [B,3,dim]

        weights = torch.softmax(
            self.scale_weights,
            dim=0
        )                       # [3]

        content = (
            content_embs *
            weights.view(1,3,1)
        ).sum(dim=1)

        content = F.normalize(
            content,
            dim=1
        )

        return content