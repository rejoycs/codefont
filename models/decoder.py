import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# Stable Upsample
# ======================================================
def upsample(cin, cout):
    return nn.Sequential(
        nn.Upsample(
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        ),
        nn.Conv2d(cin, cout, 3, 1, 1)
    )


# ======================================================
# AdaIN
# ======================================================
class AdaIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, gamma, beta):

        B,C,H,W = x.shape

        mu = x.view(B,C,-1).mean(2).view(B,C,1,1)
        var = x.view(B,C,-1).var(
            2,
            unbiased=False
        ).view(B,C,1,1)

        x = (x - mu) / torch.sqrt(var + self.eps)

        gamma = gamma.view(B,C,1,1)
        beta  = beta.view(B,C,1,1)

        return gamma * x + beta


# ======================================================
# Style Gate
# ======================================================
class StyleGate(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()

        self.fc = nn.Linear(
            style_dim,
            channels
        )

    def forward(self, x, style):

        g = torch.sigmoid(
            self.fc(style)
        ).unsqueeze(-1).unsqueeze(-1)

        return x * g


# ======================================================
# Residual Block
# ======================================================
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.GroupNorm(16, ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.GroupNorm(16, ch)
        )

    def forward(self, x):
        return F.relu(
            x + self.block(x),
            inplace=True
        )


# ======================================================
# Decoder
# ======================================================
class StyleContentDecoder(nn.Module):

    def __init__(
        self,
        style_dim=256,
        content_dim=256,
        out_channels=3,
        ngf=64,
        spatial=4
    ):
        super().__init__()

        self.spatial = spatial
        self.init_ch = ngf * 16

        # ----------------------------------
        # Content seed
        # ----------------------------------
        self.content_fc = nn.Sequential(
            nn.Linear(
                content_dim,
                self.init_ch * spatial * spatial
            ),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------
        # AdaIN
        # ----------------------------------
        self.adain1 = AdaIN(self.init_ch)
        self.adain2 = AdaIN(ngf * 8)
        self.adain3 = AdaIN(ngf * 4)
        self.adain4 = AdaIN(ngf * 2)

        # ----------------------------------
        # Style affine
        # ----------------------------------
        self.style_fc1 = nn.Linear(
            style_dim,
            self.init_ch * 2
        )

        self.style_fc2 = nn.Linear(
            style_dim,
            ngf * 8 * 2
        )

        self.style_fc3 = nn.Linear(
            style_dim,
            ngf * 4 * 2
        )

        self.style_fc4 = nn.Linear(
            style_dim,
            ngf * 2 * 2
        )

        # ----------------------------------
        # Style gates
        # ----------------------------------
        self.gate1 = StyleGate(
            style_dim,
            ngf * 8
        )

        self.gate2 = StyleGate(
            style_dim,
            ngf * 4
        )

        self.gate3 = StyleGate(
            style_dim,
            ngf * 2
        )

        # ----------------------------------
        # Upsampling path
        # ----------------------------------
        self.up1 = nn.Sequential(
            upsample(self.init_ch, ngf * 8),
            ResBlock(ngf * 8)
        )

        self.up2 = nn.Sequential(
            upsample(ngf * 8, ngf * 4),
            ResBlock(ngf * 4)
        )

        self.up3 = nn.Sequential(
            upsample(ngf * 4, ngf * 2),
            ResBlock(ngf * 2)
        )

        self.up4 = nn.Sequential(
            upsample(ngf * 2, ngf),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(
                ngf,
                out_channels,
                3,
                1,
                1
            ),
            nn.Tanh()
        )

    def forward(
        self,
        style_emb,
        content_emb
    ):

        B = content_emb.size(0)

        # ----------------------------------
        # Content seed
        # ----------------------------------
        y = self.content_fc(content_emb)

        y = y.view(
            B,
            self.init_ch,
            self.spatial,
            self.spatial
        )

        # ----------------------------------
        # Block 1
        # ----------------------------------
        gamma, beta = self.style_fc1(
            style_emb
        ).chunk(2, dim=1)

        y = self.adain1(
            y,
            gamma,
            beta
        )

        y = self.up1(y)
        y = self.gate1(y, style_emb)

        # ----------------------------------
        # Block 2
        # ----------------------------------
        gamma, beta = self.style_fc2(
            style_emb
        ).chunk(2, dim=1)

        y = self.adain2(
            y,
            gamma,
            beta
        )

        y = self.up2(y)
        y = self.gate2(y, style_emb)

        # ----------------------------------
        # Block 3
        # ----------------------------------
        gamma, beta = self.style_fc3(
            style_emb
        ).chunk(2, dim=1)

        y = self.adain3(
            y,
            gamma,
            beta
        )

        y = self.up3(y)
        y = self.gate3(y, style_emb)

        # ----------------------------------
        # Block 4
        # ----------------------------------
        gamma, beta = self.style_fc4(
            style_emb
        ).chunk(2, dim=1)

        y = self.adain4(
            y,
            gamma,
            beta
        )

        y = self.up4(y)

        return self.out(y)