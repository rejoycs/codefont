import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

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
class MultiScaleBackbone(nn.Module):
    def __init__(self, in_channels=3, ngf=64):
        super().__init__()

        self.init = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 3, 1, 1),
            nn.GroupNorm(8, ngf),
            nn.ReLU(inplace=True)
        )

        self.ml1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, 1, 1),
            nn.GroupNorm(16, ngf * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(ngf * 2)
        )

        self.ml2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 5, 1, 2),
            nn.GroupNorm(16, ngf * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(ngf * 2)
        )

        self.ml3 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 7, 1, 3),
            nn.GroupNorm(16, ngf * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(ngf * 2)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(ngf * 2 * 3, ngf * 4, 3, 1, 1),
            nn.GroupNorm(16, ngf * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(ngf * 4)
        )

    def forward(self, x):
        x = self.init(x)
        x1 = self.ml1(x)
        x2 = self.ml2(x)
        x3 = self.ml3(x)
        y = torch.cat([x1, x2, x3], dim=1)
        y = self.fusion(y)

        return y
        

class StyleHead(nn.Module):
    def __init__(
        self,
        in_channels,
        dim=256,
        num_queries=8,
        num_heads=8
    ):
        super().__init__()

        self.num_queries = num_queries
        self.proj = nn.Conv2d(in_channels, dim, 1)

        self.queries = nn.Parameter(
            torch.randn(num_queries, dim) * 0.02
        )

        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, f):

        B,C,H,W = f.shape
        x = self.proj(f)                     
        x = x.flatten(2).transpose(1,2)      
        q = self.queries.unsqueeze(0).expand(B, -1, -1)                                    

        style_tokens, attn_map = self.attn(q,x,x,need_weights=True)

        style_tokens = self.norm(style_tokens)
        style_tokens = self.mlp(style_tokens)
        style = style_tokens.mean(dim=1)
        style = F.normalize(style, dim=1)

        return style, style_tokens, attn_map


# ======================================================
# Encoder
# ======================================================
class StyleEncoder(nn.Module):

    def __init__(
        self,
        in_channels=3,
        ngf=64,
        dim=256,
        num_queries=8
    ):
        super().__init__()

        self.backbone = MultiScaleBackbone(
            in_channels=in_channels,
            ngf=ngf
        )

        self.style_head = StyleHead(
            in_channels=ngf * 4,
            dim=dim,
            num_queries=num_queries
        )

    def forward(self, x, only_style=True):
        feat = self.backbone(x)

        style,style_tokens,attn_map = self.style_head(feat)

        if not only_style:
            return {
                "style": style,                 # [B,256]
                "tokens": style_tokens,         # [B,Q,256]
                "attention": attn_map           # [B,Q,H*W]
            }
        else: 
            return style