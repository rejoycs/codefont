import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# Contrastive Loss
# --------------------------------------------------
class CircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=32, eps=1e-8):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.eps = eps

    def forward(self, anchor, positive, negative):
        sp = torch.sum(anchor * positive, dim=1)
        sn = torch.sum(anchor * negative, dim=1)

        ap = torch.clamp_min(1 + self.m - sp.detach(), 0.)
        an = torch.clamp_min(sn.detach() + self.m, 0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -self.gamma * ap * (sp - delta_p)
        logit_n =  self.gamma * an * (sn - delta_n)

        loss = F.softplus(logit_p + logit_n)
        return loss.mean()
