import torch.nn as nn

# --------------------------------------------------
# L1 Loss
# --------------------------------------------------
class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, x, y):
        return self.loss(x, y)