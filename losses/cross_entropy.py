import torch.nn as nn

# --------------------------------------------------
# Cross Entropy Loss
# --------------------------------------------------
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x, y):
        return self.loss(x, y)