import torch.nn as nn
from torch import isnan

class MaskedBCELoss(nn.Module):
    def __init__(self, ignore_nans=True):
        super().__init__()
        self.ignore_nans = ignore_nans
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets, mask=None):
        loss = self.base_loss(inputs, targets.float())
        
        if self.ignore_nans:
            if mask is None:
                # Auto-detect NaNs in targets
                mask = ~isnan(targets)
            loss = loss * mask.float()
            
        return loss.mean()  # Only average valid pixels