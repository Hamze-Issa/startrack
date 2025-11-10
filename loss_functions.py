import torch
import torch.nn as nn
from torch import isnan
from segmentation_models_pytorch.losses import DiceLoss

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
    
class MaskedMSELoss(nn.Module):
    def __init__(self, ignore_nans=True):
        super().__init__()
        self.ignore_nans = ignore_nans
        self.base_loss = nn.MSELoss(reduction='none')
        
    def forward(self, inputs, targets, mask=None):
        loss = self.base_loss(inputs, targets.float())
        
        if self.ignore_nans:
            if mask is None:
                # Auto-detect NaNs in targets
                mask = ~isnan(targets)
            loss = loss * mask.float()
            
        return loss.mean()  # Only average valid pixels
    
class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_nans=True):
        """
        Focal loss variant of BCEWithLogitsLoss with masking and NaN filtering.
        Arguments:
            alpha (float): Class weighting factor to handle class imbalance.
            gamma (float): Focusing parameter to down-weight easy examples.
            ignore_nans (bool): Whether to mask out NaN target values.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_nans = ignore_nans
        self.base_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets, mask=None):
        # Standard BCE (per-pixel, unreduced)
        bce_loss = self.base_loss(inputs, targets.float())
        
        # Derive pt: predicted probability for the true class
        pt = torch.exp(-bce_loss)

        # Compute focal scaling
        focal_term = self.alpha * (1 - pt) ** self.gamma
        loss = focal_term * bce_loss

        # Mask invalid or NaN targets
        if self.ignore_nans:
            if mask is None:
                mask = ~torch.isnan(targets)
            loss = loss * mask.float()

        # Average over valid pixels only
        return loss.sum() / (mask.float().sum() + 1e-8)

class MaskedWeightedMSELoss(nn.Module):
    def __init__(self, ignore_nans=True):
        super().__init__()
        self.ignore_nans = ignore_nans
        self.base_loss = nn.MSELoss(reduction='none')
    
    def forward(self, inputs, targets, mask=None, weights=None, alpha=0.5):
        # Compute per-pixel squared error
        loss = self.base_loss(inputs, targets.float())
        
        # Handle NaN masking
        if self.ignore_nans:
            if mask is None:
                mask = ~isnan(targets)
            loss = loss * mask.float()
        
        if weights is None:
            weights = 1 + alpha * (targets / targets.max())

        # Apply weights (e.g. to emphasize rare/high-value pixels)
            # Ensure same shape
        if weights.shape != loss.shape:
            weights = torch.broadcast_to(weights, loss.shape)
        loss = loss * weights
        
        # Normalize by valid pixels
        valid_mask = mask.float() if mask is not None else torch.ones_like(loss)
        weighted_sum = (loss * valid_mask).sum()
        normalization = (valid_mask * (weights if weights is not None else 1)).sum()
        
        return loss.mean() #weighted_sum / normalization.clamp(min=1e-8)

LOSS_FUNCTIONS = {
    'bce': nn.BCEWithLogitsLoss,
    'dice': DiceLoss,
    # Add custom losses
    'masked_bce': MaskedBCELoss,
    'masked_mse': MaskedMSELoss,
    'masked_focal_loss': MaskedFocalLoss,
    'masked_weighted_mse': MaskedWeightedMSELoss,
}