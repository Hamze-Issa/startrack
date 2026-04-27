import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss


class MaskedLoss(nn.Module):
    """
    Universal loss wrapper. When a joint_valid mask is provided, inputs and
    targets are flattened to valid pixels only before the loss is computed.

    Handles two flattening strategies automatically based on target shape:
      - targets.dim() == inputs.dim()  →  binary / regression / one-hot:
            both flattened element-wise to [N] or [N*C]
      - targets.dim() <  inputs.dim()  →  multiclass integer indices [B,H,W]:
            inputs flattened spatially to [N, C], targets to [N]

    For structural losses that require the full spatial tensor (e.g. Dice),
    set apply_masking=False — the mask is accepted but ignored.
    """
    def __init__(self, loss_fn, apply_masking=True):
        super().__init__()
        self.loss_fn = loss_fn
        self.apply_masking = apply_masking

    def forward(self, inputs, targets, mask=None):
        if mask is None or not self.apply_masking:
            return self.loss_fn(inputs, targets)
        if targets.dim() < inputs.dim():
            # Multiclass: targets are integer indices [B, H, W],
            # inputs are per-class logits [B, C, H, W].
            # Flatten spatially while keeping the class dimension.
            spatial_mask = mask.squeeze(1).bool()               # [B, H, W]
            inputs_flat  = inputs.permute(0, 2, 3, 1)[spatial_mask]  # [N, C]
            targets_flat = targets[spatial_mask]                # [N]
        else:
            # Binary / regression / one-hot: inputs and targets share shape.
            # Flatten every element at valid spatial positions.
            valid        = mask.expand_as(inputs).bool()
            inputs_flat  = inputs[valid]                        # [N] or [N*C]
            targets_flat = targets[valid]                       # [N] or [N*C]
        return self.loss_fn(inputs_flat, targets_flat)


LOSS_FUNCTIONS = {
    # Per-pixel losses — masking via flattening to valid pixels only.
    # For z-score normalized inputs the fill value 0.0 equals the channel mean,
    # so invalid pixels look like average input to the model (minimal contamination).
    'bce':           lambda **kwargs: MaskedLoss(nn.BCEWithLogitsLoss(**kwargs)),
    'mse':           lambda **kwargs: MaskedLoss(nn.MSELoss(**kwargs)),
    'huber':         lambda **kwargs: MaskedLoss(nn.HuberLoss(**kwargs)),
    'cross_entropy': lambda **kwargs: MaskedLoss(nn.CrossEntropyLoss(**kwargs)),
    # Structural loss — requires full spatial tensor; masking is not applied.
    # Invalid pixels (filled with fill_value) will influence the Dice score.
    # Use only when invalid regions are small relative to the valid area.
    'dice':          lambda **kwargs: MaskedLoss(DiceLoss(**kwargs), apply_masking=False),
}
