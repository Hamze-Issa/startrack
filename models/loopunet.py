import torch.nn as nn
from segmentation_models_pytorch import Unet

class LoopUnet(nn.Module):
    def __init__(self, in_channels=2, num_classes=1, encoder='resnet50', weights=None):
        super().__init__()
        self.unet = Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
        )
        
    def forward(self, x):
        if isinstance(x, dict):
            return self.unet(x["image"])
        return self.unet(x)