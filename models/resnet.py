import torch.nn as nn
from torchvision import models

class ResNetRegression(nn.Module):
    def __init__(self, num_outputs=1, resnet_version=18, weights=None):
        super().__init__()
        if resnet_version == 18:
            self.resnet = models.resnet18(weights=weights)
        elif resnet_version == 34:
            self.resnet = models.resnet34(weights=weights)
        elif resnet_version == 50:
            self.resnet = models.resnet50(weights=weights)
        else:
            raise ValueError("Unsupported ResNet version")
        # Replace last fully connected layer for regression
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_outputs)

    def forward(self, x):
        return self.resnet(x)
