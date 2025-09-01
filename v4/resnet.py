# resnet_audio.py
import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBird(nn.Module):
    def __init__(self, n_classes: int, pretrained: bool = True):
        super().__init__()
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Change first conv layer to accept 1-channel (spectrogram) instead of 3-channel RGB
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Replace the classification head
        in_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_feats, n_classes)

    def forward(self, x):
        return self.resnet(x)
