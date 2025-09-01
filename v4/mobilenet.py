# v4/mobilenet.py
import torch.nn as nn
import torchvision.models as models

class MobileNetBird(nn.Module):
    def __init__(self, n_classes: int, pretrained: bool = True):
        super().__init__()
        # Load pretrained MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)

        # Change first conv layer to accept 1-channel (spectrogram) instead of 3-channel RGB
        self.mobilenet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        # Replace the classification head
        in_feats = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_feats, n_classes)

    def forward(self, x):
        return self.mobilenet(x)