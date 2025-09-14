# v4/mobilenet.py
import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetBird(nn.Module):
    def __init__(self, n_classes: int, pretrained: bool = True, multi_label: bool = False):
        super().__init__()
        self.multi_label = multi_label
        
        # Load pretrained MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)

        # Change first conv layer to accept 1-channel (spectrogram) instead of 3-channel RGB
        self.mobilenet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        # Replace the classification head
        # Keep the final Linear(in_feats, n_classes) the same
        in_feats = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(in_feats, n_classes)

    def forward(self, x):
        # Get raw logits from the model
        logits = self.mobilenet(x)
        
        if self.multi_label:
            # For multi-label: use sigmoid for independent per-class probabilities
            return torch.sigmoid(logits)
        else:
            # For multi-class: return raw logits (used with BCEWithLogitsLoss)
            return logits