# efficientnet.py
import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetBird(nn.Module):
    def __init__(self, n_classes: int, pretrained: bool = True, multi_label: bool = False, variant: str = "b0"):
        """
        EfficientNet wrapper for bird classification.
        
        Args:
            n_classes (int): number of output classes
            pretrained (bool): load pretrained weights on ImageNet
            multi_label (bool): whether task is multi-label
            variant (str): EfficientNet version (b0–b7, v2 variants if available in torchvision)
        """
        super().__init__()
        self.multi_label = multi_label
        
        # Load EfficientNet backbone
        if variant == "b0":
            self.effnet = models.efficientnet_b0(pretrained=pretrained)
        elif variant == "b1":
            self.effnet = models.efficientnet_b1(pretrained=pretrained)
        elif variant == "b2":
            self.effnet = models.efficientnet_b2(pretrained=pretrained)
        elif variant == "b3":
            self.effnet = models.efficientnet_b3(pretrained=pretrained)
        elif variant == "b4":
            self.effnet = models.efficientnet_b4(pretrained=pretrained)
        elif variant == "b5":
            self.effnet = models.efficientnet_b5(pretrained=pretrained)
        elif variant == "b6":
            self.effnet = models.efficientnet_b6(pretrained=pretrained)
        elif variant == "b7":
            self.effnet = models.efficientnet_b7(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported EfficientNet variant: {variant}")

        # Modify first conv to accept 1-channel input (spectrograms)
        self.effnet.features[0][0] = nn.Conv2d(
            1,
            self.effnet.features[0][0].out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

        # Replace classification head
        in_feats = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(in_feats, n_classes)

    def forward(self, x):
        logits = self.effnet(x)
        if self.multi_label:
            return torch.sigmoid(logits)
        else:
            return logits
