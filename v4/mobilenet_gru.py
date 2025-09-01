# v4/mobilenet_gru.py
import torch.nn as nn
import torchvision.models as models

class MobileNetGRUBird(nn.Module):
    def __init__(self, n_classes: int, hidden_size: int = 128, num_layers: int = 2, pretrained: bool = True):
        super().__init__()
        # Load pretrained MobileNetV2
        base_mobilenet = models.mobilenet_v2(pretrained=pretrained)

        # Adapt first conv for 1-channel input
        base_mobilenet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        # Remove the final classifier
        self.feature_extractor = base_mobilenet.features

        # GRU expects features as (B, T, F). We'll flatten spatial dims later.
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=1280, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        # Extract CNN features: (B, 1280, H, T)
        feats = self.feature_extractor(x)

        # Collapse frequency dimension → (B, 1280, T)
        b, c, f, t = feats.size()
        feats = feats.mean(2)  # average over freq

        # Rearrange for GRU → (B, T, 1280)
        feats = feats.permute(0, 2, 1)

        # GRU
        out, _ = self.gru(feats)

        # Take last time step
        out = self.fc(out[:, -1, :])
        return out