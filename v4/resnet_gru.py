# resnet_gru_audio.py
import torch.nn as nn
import torchvision.models as models

class ResNetGRUBird(nn.Module):
    def __init__(self, n_classes: int, hidden_size: int = 128, num_layers: int = 2, pretrained: bool = True):
        super().__init__()
        # Load pretrained ResNet18
        base_resnet = models.resnet18(pretrained=pretrained)
        
        # Adapt first conv for 1-channel input
        base_resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Remove the final classifier, keep everything up to avgpool
        self.feature_extractor = nn.Sequential(*list(base_resnet.children())[:-2])  
        
        # GRU expects features as (B, T, F). We'll flatten spatial dims later.
        self.hidden_size = hidden_size
        self.gru = nn.GRU(
            input_size=512, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True
        )
        
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        # Extract CNN features: (B, 512, H, T)
        feats = self.feature_extractor(x)
        
        # Collapse frequency dimension → (B, 512, T)
        b, c, f, t = feats.size()
        feats = feats.mean(2)  # average over freq
        
        # Rearrange for GRU → (B, T, 512)
        feats = feats.permute(0, 2, 1)
        
        # GRU
        out, _ = self.gru(feats)
        
        # Take last time step
        out = self.fc(out[:, -1, :])
        return out
