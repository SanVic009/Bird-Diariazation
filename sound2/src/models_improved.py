import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, p=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.drop = nn.Dropout(p)
        
        # Skip connection
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )
            
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = F.relu(out)
        out = self.drop(out)
        return out

class ImprovedMelCNN(nn.Module):
    def __init__(self, n_mels=128, n_classes=10, width=64, dropout=0.2):
        super().__init__()
        
        # Initial conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks with increasing channels
        self.layer1 = self._make_layer(width, width, blocks=2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(width, width*2, blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(width*2, width*4, blocks=2, stride=2, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(width*4, width, kernel_size=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.Conv2d(width*4, width*2, kernel_size=1),
            nn.BatchNorm2d(width*2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(width*2, n_classes, kernel_size=1)
        )
        
    def _make_layer(self, in_ch, out_ch, blocks, stride, dropout):
        layers = []
        layers.append(ResBlock(in_ch, out_ch, stride=stride, p=dropout))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_ch, out_ch, stride=1, p=dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):  # x: [B,1,n_mels,T]
        # Feature extraction
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)  # [B, C, H, T]
        
        # Attention weighting
        att = self.attention(x)  # [B, 1, H, T]
        x = x * att  # Apply attention
        
        # Global average pooling over frequency
        x = x.mean(dim=2, keepdim=True)  # [B, C, 1, T]
        
        # Classification
        logits = self.head(x).squeeze(2).transpose(1,2)  # [B, T, n_classes]
        return logits

class EfficientNetBirds(nn.Module):
    def __init__(self, n_classes, model_name='efficientnet_b0'):
        super().__init__()
        self.backbone = time.create_model(model_name, pretrained=True, in_chans=1)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        self.head = nn.Sequential(
            nn.Linear(n_features, n_features//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(n_features//2, n_classes)
        )
        
    def forward(self, x):
        # Assuming x is [B, 1, n_mels, T]
        features = self.backbone(x)  # [B, n_features]
        logits = self.head(features)  # [B, n_classes]
        return logits.unsqueeze(1)  # [B, 1, n_classes] to match expected shape
