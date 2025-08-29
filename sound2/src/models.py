import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d((2,1))
        self.drop = nn.Dropout(p)
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = self.drop(x)
        return x

class MelCNN(nn.Module):
    def __init__(self, n_mels=128, n_classes=10, width=64, dropout=0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, width, dropout),
            ConvBlock(width, width*2, dropout),
            ConvBlock(width*2, width*2, dropout),
        )
        self.head = nn.Conv2d(width*2, n_classes, kernel_size=1)
        # final pooling over mel axis -> frame-wise logits over time
    def forward(self, x):  # x: [B,1,n_mels,T]
        h = self.backbone(x)            # [B, C, n_mels/8, T]
        h = h.mean(dim=2, keepdim=True) # pool mel -> [B, C, 1, T]
        logits = self.head(h).squeeze(2).transpose(1,2)  # [B, T, C]
        return logits  # frame-wise logits
