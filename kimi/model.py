# model.py
import torch.nn as nn
import torch

class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=128, n_mels=64):
        super().__init__()
        self.n_mels = n_mels  # Store as instance variable
        self.embed_dim = embed_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x):          # x: (B, 1, T)
        spec = torch.stft(x.squeeze(1), n_fft=1024, hop_length=512, return_complex=True).abs()
        spec = spec[:, :self.n_mels, :] # crude mel-like slice
        spec = spec.unsqueeze(1)   # (B,1,F,T)
        h = self.conv(spec).flatten(1)
        return self.fc(h)