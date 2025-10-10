#!/usr/bin/env python3
"""
improved_models.py - Enhanced Bird Diarization Models
Features:
- ResNet backbone with attention mechanisms
- Temporal modeling for sequential patterns
- Multi-scale feature extraction
- Improved embedding projection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Enhanced residual block with SE attention"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(residual)
        return F.relu(out)

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal modeling"""
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales"""
    def __init__(self, in_channels):
        super().__init__()
        # Different kernel sizes for multi-scale analysis
        self.conv1x1 = nn.Conv2d(in_channels, in_channels//4, 1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels//4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels//4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, in_channels//4, 7, padding=3)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x7 = self.conv7x7(x)
        out = torch.cat([x1, x3, x5, x7], dim=1)
        return F.relu(self.bn(out))

class ImprovedDiarizationEncoder(nn.Module):
    """Enhanced diarization encoder with all improvements"""
    
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Enhanced CNN backbone
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet-style layers with SE attention
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout=dropout)
        
        # Multi-scale feature extraction
        self.multiscale = MultiScaleFeatureExtractor(512)
        
        # Global attention pooling
        self.attention_pool = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        
        # Adaptive pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal modeling with transformer
        self.pos_encoding = PositionalEncoding(512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=num_heads, dim_feedforward=1024, 
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Enhanced embedding projection
        self.projector = nn.Sequential(
            nn.Linear(512, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # ResNet layers
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Multi-scale features
        x4 = self.multiscale(x4)
        
        # Attention-based pooling
        att_weights = self.attention_pool(x4)
        x4_att = x4 * att_weights
        
        # Global pooling
        features = self.global_pool(x4_att)
        features = features.view(batch_size, -1)
        
        # Optional temporal modeling (for sequences)
        if features.dim() == 3:  # If we have temporal dimension
            features = self.pos_encoding(features.transpose(0, 1)).transpose(0, 1)
            features = self.transformer(features)
            features = features.mean(dim=1)  # Global temporal pooling
        
        # Generate embeddings
        embeddings = self.projector(features)
        
        # L2 normalize for better clustering
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        if return_features:
            return embeddings, features
        return embeddings

class TemporalDiarizationEncoder(nn.Module):
    """Temporal-aware diarization encoder for sequence processing"""
    
    def __init__(self, embed_dim=256, sequence_length=10):
        super().__init__()
        self.sequence_length = sequence_length
        self.frame_encoder = ImprovedDiarizationEncoder(embed_dim=embed_dim//2)
        
        # Temporal modeling
        self.temporal_conv = nn.Conv1d(embed_dim//2, embed_dim//2, kernel_size=3, padding=1)
        self.temporal_norm = nn.BatchNorm1d(embed_dim//2)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            embed_dim//2, embed_dim//2, 
            num_layers=2, dropout=0.1, bidirectional=True, batch_first=True
        )
        
        # Final projection
        self.final_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x shape: [batch, sequence, channels, height, width]
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Process each frame
        frame_embeddings = []
        for i in range(seq_len):
            frame_emb = self.frame_encoder(x[:, i])
            frame_embeddings.append(frame_emb)
        
        # Stack embeddings
        sequence_embeddings = torch.stack(frame_embeddings, dim=1)  # [batch, seq, embed]
        
        # Temporal convolution
        temp_features = sequence_embeddings.transpose(1, 2)  # [batch, embed, seq]
        temp_features = F.relu(self.temporal_norm(self.temporal_conv(temp_features)))
        temp_features = temp_features.transpose(1, 2)  # [batch, seq, embed]
        
        # LSTM processing
        lstm_out, _ = self.lstm(temp_features)
        
        # Global temporal pooling
        final_embedding = lstm_out.mean(dim=1)  # [batch, embed*2]
        final_embedding = self.final_proj(final_embedding)
        
        # L2 normalize
        final_embedding = F.normalize(final_embedding, p=2, dim=1)
        
        return final_embedding

# For backward compatibility
class LegacyDiarizationEncoder(nn.Module):
    """Wrapper to maintain compatibility with existing code"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.encoder = ImprovedDiarizationEncoder(embed_dim=embed_dim)
    
    def forward(self, x):
        return self.encoder(x)

if __name__ == "__main__":
    # Test the models
    print("Testing Improved Diarization Models...")
    
    # Test basic encoder
    model = ImprovedDiarizationEncoder(embed_dim=256)
    x = torch.randn(4, 1, 128, 501)  # [batch, channels, mel_bins, time]
    
    print(f"Input shape: {x.shape}")
    
    embeddings = model(x)
    print(f"Output embeddings shape: {embeddings.shape}")
    
    # Test temporal encoder
    temp_model = TemporalDiarizationEncoder(embed_dim=256, sequence_length=5)
    x_seq = torch.randn(2, 5, 1, 128, 501)  # [batch, seq, channels, mel_bins, time]
    
    temp_embeddings = temp_model(x_seq)
    print(f"Temporal embeddings shape: {temp_embeddings.shape}")
    
    print("✅ All models working correctly!")