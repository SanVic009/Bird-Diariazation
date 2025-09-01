# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# ------------------------------
# CNN + LSTM model
# ------------------------------
class BirdCNNLSTM(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.5, n_mels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        # Calculate LSTM input size dynamically
        # After 3 pooling layers: n_mels / 2^3 = n_mels / 8
        pooled_freq_bins = n_mels // 8
        lstm_input_size = 128 * pooled_freq_bins  # channels * freq_bins after CNN pooling
        
        self.lstm_hidden = 128
        self.lstm = nn.GRU(
            input_size=lstm_input_size, hidden_size=self.lstm_hidden,
            num_layers=2, batch_first=True, bidirectional=True
        )

        self.fc = nn.Linear(self.lstm_hidden * 2, n_classes)

    def forward(self, x):
        # x shape: (B, 1, n_mels, time)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # shape (B, C, F, T)
        x = self.dropout(x)

        # collapse frequency dimension → (B, T, C*F)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        lstm_out, _ = self.lstm(x)  # (B, T, 2*hidden)
        out = self.fc(lstm_out[:, -1, :])  # last time step
        return out


# ------------------------------
# CNN feature extractor (for RF)
# ------------------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, out_dim=256, n_mels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.out_dim = out_dim
        
        # Calculate the feature size after pooling
        # After 3 pooling layers: n_mels / 2^3 = n_mels / 8
        pooled_freq_bins = n_mels // 8
        self.fc = nn.Linear(128 * pooled_freq_bins, out_dim)

    def forward(self, x):
        # x: (B, 1, n_mels, time)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        b, c, f, t = x.shape
        x = x.mean(-1)  # global pool over time
        x = x.view(b, -1)
        x = self.fc(x)
        return x

# ------------------------------
# CNN + Random Forest wrapper
# ------------------------------
class CNNRandomForest:
    def __init__(self, feature_dim=256, n_estimators=200, n_mels=128):
        self.feature_extractor = CNNFeatureExtractor(out_dim=feature_dim, n_mels=n_mels)
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )
        self.scaler = StandardScaler()

    def fit(self, dataloader, device="cpu"):
        self.feature_extractor.to(device)
        self.feature_extractor.eval()
        features, labels = [], []

        with torch.no_grad():
            for xb, yb in dataloader:
                xb = xb.to(device)
                feats = self.feature_extractor(xb).cpu().numpy()
                features.append(feats)
                labels.append(yb.numpy())

        features = np.vstack(features)
        labels = np.hstack(labels)
        
        # Fit and transform features with StandardScaler
        features = self.scaler.fit_transform(features)
        
        self.rf.fit(features, labels)

    def predict(self, x, device="cpu"):
        self.feature_extractor.to(device)
        self.feature_extractor.eval()
        with torch.no_grad():
            feats = self.feature_extractor(x.to(device)).cpu().numpy()
        
        # Transform features with the fitted StandardScaler
        feats = self.scaler.transform(feats)
        
        return self.rf.predict(feats)

    def save(self, path: str):
        joblib.dump(self.rf, path + "_rf.pkl")
        torch.save(self.feature_extractor.state_dict(), path + "_cnn.pth")
        joblib.dump(self.scaler, path + "_scaler.pkl")

    def load(self, path: str, device="cpu"):
        self.rf = joblib.load(path + "_rf.pkl")
        self.feature_extractor.load_state_dict(torch.load(path + "_cnn.pth", map_location=device))
        self.scaler = joblib.load(path + "_scaler.pkl")
