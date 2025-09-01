# train_rf.py
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import joblib
from models import CNNRandomForest
import logging

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

def extract_features(dataloader, feature_extractor, device="cpu"):
    feature_extractor.to(device)
    feature_extractor.eval()
    feats, labels = [], []

    with torch.no_grad():
        for xb, yb in tqdm(dataloader, desc="Extract", leave=False):
            if xb is None:
                continue
            xb = xb.to(device)
            out = feature_extractor(xb).cpu().numpy()
            feats.append(out)
            labels.append(yb.numpy())

    feats = np.vstack(feats)
    labels = np.hstack(labels)
    return feats, labels

def train_rf(train_loader, val_loader, feature_dim=256, n_estimators=200, device="cpu", save_dir="checkpoints_rf", n_mels=128):
    os.makedirs(save_dir, exist_ok=True)

    # Log model type and training start
    logger.info(f"Starting CNN+RandomForest training with {n_estimators} estimators")
    logger.info(f"Training parameters: feature_dim={feature_dim}, device={device}, n_mels={n_mels}")

    # Initialize wrapper
    model = CNNRandomForest(feature_dim=feature_dim, n_estimators=n_estimators, n_mels=n_mels)
    feature_extractor = model.feature_extractor

    # Extract embeddings
    X_train, y_train = extract_features(train_loader, feature_extractor, device)
    X_val, y_val = extract_features(val_loader, feature_extractor, device)

    # Train RF
    model.rf.fit(X_train, y_train)

    # Validation
    preds = model.rf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")
    print(f"RF Val Acc: {acc:.4f} | F1: {f1:.4f}")
    logger.info(f"RF Val Acc: {acc:.4f} | F1: {f1:.4f}")

    # Generate timestamp for consistent naming
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save both parts with accuracy in filename
    joblib.dump(model.rf, os.path.join(save_dir, f"rf_{timestamp}_acc{acc:.4f}.pkl"))
    torch.save(feature_extractor.state_dict(), os.path.join(save_dir, f"cnn_feat_{timestamp}_acc{acc:.4f}.pth"))
    
    print(f"Saved Random Forest model with accuracy: {acc:.4f}")
    logger.info(f"Saved Random Forest model with accuracy: {acc:.4f}")

    return model
