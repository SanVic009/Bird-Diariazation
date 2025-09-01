# svm_audio.py
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# -------------------------
# Feature Extraction
# -------------------------
def extract_features(file_path, sr=32000, n_mfcc=20):
    try:
        y, sr = librosa.load(file_path, sr=sr, mono=True)
        
        # 1. MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        
        # 2. Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        
        # 3. Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = np.mean(bandwidth)
        
        # 4. Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # Concatenate all features into flat vector
        features = np.hstack([mfcc_mean, mfcc_var, centroid_mean, bandwidth_mean, zcr_mean])
        return features
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        logging.error(f"[ERROR] Failed to process {file_path}: {e}")
        return None

# -------------------------
# Dataset Prep
# -------------------------
def build_dataset(audio_dir, labels_dict):
    X, y = [], []
    for fname, label in labels_dict.items():
        fpath = os.path.join(audio_dir, fname)
        feats = extract_features(fpath)
        if feats is not None:
            X.append(feats)
            y.append(label)
    return np.array(X), np.array(y)

# -------------------------
# Training Script
# -------------------------
if __name__ == "__main__":
    # Example mapping filenames to species labels
    audio_dir = "bird_audio"
    labels_dict = {
        "bird1.wav": 0,
        "bird2.wav": 1,
        "bird3.wav": 2,
        # add more
    }

    # Build dataset
    X, y = build_dataset(audio_dir, labels_dict)
    print(f"Feature matrix shape: {X.shape}")
    logging.info(f"Feature matrix shape: {X.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features (SVM needs scaling)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVM classifier
    clf = SVC(
        kernel="rbf",       # radial basis function kernel
        C=10,               # regularization
        gamma="scale",      # kernel coefficient
        probability=True    # enable probability outputs
    )

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    logging.info(classification_report(y_test, preds))
