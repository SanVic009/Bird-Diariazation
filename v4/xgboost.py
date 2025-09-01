# xgboost_audio.py
import os
import numpy as np
import librosa
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import time
import logging



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
        
        # Concatenate all features into a flat vector
        features = np.hstack([mfcc_mean, mfcc_var, centroid_mean, bandwidth_mean, zcr_mean])
        return features
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        logging.error(f"[ERROR] Failed to process {file_path}: {e}")
        return None

# -------------------------
# Dataset Prep
# -------------------------
def build_dataset(audio_dir, labels_to_idx, max_files_per_class=None, cache_dir="."):
    # Define cache file paths
    if max_files_per_class is not None:
        cache_file_X = os.path.join(cache_dir, f"X_{max_files_per_class}.npy")
        cache_file_y = os.path.join(cache_dir, f"y_{max_files_per_class}.npy")
    else:
        cache_file_X = os.path.join(cache_dir, "X_full.npy")
        cache_file_y = os.path.join(cache_dir, "y_full.npy")

    # Check if cached files exist
    if os.path.exists(cache_file_X) and os.path.exists(cache_file_y):
        print(f"Loading cached data from {cache_file_X} and {cache_file_y}...")
        logging.info(f"Loading cached data from {cache_file_X} and {cache_file_y}...")
        X = np.load(cache_file_X)
        y = np.load(cache_file_y)
        return X, y

    # If not cached, process the data
    print("No cached data found. Starting preprocessing...")
    logging.info("No cached data found. Starting preprocessing...")
    X, y = [], []
    species_dirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
    num_species = len(species_dirs)
    print(f"Found {num_species} species directories.")
    logging.info(f"Found {num_species} species directories.")

    for i, species_dir in enumerate(species_dirs):
        species_path = os.path.join(audio_dir, species_dir)
        label = labels_to_idx.get(species_dir);
        if label is not None:
            print(f"Processing species {i+1}/{num_species}: {species_dir}")
            logging.info(f"Processing species {i+1}/{num_species}: {species_dir}")
            fnames = os.listdir(species_path)
            if max_files_per_class is not None:
                fnames = fnames[:max_files_per_class]
            num_files = len(fnames)
            for j, fname in enumerate(fnames):
                fpath = os.path.join(species_path, fname)
                feats = extract_features(fpath)
                if feats is not None:
                    X.append(feats)
                    y.append(label)
                if (j + 1) % 50 == 0:
                    print(f"  Processed {j+1}/{num_files} files...")
                    logging.info(f"  Processed {j+1}/{num_files} files...")
    
    print("Dataset building complete.")
    logging.info("Dataset building complete.")
    
    # Save the processed data to cache
    print(f"Saving preprocessed data to {cache_file_X} and {cache_file_y}...")
    logging.info(f"Saving preprocessed data to {cache_file_X} and {cache_file_y}...")
    np.save(cache_file_X, np.array(X))
    np.save(cache_file_y, np.array(y))

    return np.array(X), np.array(y)

# -------------------------
# Training Script
# -------------------------
if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load labels
    print("Loading labels...")
    logging.info("Loading labels...")
    labels_path = os.path.join(script_dir, "../labels.json")
    with open(labels_path) as f:
        labels_data = json.load(f)
    labels_to_idx = labels_data["label_to_idx"]

    # Set audio directory
    audio_dir = os.path.join(script_dir, "../birdclef-2024/train_audio")

    # Build dataset
    print("Building dataset...")
    logging.info("Building dataset...")
    cache_dir = os.path.dirname(os.path.abspath(__file__))
    X, y = build_dataset(audio_dir, labels_to_idx, max_files_per_class=250, cache_dir=cache_dir)
    print(f"Feature matrix shape: {X.shape}")
    logging.info(f"Feature matrix shape: {X.shape}")

    # Train/test split
    print("Splitting data into train and test sets...")
    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(set(y)),
        eval_metric="mlogloss",
        n_jobs=-1,
        tree_method='gpu_hist'  # Use GPU for training
    )

    # Train
    print("Training XGBoost model...")
    logging.info("Training XGBoost model...")
    model.fit(X_train, y_train)
    print("Training complete.")
    logging.info("Training complete.")

    # Evaluate
    print("Evaluating model...")
    logging.info("Evaluating model...")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
    logging.info(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    logging.info(classification_report(y_test, preds))

    # Generate and save confusion matrix
    print("Generating and saving confusion matrix...")
    logging.info("Generating and saving confusion matrix...")
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels_data["labels"], yticklabels=labels_data["labels"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    cm_path = os.path.join(script_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    logging.info(f"Confusion matrix saved to {cm_path}")
    plt.close()

    # Generate and save feature importance plot
    print("Generating and saving feature importance plot...")
    logging.info("Generating and saving feature importance plot...")
    feature_names = [f"mfcc_mean_{i}" for i in range(20)] + [f"mfcc_var_{i}" for i in range(20)] + ["centroid_mean", "bandwidth_mean", "zcr_mean"]
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title("XGBoost Feature Importance")
    fi_path = os.path.join(script_dir, f"feature_importance_{timestamp}.png")
    plt.savefig(fi_path)
    print(f"Feature importance plot saved to {fi_path}")
    logging.info(f"Feature importance plot saved to {fi_path}")
    plt.close()