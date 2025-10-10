import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt
import os
import random

# ------------------------
# Simple augmentations
# ------------------------
def augment_audio(y, sr):
    # 1. Add Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.005, y.shape)
        y = y + noise

    # 2. Time-stretch (90%-110%)
    if random.random() < 0.5:
        rate = random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y, rate=rate)

    # 3. Pitch shift (-2 to +2 semitones)
    if random.random() < 0.5:
        n_steps = random.uniform(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    # 4. Volume scaling (0.8-1.2)
    if random.random() < 0.5:
        gain = random.uniform(0.8, 1.2)
        y = y * gain

    return y

def extract_features(y, sr, win, hop, n_mfcc):
    """Extract mean MFCC features over sliding windows."""
    win_len = int(sr * win)
    hop_len = int(sr * hop)
    segments, times = [], []

    for start in range(0, len(y) - win_len, hop_len):
        seg = y[start:start+win_len]
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)
        segments.append(mfcc_mean)
        times.append(start / sr)

    return np.array(segments), np.array(times)

def select_best_gmm(X_scaled, max_clusters=6):
    """Fit GMMs with different n_components and select best by BIC."""
    best_gmm, lowest_bic, bic_scores = None, np.inf, []
    for n in range(1, max_clusters+1):
        gmm = GaussianMixture(n_components=n, covariance_type="diag", random_state=0)
        gmm.fit(X_scaled)
        bic = gmm.bic(X_scaled)
        bic_scores.append(bic)
        if bic < lowest_bic:
            best_gmm, lowest_bic = gmm, bic
    return best_gmm, bic_scores

def bird_diarization(audio_path, sr=32000, win=2.0, hop=1.0, n_mfcc=13,
                     max_clusters=6, smooth_kernel=5, augment=False):
    """
    End-to-end bird diarization with MFCC + GMM + BIC-based cluster selection.

    Parameters
    ----------
    audio_path : str
        Path to audio file
    sr : int
        Sampling rate for loading
    win : float
        Window length in seconds
    hop : float
        Hop size in seconds
    n_mfcc : int
        Number of MFCC features
    max_clusters : int
        Maximum number of clusters to consider
    smooth_kernel : int
        Median filter kernel size for label smoothing
    augment : bool
        If True, apply audio augmentation
    """
    # Load audio
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    # Augmentation
    if augment:
        y = augment_audio(y, sr)

    # Feature extraction
    X, times = extract_features(y, sr, win, hop, n_mfcc)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model selection
    best_gmm, bic_scores = select_best_gmm(X_scaled, max_clusters=max_clusters)
    labels = best_gmm.predict(X_scaled)
    n_clusters = best_gmm.n_components

    # Temporal smoothing
    if smooth_kernel > 1:
        labels = medfilt(labels, kernel_size=smooth_kernel)

    # --- Plot BIC curve ---
    plt.figure(figsize=(6,3))
    plt.plot(range(1, max_clusters+1), bic_scores, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("BIC")
    plt.title("Model selection via BIC")
    plt.show()

    # --- Plot diarization timeline ---
    plt.figure(figsize=(12,3))
    for t, label in zip(times, labels):
        plt.plot([t, t+win], [label, label], linewidth=8)
    plt.yticks(range(n_clusters), [f"Bird {i}" for i in range(n_clusters)])
    plt.xlabel("Time (s)")
    plt.ylabel("Cluster")
    plt.title(f"Bird Diarization (MFCC + GMM, {n_clusters} clusters)")
    plt.show()

    return times, labels, n_clusters

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    audio_file = "/home/sanvict/Downloads/WhatsApp Audio 2025-09-15 at 02.28.57.mp4"   # replace with your bird audio
    
    print("--- Running standard diarization ---")
    times, labels, n_clusters = bird_diarization(
        audio_file,
        max_clusters=6
    )
    print(f"Detected {n_clusters} clusters")

    print("\n--- Running diarization with augmentation ---")
    times_aug, labels_aug, n_clusters_aug = bird_diarization(
        audio_file,
        max_clusters=6,
        augment=True
    )
    print(f"Detected {n_clusters_aug} clusters with augmentation")