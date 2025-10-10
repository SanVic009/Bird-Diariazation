import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt
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

# ------------------------
# Feature extraction (same)
# ------------------------
def extract_features(y, sr, win, hop, n_mfcc):
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

# ------------------------
# Bird diarization (single bird)
# ------------------------
def bird_diarization_one_bird_aug(audio_path, sr=32000, win=2.0, hop=1.0, n_mfcc=13,
                                  smooth_kernel=5, augment=True):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    if augment:
        y = augment_audio(y, sr)

    X, times = extract_features(y, sr, win, hop, n_mfcc)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=1, covariance_type="diag", random_state=0)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)

    if smooth_kernel > 1:
        labels = medfilt(labels, kernel_size=smooth_kernel)

    plt.figure(figsize=(12,3))
    for t, label in zip(times, labels):
        plt.plot([t, t+win], [label, label], linewidth=8)
    plt.yticks([0], ["Bird 0"])
    plt.xlabel("Time (s)")
    plt.ylabel("Cluster")
    plt.title("Bird Diarization (Single Bird + Augmentation)")
    plt.show()

    return times, labels

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    audio_file = "birdclef-2024/train_audio/asbfly/XC49755.ogg"
    times, labels = bird_diarization_one_bird_aug(
        audio_file,
        augment=True
    )
    print(f"[INFO] Diarization complete: {len(labels)} segments detected")
