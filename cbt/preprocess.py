#!/usr/bin/env python3
"""
preprocess.py
Preprocess BirdCLEF-2024 audio files:
- Convert to mel-spectrograms
- Apply augmentations (SpecAugment, MixUp/CutMix, background mixing,
  random gain, pink-noise, random crops)
- Save processed tensors to disk for fast training
"""

import os, random
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, FrequencyMasking, TimeMasking

SR = 32000
N_MELS = 128
CACHE_DIR = Path("cache_mels")
CACHE_DIR.mkdir(exist_ok=True)

mel_fn = MelSpectrogram(
    sample_rate=SR,
    n_fft=1024,
    hop_length=320,
    win_length=1024,
    f_min=20,
    f_max=16000,
    n_mels=N_MELS,
)
db_fn = AmplitudeToDB()

# ---------------------------
# Augmentations
# ---------------------------
def random_crop(x, crop_sec=5.0, max_len_sec=10.0):
    crop_len = int(crop_sec * SR)
    max_len = int(max_len_sec * SR)
    if len(x) < crop_len:
        pad = crop_len - len(x)
        x = np.pad(x, (0, pad))
    else:
        start = random.randint(0, max(1, len(x) - crop_len))
        x = x[start : start + crop_len]
    return x

def add_pink_noise(x, noise_level=0.01):
    # Pink noise ~ 1/f spectrum
    wn = np.random.randn(len(x))
    fft = np.fft.rfft(wn)
    freqs = np.fft.rfftfreq(len(x))
    fft /= np.sqrt(freqs + 1e-6)
    pink = np.fft.irfft(fft, n=len(x))
    pink /= np.max(np.abs(pink))
    return x + noise_level * pink[:len(x)]

def random_gain(x, low=0.7, high=1.3):
    return x * np.random.uniform(low, high)

def background_mix(fg, bg, ratio=None):
    if ratio is None:
        ratio = np.random.uniform(0.1, 0.3)
    fg = fg[:len(bg)]
    return fg * (1 - ratio) + bg * ratio

def spec_augment(spec):
    freq_mask = FrequencyMasking(freq_mask_param=15)
    time_mask = TimeMasking(time_mask_param=25)
    return time_mask(freq_mask(spec))

def mixup(x1, x2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    return lam * x1 + (1 - lam) * x2

# ---------------------------
# Processing
# ---------------------------
def process_audio(file_path, bg_files=None):
    wav, sr = sf.read(file_path, dtype="float32")
    if sr != SR:
        wav = torchaudio.functional.resample(torch.tensor(wav), sr, SR).numpy()

    # Random crop from 10s
    wav = random_crop(wav, crop_sec=5.0, max_len_sec=10.0)

    # Augment
    if random.random() < 0.5:
        wav = add_pink_noise(wav)
    if random.random() < 0.5:
        wav = random_gain(wav)
    if bg_files and random.random() < 0.3:
        bg_file = random.choice(bg_files)
        bg, sr_bg = sf.read(bg_file, dtype="float32")
        if sr_bg != SR:
            bg = torchaudio.functional.resample(torch.tensor(bg), sr_bg, SR).numpy()
        bg = random_crop(bg, crop_sec=5.0)
        wav = background_mix(wav, bg)

    # To tensor
    wav_t = torch.tensor(wav, dtype=torch.float32)
    mel = mel_fn(wav_t)
    mel_db = db_fn(mel)

    # SpecAugment
    if random.random() < 0.5:
        mel_db = spec_augment(mel_db)

    return mel_db

def preprocess_dataset(audio_dir, bg_dir, out_meta="metadata.csv"):
    audio_dir = Path(audio_dir)
    bg_files = list(Path(bg_dir).glob("*.wav")) if bg_dir else []

    meta = []
    audio_files = list(audio_dir.rglob("*.wav")) + list(audio_dir.rglob("*.ogg"))
    num_files = len(audio_files)
    print(f"Found {num_files} audio files to process. Will skip already processed files.")
    for i, audio_file in enumerate(audio_files):
        label = audio_file.parent.name
        out_name = f"{label}_{audio_file.stem}.pt"
        out_path = CACHE_DIR / out_name

        if out_path.exists():
            meta.append((audio_file.name, label, str(out_path)))
            continue

        # If we reach here, the file needs processing.
        print(f"Processing file {i+1}/{num_files}: {audio_file.name}")
        mel = process_audio(str(audio_file), bg_files=bg_files)
        
        torch.save(mel, out_path)
        meta.append((audio_file.name, label, str(out_path)))

    import csv
    with open(out_meta, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "tensor_path"])
        writer.writerows(meta)

    print(f"Finished. Preprocessed {len(meta)} total files. Saved to {CACHE_DIR}")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to BirdCLEF audio")
    parser.add_argument("--bg_dir", type=str, default=None, help="Path to background noise wavs")
    parser.add_argument("--out_meta", type=str, default="metadata.csv")
    args = parser.parse_args()

    preprocess_dataset(args.audio_dir, args.bg_dir, args.out_meta)
