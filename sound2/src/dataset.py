import os, json, random, numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset
from utils.audio import load_audio, mel_spectrogram
from utils.augment import mix_signals, random_crop

class BirdFrameDataset(Dataset):
    def __init__(self, csv_path, data_root, audio_subdir, labels_json,
                 sample_rate=32000, n_mels=128, fmin=50, fmax=14000,
                 frame_sec=1.0, hop_sec=0.5, mix_prob=0.5, mix_snr_db=3.0, train=True):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        with open(labels_json, "r") as f:
            meta = json.load(f)
        self.labels = meta["labels"]
        self.label_to_idx = meta["label_to_idx"]
        self.num_classes = len(self.labels)
        self.data_root = data_root
        self.audio_subdir = audio_subdir
        self.sr = sample_rate
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.frame_sec = frame_sec
        self.hop_sec = hop_sec
        self.mix_prob = mix_prob if train else 0.0
        self.mix_snr_db = mix_snr_db
        self.train = train

    def __len__(self):
        return len(self.df)

    def _get_path(self, row):
        return os.path.join(self.data_root, self.audio_subdir, row["filename"])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y, sr = load_audio(self._get_path(row), self.sr)
        # optional random crop for train to a fixed few seconds to speed up
        target_len = int(self.sr * max(3.0, self.frame_sec*4))
        if len(y) < target_len:
            # Pad with zeros if audio is too short
            pad_len = target_len - len(y)
            y = np.pad(y, (0, pad_len))
        else:
            # Crop if audio is too long
            y = random_crop(y, target_len) if self.train else y[:target_len]

        # Mix augmentation: mix with another random species
        label_idx = self.label_to_idx[row["primary_label"]]
        multi_hot = np.zeros(self.num_classes, dtype=np.float32)
        multi_hot[label_idx] = 1.0

        if self.train and random.random() < self.mix_prob:
            # pick a different species row
            for _ in range(10):
                j = random.randint(0, len(self.df)-1)
                if self.df.iloc[j]["primary_label"] != row["primary_label"]:
                    break
            row2 = self.df.iloc[j]
            y2, _ = load_audio(self._get_path(row2), self.sr)
            y2 = random_crop(y2, len(y))
            y = mix_signals(y, y2, snr_db=self.mix_snr_db)
            multi_hot[self.label_to_idx[row2["primary_label"]]] = 1.0

        # Mel-spec
        M, hop_length, n_fft = mel_spectrogram(
            y, self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
            frame_sec=self.frame_sec, hop_sec=self.hop_sec
        )  # [n_mels, T]
        # Normalize per-utterance
        M = (M - M.mean()) / (M.std() + 1e-6)
        # Targets per frame: use same multi-hot for all frames in this clip
        T = M.shape[1]
        target = np.repeat(multi_hot[None, :], T, axis=0)  # [T, C]

        return (
            torch.tensor(M).unsqueeze(0),  # [1, n_mels, T]
            torch.tensor(target)          # [T, C]
        )
