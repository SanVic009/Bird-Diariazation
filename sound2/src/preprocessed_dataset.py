import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class PreprocessedBirdDataset(Dataset):
    def __init__(self, csv_path, preproc_dir, labels_json, mix_prob=0.5, mix_snr_db=3.0, train=True):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        with open(labels_json, "r") as f:
            meta = json.load(f)
        self.labels = meta["labels"]
        self.label_to_idx = meta["label_to_idx"]
        self.num_classes = len(self.labels)
        self.preproc_dir = preproc_dir
        self.mix_prob = mix_prob if train else 0.0
        self.mix_snr_db = mix_snr_db
        self.train = train

    def __len__(self):
        return len(self.df)

    def _get_path(self, row):
        return os.path.join(self.preproc_dir, f"{os.path.splitext(row['filename'])[0]}.pt")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = torch.load(self._get_path(row))
        mel_spec = data['mel_spectrogram']
        label = data['label']
        
        if self.train and random.random() < self.mix_prob:
            # Mix with another random sample
            for _ in range(10):
                j = random.randint(0, len(self.df)-1)
                if self.df.iloc[j]["primary_label"] != row["primary_label"]:
                    break
            
            mix_data = torch.load(self._get_path(self.df.iloc[j]))
            mix_mel = mix_data['mel_spectrogram']
            
            # Adjust the length if needed
            min_len = min(mel_spec.shape[1], mix_mel.shape[1])
            mel_spec = mel_spec[:, :min_len]
            mix_mel = mix_mel[:, :min_len]
            
            # Mix the spectrograms (simple average for demonstration)
            mix_weight = np.random.beta(5, 5)  # Random mixing weight
            mel_spec = mel_spec * mix_weight + mix_mel * (1 - mix_weight)
            
            # Mix the labels
            label = torch.max(torch.stack([label, mix_data['label']]), dim=0)[0]
        
        # Repeat the label for each time frame
        T = mel_spec.shape[1]
        target = label.unsqueeze(0).repeat(T, 1)  # [T, C]
        
        return (
            mel_spec.unsqueeze(0),  # [1, n_mels, T]
            target                  # [T, C]
        )
