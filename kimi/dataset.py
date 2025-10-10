# dataset.py
import os, random, librosa
import torch, numpy as np
from torch.utils.data import Dataset

class BirdCLEFUnsupervised(Dataset):
    def __init__(self, root, dur=2.0, sr=32_000, augment=None):
        self.sr = sr
        self.dur = int(dur * sr)
        self.augment = augment
        # collect all ogg paths
        self.files = []
        for sub in os.listdir(root):
            subdir = os.path.join(root, sub)
            if os.path.isdir(subdir):
                self.files += [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith(".ogg")]
        print(f"Found {len(self.files)} training recordings")

    def __len__(self): return len(self.files) * 10   # 10 crops per file

    def __getitem__(self, idx):
        file = self.files[idx % len(self.files)]
        try:
            y, _ = librosa.load(file, sr=self.sr, mono=True)
        except Exception as e:
            # Skip corrupted files by trying the next file
            print(f"Warning: Skipping corrupted file {file}: {e}")
            return self.__getitem__((idx + 1) % len(self.files))
        
        # random crop
        if len(y) > self.dur:
            start = random.randint(0, len(y) - self.dur)
            y = y[start:start + self.dur]
        else:
            y = np.pad(y, (0, self.dur - len(y)))
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)  # (1, T)
        if self.augment:
            y = y.unsqueeze(0)  # Add batch dimension: (1, 1, T)
            y = self.augment(y)
            y = y.squeeze(0)    # Remove batch dimension: (1, T)
        return y