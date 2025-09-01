import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchaudio

class BirdClefDataset(Dataset):
    def __init__(self, processed_csv_file, transform=None):
        """
        Args:
            processed_csv_file (str): Path to the metadata CSV in the processed directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(processed_csv_file)
        self.audio_files = self.df['filepath'].values # Assuming 'filepath' column exists
        self.labels = self.df['primary_label'].values # Assuming 'primary_label' column exists

        # Create label encoder
        self.unique_labels = sorted(self.df["primary_label"].unique())
        self.label2idx = {l: i for i, l in enumerate(self.unique_labels)}
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.label2idx[self.labels[idx]] # Convert label to index

        try:
            spectrogram = np.load(audio_path)
            # Convert numpy array to torch tensor
            spectrogram = torch.from_numpy(spectrogram).float()
            # Add a channel dimension if it's missing (e.g., for CNNs expecting (C, H, W))
            if spectrogram.ndim == 2:
                spectrogram = spectrogram.unsqueeze(0) # Add channel dimension
        except Exception as e:
            print(f"[ERROR] Failed to load spectrogram {audio_path}: {e}")
            logging.error(f"[ERROR] Failed to load spectrogram {audio_path}: {e}")
            return None # Indicate failure to load

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, torch.tensor(label, dtype=torch.long)

def pad_collate_fn(batch):
    # Filter out None values from batch (failed loads)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None # Return None if batch is empty after filtering

    # batch is a list of (mel, label) tuples
    mels, labels = zip(*batch)
    
    # Find max width (time dimension)
    max_width = max(m.shape[2] for m in mels)
    
    # Pad mels to max_width
    padded_mels = []
    for m in mels:
        pad_width = max_width - m.shape[2]
        # Pad on the right side of the time dimension
        p = torch.nn.functional.pad(m, (0, pad_width), mode='constant', value=0)
        padded_mels.append(p)
        
    # Stack them
    mels = torch.stack(padded_mels)
    labels = torch.stack(labels)
    
    return mels, labels

def create_dataloaders(processed_csv_file='processed/metadata.csv', batch_size=64, num_workers=4, split=0.8, transform=None, **kwargs):
    """Splits BirdCLEF dataset into train/val loaders from pre-processed files."""
    full_ds = BirdClefDataset(processed_csv_file, transform=transform)
    n_train = int(len(full_ds) * split)
    n_val = len(full_ds) - n_train

    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=pad_collate_fn)
    return train_loader, val_loader, full_ds.labels
