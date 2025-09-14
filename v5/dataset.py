import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging

class BirdClefDataset(Dataset):
    def __init__(self, metadata_csv_file, audio_dir=None, transform=None, use_processed=False):
        """
        Args:
            metadata_csv_file (str): Path to the metadata CSV file.
            audio_dir (str): Path to the audio directory. If None, inferred from metadata.
            transform (callable, optional): Optional transform to be applied on a sample.
            use_processed (bool): If True, expects processed .npy files, else raw .flac files
        """
        self.df = pd.read_csv(metadata_csv_file)
        self.use_processed = use_processed
        self.transform = transform
        
        if use_processed:
            # For processed data (RFCX)
            self.audio_files = self.df['filepath'].values
            self.labels = self.df['species_id'].values
            # Extract individual species from comma-separated combinations (for compatibility)
            all_species = set()
            for label_str in self.df["species_id"]:
                if isinstance(label_str, str) and ',' in label_str:
                    species = [s.strip() for s in label_str.split(',')]
                    all_species.update(species)
                else:
                    all_species.add(str(label_str).strip())
            self.unique_labels = sorted(list(all_species))
        else:
            # For original BirdCLEF data
            if audio_dir is None:
                # Infer audio directory from metadata file path
                audio_dir = os.path.join(os.path.dirname(metadata_csv_file), 'train')
            
            self.audio_dir = audio_dir
            self.labels = self.df['species_id'].values
            # Build full audio file paths
            self.audio_files = [os.path.join(audio_dir, f"{filename}.flac") for filename in self.df['recording_id'].values]
            # Simple species extraction
            self.unique_labels = sorted(self.df['species_id'].unique())
        
        self.label2idx = {l: i for i, l in enumerate(self.unique_labels)}
        self.n_classes = len(self.unique_labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label_str = self.labels[idx]
        
        if self.use_processed:
            # Extract the first species from comma-separated combinations
            if isinstance(label_str, str) and ',' in label_str:
                first_species = label_str.split(',')[0].strip()
            else:
                first_species = str(label_str).strip()
        else:
            # For original data, label is already a single species
            first_species = str(label_str).strip()
            
        label = self.label2idx[first_species] # Convert label to index

        try:
            if self.use_processed:
                # Load processed spectrogram (.npy file)
                spectrogram = np.load(audio_path)
                spectrogram = torch.from_numpy(spectrogram).float()
            else:
                # Load and process raw audio (.flac file)
                import librosa
                # Load audio file
                audio, sr = librosa.load(audio_path, sr=32000)
                # Create mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=audio, sr=sr, n_mels=128, fmax=16000
                )
                # Convert to log scale
                spectrogram = torch.from_numpy(librosa.power_to_db(mel_spec)).float()
            
            # Add a channel dimension if it's missing (e.g., for CNNs expecting (C, H, W))
            if spectrogram.ndim == 2:
                spectrogram = spectrogram.unsqueeze(0) # Add channel dimension
                
        except Exception as e:
            print(f"[ERROR] Failed to load audio {audio_path}: {e}")
            logging.error(f"[ERROR] Failed to load audio {audio_path}: {e}")
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

def create_dataloaders(metadata_csv_file, audio_dir=None, 
                      batch_size=64, num_workers=4, split=0.8, transform=None, use_processed=False, **kwargs):
    """Creates train/val loaders from the dataset."""
    full_ds = BirdClefDataset(metadata_csv_file, audio_dir=audio_dir, transform=transform, use_processed=use_processed)
    n_train = int(len(full_ds) * split)
    n_val = len(full_ds) - n_train

    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=pad_collate_fn)
    return train_loader, val_loader, full_ds.unique_labels, full_ds.n_classes