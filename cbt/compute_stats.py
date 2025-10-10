#!/usr/bin/env python3
"""
compute_stats.py - Compute dataset statistics for better normalization
"""

import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm

def compute_dataset_stats(data_dir="cache_mels/"):
    """Compute mean and std across entire dataset"""
    data_path = Path(data_dir)
    files = list(data_path.glob("*.pt"))
    
    print(f"Computing stats from {len(files)} files...")
    
    # Collect all values
    all_values = []
    
    for file in tqdm(files):
        try:
            x = torch.load(file)
            all_values.append(x.flatten())
        except:
            continue
    
    # Concatenate and compute stats
    all_values = torch.cat(all_values)
    mean = all_values.mean().item()
    std = all_values.std().item()
    
    print(f"Dataset mean: {mean:.6f}")
    print(f"Dataset std: {std:.6f}")
    
    # Save stats
    torch.save({'mean': mean, 'std': std}, 'dataset_stats.pt')
    print("Stats saved to dataset_stats.pt")
    
    return mean, std

if __name__ == "__main__":
    compute_dataset_stats()