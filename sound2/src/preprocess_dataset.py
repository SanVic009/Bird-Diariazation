import os
import json
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils.audio import load_audio, mel_spectrogram
import argparse

def preprocess_audio(audio_path, cfg):
    """Process a single audio file to mel spectrogram."""
    y, sr = load_audio(audio_path, cfg["audio"]["sample_rate"])
    
    # Ensure minimum length or pad
    target_len = int(sr * max(3.0, cfg["audio"]["frame_sec"]*4))
    if len(y) < target_len:
        pad_len = target_len - len(y)
        y = np.pad(y, (0, pad_len))
    else:
        y = y[:target_len]
    
    # Convert to mel spectrogram
    M, _, _ = mel_spectrogram(
        y, sr,
        n_mels=cfg["audio"]["n_mels"],
        fmin=cfg["audio"]["fmin"],
        fmax=cfg["audio"]["fmax"],
        frame_sec=cfg["audio"]["frame_sec"],
        hop_sec=cfg["audio"]["hop_sec"]
    )
    
    # Normalize
    M = (M - M.mean()) / (M.std() + 1e-6)
    return M

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Create preprocessed data directory
    preproc_dir = os.path.join(cfg["outputs"]["dir"], "preprocessed")
    os.makedirs(preproc_dir, exist_ok=True)
    
    # Process training data
    train_df = pd.read_csv(os.path.join(cfg["data"]["splits_dir"], "train.csv"))
    val_df = pd.read_csv(os.path.join(cfg["data"]["splits_dir"], "val.csv"))
    
    # Load label mapping
    with open(cfg["data"]["labels_json"], "r") as f:
        label_meta = json.load(f)
    
    # Process train and validation sets
    for split, df in [("train", train_df), ("val", val_df)]:
        split_dir = os.path.join(preproc_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        print(f"\nProcessing {split} set...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # Maintain the same subfolder structure as the original dataset
            audio_path = os.path.join(cfg["data"]["root"], 
                                    cfg["data"]["audio_subdir"], 
                                    row["filename"])
            
            # Create output directory maintaining the same structure
            species_dir = os.path.dirname(row["filename"])
            out_dir = os.path.join(split_dir, species_dir)
            os.makedirs(out_dir, exist_ok=True)
            
            # Generate output path
            out_path = os.path.join(split_dir, row["filename"].replace(".ogg", ".pt"))
            
            # Skip if already processed
            if os.path.exists(out_path):
                continue
                
            # Process audio
            try:
                mel_spec = preprocess_audio(audio_path, cfg)
                
                # Create label
                label_idx = label_meta["label_to_idx"][row["primary_label"]]
                multi_hot = np.zeros(len(label_meta["labels"]), dtype=np.float32)
                multi_hot[label_idx] = 1.0
                
                # Save preprocessed data
                torch.save({
                    'mel_spectrogram': torch.from_numpy(mel_spec),
                    'label': torch.from_numpy(multi_hot),
                    'filename': row["filename"],
                    'primary_label': row["primary_label"]
                }, out_path)
                
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                continue

    print("\nPreprocessing completed!")

if __name__ == "__main__":
    main()
