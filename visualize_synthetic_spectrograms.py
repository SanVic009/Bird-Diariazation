#!/usr/bin/env python3
"""
Script to visualize mel spectrograms from 5 random synthetic audio files.
Displays all spectrograms in a single PNG file.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path

def load_audio_and_create_spectrogram(audio_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """
    Load audio file and convert to mel spectrogram.
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate
        n_mels: Number of mel frequency bins
        n_fft: FFT window size
        hop_length: Hop length for STFT
    
    Returns:
        mel_spectrogram: Log mel spectrogram
        duration: Audio duration in seconds
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        duration = len(y) / sr
        
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec, duration
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None

def visualize_random_synthetic_spectrograms(synthetic_audio_dir="synthetic_audio", 
                                          output_file="synthetic_spectrograms_visualization.png",
                                          num_samples=5):
    """
    Create visualization of mel spectrograms from random synthetic audio files.
    
    Args:
        synthetic_audio_dir: Directory containing synthetic audio files
        output_file: Output PNG file name
        num_samples: Number of random samples to visualize
    """
    
    # Check if synthetic_audio directory exists
    if not os.path.exists(synthetic_audio_dir):
        print(f"Error: {synthetic_audio_dir} directory not found!")
        return
    
    # Get all audio files from synthetic_audio directory
    audio_files = []
    for ext in ['*.wav', '*.ogg', '*.mp3', '*.flac']:
        audio_files.extend(Path(synthetic_audio_dir).glob(ext))
    
    if len(audio_files) == 0:
        print(f"No audio files found in {synthetic_audio_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files in {synthetic_audio_dir}")
    
    # Randomly select files
    selected_files = random.sample(audio_files, min(num_samples, len(audio_files)))
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    fig.suptitle('Mel Spectrograms from Random Synthetic Audio Files', fontsize=16, y=0.98)
    
    # Process each selected file
    for i, audio_file in enumerate(selected_files):
        print(f"Processing {i+1}/{len(selected_files)}: {audio_file.name}")
        
        # Create mel spectrogram
        mel_spec, duration = load_audio_and_create_spectrogram(str(audio_file))
        
        if mel_spec is not None:
            # Plot spectrogram
            img = librosa.display.specshow(
                mel_spec, 
                sr=22050, 
                hop_length=512, 
                x_axis='time', 
                y_axis='mel',
                ax=axes[i],
                cmap='viridis'
            )
            
            # Extract species from filename (assuming format like "species1_species2_00000.wav")
            filename = audio_file.stem
            species_info = filename.split('_')[:-1]  # Remove the number part
            species_str = ' + '.join(species_info) if len(species_info) > 1 else species_info[0]
            
            axes[i].set_title(f'{species_str} ({duration:.2f}s)\n{filename}')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Mel Frequency')
            
            # Add colorbar
            plt.colorbar(img, ax=axes[i], format='%+2.0f dB')
        else:
            axes[i].text(0.5, 0.5, f'Failed to load:\n{audio_file.name}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'Error loading file {i+1}')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as: {output_file}")
    print(f"Processed files:")
    for i, file in enumerate(selected_files):
        print(f"  {i+1}. {file.name}")

def main():
    """Main function to run the visualization."""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Creating mel spectrogram visualization from synthetic audio files...")
    print("="*60)
    
    # Run visualization
    visualize_random_synthetic_spectrograms(
        synthetic_audio_dir="synthetic_audio",
        output_file="synthetic_spectrograms_visualization.png",
        num_samples=5
    )

if __name__ == "__main__":
    main()
