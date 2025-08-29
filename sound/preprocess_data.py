#!/usr/bin/env python3
"""
BirdCLEF-2024 Data Preprocessing (Improved)
===========================================
- Extracts richer features: log-mel spectrogram, MFCC, chroma.
- Normalizes features per file.
- Leaves augmentation to training time (not applied here).
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import librosa
import noisereduce as nr
from tqdm import tqdm
from training_configs import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocess_data.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def extract_features(audio, sr, audio_config):
    try:
        max_len = audio_config['max_len']
        if len(audio) > max_len * sr:
            audio = audio[:max_len * sr]
        else:
            audio = np.pad(audio, (0, max_len * sr - len(audio)), mode='constant')

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr,
            n_mfcc=audio_config['n_mfcc'],
            n_fft=audio_config['n_fft'],
            hop_length=audio_config['hop_length']
        )

        # Log-mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=audio_config['n_fft'],
            hop_length=audio_config['hop_length'], n_mels=64
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=audio_config['n_fft'], hop_length=audio_config['hop_length'])

        # Ensure consistent frame count
        target_frames = mfcc.shape[1]
        def resize_feat(feat):
            return np.resize(feat, (feat.shape[0], target_frames))

        mfcc, log_mel, chroma = map(resize_feat, [mfcc, log_mel, chroma])

        # Combine
        features = np.vstack([mfcc, log_mel, chroma])

        # Normalize per file (zero mean, unit variance)
        features = (features - np.mean(features)) / (np.std(features) + 1e-6)

        return features

    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        return np.zeros((audio_config['n_mfcc'] + 64 + 12, audio_config['max_len']))

def process_file(args):
    filename, file_path, output_path, audio_config = args
    if os.path.exists(file_path):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            audio, sr = librosa.load(file_path, sr=audio_config['sample_rate'])
            if audio_config['noise_reduction']:
                # Estimate noise profile from the first few frames (usually containing background noise)
                noise_clip = audio[:int(sr * 0.5)]  # Use first 0.5 seconds for noise profile
                
                # Apply advanced noise reduction with spectral gating
                audio = nr.reduce_noise(
                    y=audio,
                    sr=sr,
                    prop_decrease=0.75,  # How much to reduce the noise
                    n_fft=2048,
                    win_length=2048,
                    hop_length=512,
                    n_std_thresh_stationary=1.5,  # More aggressive stationary noise reduction
                    stationary=True,  # Assume relatively constant background noise
                    time_constant_s=2.0  # Longer time constant for better noise estimation
                )
                
                # Additional step: Spectral Subtraction for remaining background noise
                S = librosa.stft(audio)
                S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
                noise_floor = np.mean(S_db[:, :50])  # Estimate noise floor from first 50 frames
                mask = S_db > (noise_floor + 20)  # Keep only components 20dB above noise floor
                S_clean = S * mask
                audio = librosa.istft(S_clean)
                
            features = extract_features(audio, sr, audio_config)
            np.save(output_path, features)
            return True, filename
        except Exception as e:
            return False, f"Error processing {filename}: {str(e)}"
    return False, f"File not found: {filename}"

def preprocess_data(config_name):
    config = get_config(config_name)
    audio_config = config['audio']

    import multiprocessing
    n_cores = multiprocessing.cpu_count()
    os.environ['NUMBA_NUM_THREADS'] = str(n_cores)
    logger.info(f"Using {n_cores} CPU cores")

    output_dir = os.path.join('train_features2', config_name)
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_csv('train_metadata.csv')
    audio_dir = 'train_audio'

    process_args = []
    for _, row in train_df.iterrows():
        filename = row['filename']
        file_path = os.path.join(audio_dir, filename)
        output_path = os.path.join(output_dir, filename.replace('.ogg', '.npy'))
        process_args.append((filename, file_path, output_path, audio_config))

    from concurrent.futures import ProcessPoolExecutor, as_completed
    n_workers = max(1, n_cores - 1)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_file, args) for args in process_args]
        with tqdm(total=len(futures), desc="Preprocessing") as pbar:
            for future in as_completed(futures):
                success, msg = future.result()
                if not success:
                    logger.warning(msg)
                pbar.update(1)

    logger.info("Preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='full', choices=['quick','dev','full','balanced','gpu'])
    args = parser.parse_args()
    preprocess_data(args.config)
