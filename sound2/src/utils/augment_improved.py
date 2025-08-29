import numpy as np
import random
import torch
import torchaudio
import torch.nn as nn

def rms(x):
    return np.sqrt(np.mean(np.maximum(1e-12, x**2)))

def mix_signals(a, b, snr_db=3.0):
    # Scale b to achieve target SNR vs a
    ra, rb = rms(a), rms(b)
    if rb < 1e-9:
        return a
    snr = 10.0**(snr_db/20.0)
    b_scaled = b * (ra / (rb * snr))
    # Align lengths
    L = max(len(a), len(b_scaled))
    a_pad = np.pad(a, (0, L-len(a)))
    b_pad = np.pad(b_scaled, (0, L-len(b_scaled)))
    return a_pad + b_pad

def random_crop(x, target_len):
    if len(x) <= target_len:
        return np.pad(x, (0, target_len-len(x)))
    start = random.randint(0, len(x)-target_len)
    return x[start:start+target_len]

class AudioAugmentor:
    def __init__(self, sample_rate=32000):
        self.sample_rate = sample_rate
        
    def time_shift(self, audio, shift_limit=0.2):
        """Randomly shift audio in time"""
        if random.random() < 0.5:
            return audio
        shift = int(random.uniform(-shift_limit, shift_limit) * len(audio))
        return np.roll(audio, shift)
    
    def pitch_shift(self, audio, pitch_limit=3):
        """Random pitch shifting"""
        if random.random() < 0.5:
            return audio
        n_steps = random.uniform(-pitch_limit, pitch_limit)
        return torchaudio.functional.pitch_shift(
            torch.tensor(audio), 
            self.sample_rate, 
            n_steps
        ).numpy()
    
    def time_stretch(self, audio, stretch_limit=0.2):
        """Random time stretching"""
        if random.random() < 0.5:
            return audio
        rate = 1.0 + random.uniform(-stretch_limit, stretch_limit)
        return torchaudio.functional.time_stretch(
            torch.tensor(audio).unsqueeze(0), 
            rate
        ).squeeze(0).numpy()
    
    def add_gaussian_noise(self, audio, noise_limit=0.005):
        """Add random gaussian noise"""
        if random.random() < 0.5:
            return audio
        noise = np.random.normal(0, random.uniform(0, noise_limit), len(audio))
        return audio + noise
    
    def random_gain(self, audio, gain_limit=10):
        """Apply random gain"""
        if random.random() < 0.5:
            return audio
        gain = random.uniform(-gain_limit, gain_limit)
        return audio * (10 ** (gain / 20.0))

class SpecAugment(nn.Module):
    def __init__(self, time_mask_param=40, freq_mask_param=20, num_masks=1):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_masks = num_masks
        
    def forward(self, spec):
        """
        spec: [1, n_mels, time]
        """
        spec = spec.clone()
        
        # Time masking
        for _ in range(self.num_masks):
            if random.random() < 0.5:
                time_mask_len = random.randint(0, self.time_mask_param)
                time_start = random.randint(0, spec.shape[2] - time_mask_len)
                spec[:, :, time_start:time_start+time_mask_len] = 0
        
        # Frequency masking
        for _ in range(self.num_masks):
            if random.random() < 0.5:
                freq_mask_len = random.randint(0, self.freq_mask_param)
                freq_start = random.randint(0, spec.shape[1] - freq_mask_len)
                spec[:, freq_start:freq_start+freq_mask_len, :] = 0
                
        return spec

def apply_augmentations(audio, sr=32000):
    """Apply all augmentations with some probability"""
    augmentor = AudioAugmentor(sr)
    
    # Chain augmentations
    audio = augmentor.time_shift(audio)
    audio = augmentor.pitch_shift(audio)
    audio = augmentor.time_stretch(audio)
    audio = augmentor.add_gaussian_noise(audio)
    audio = augmentor.random_gain(audio)
    
    return audio
