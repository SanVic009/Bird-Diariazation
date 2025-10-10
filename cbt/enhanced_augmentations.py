#!/usr/bin/env python3
"""
enhanced_augmentations.py - Advanced Audio Augmentations for Bird Diarization

Features:
- Frequency shifting (pitch variations)
- Time stretching simulation
- Gaussian noise injection
- Advanced masking techniques
- Spectral augmentations
- Environmental noise simulation
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import math

class AdvancedAudioAugmenter:
    """Advanced audio augmentation for bird diarization"""
    
    def __init__(self, 
                 time_mask_prob=0.4,
                 freq_mask_prob=0.3,
                 freq_shift_prob=0.3,
                 noise_prob=0.2,
                 mixup_prob=0.15,
                 spec_augment_prob=0.3):
        self.time_mask_prob = time_mask_prob
        self.freq_mask_prob = freq_mask_prob
        self.freq_shift_prob = freq_shift_prob
        self.noise_prob = noise_prob
        self.mixup_prob = mixup_prob
        self.spec_augment_prob = spec_augment_prob
        
    def __call__(self, x, training=True):
        """Apply augmentations to input spectrogram"""
        if not training:
            return x
            
        x = x.clone()
        
        # Apply different augmentations with specified probabilities
        x = self.time_masking(x)
        x = self.frequency_masking(x)
        x = self.frequency_shifting(x)
        x = self.gaussian_noise(x)
        x = self.spectral_augmentations(x)
        x = self.volume_augmentation(x)
        
        return x
    
    def time_masking(self, x):
        """Enhanced time masking with multiple masks"""
        if torch.rand(1) < self.time_mask_prob:
            time_dim = x.shape[-1]
            
            # Multiple smaller masks instead of one large mask
            num_masks = torch.randint(1, 4, (1,)).item()
            
            for _ in range(num_masks):
                mask_size = torch.randint(5, min(30, time_dim // 4), (1,)).item()
                mask_start = torch.randint(0, max(1, time_dim - mask_size), (1,)).item()
                
                # Apply different masking strategies
                mask_type = random.choice(['zero', 'noise', 'interpolate'])
                
                if mask_type == 'zero':
                    x[..., mask_start:mask_start + mask_size] = 0
                elif mask_type == 'noise':
                    noise = torch.randn_like(x[..., mask_start:mask_start + mask_size]) * 0.01
                    x[..., mask_start:mask_start + mask_size] = noise
                else:  # interpolate
                    if mask_start > 0 and mask_start + mask_size < time_dim:
                        start_val = x[..., mask_start-1:mask_start]
                        end_val = x[..., mask_start+mask_size:mask_start+mask_size+1]
                        # Linear interpolation
                        for i in range(mask_size):
                            alpha = i / (mask_size - 1) if mask_size > 1 else 0
                            x[..., mask_start + i:mask_start + i + 1] = (1 - alpha) * start_val + alpha * end_val
        
        return x
    
    def frequency_masking(self, x):
        """Enhanced frequency masking"""
        if torch.rand(1) < self.freq_mask_prob:
            freq_dim = x.shape[-2]
            
            # Multiple frequency masks
            num_masks = torch.randint(1, 3, (1,)).item()
            
            for _ in range(num_masks):
                mask_size = torch.randint(3, min(15, freq_dim // 3), (1,)).item()
                mask_start = torch.randint(0, max(1, freq_dim - mask_size), (1,)).item()
                
                # Different masking intensities
                if torch.rand(1) < 0.5:
                    # Complete masking
                    x[..., mask_start:mask_start + mask_size, :] = 0
                else:
                    # Partial masking (reduce intensity)
                    x[..., mask_start:mask_start + mask_size, :] *= torch.rand(1) * 0.3
        
        return x
    
    def frequency_shifting(self, x):
        """Simulate pitch variations by shifting frequency bins"""
        if torch.rand(1) < self.freq_shift_prob:
            freq_dim = x.shape[-2]
            
            # Random shift amount (positive or negative)
            max_shift = min(8, freq_dim // 8)
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            
            if shift != 0:
                # Roll frequency dimension
                x = torch.roll(x, shift, dims=-2)
                
                # Fill the wrapped regions with noise or zeros
                if shift > 0:
                    if torch.rand(1) < 0.5:
                        x[..., :shift, :] = 0  # Zero padding
                    else:
                        x[..., :shift, :] = torch.randn_like(x[..., :shift, :]) * 0.01  # Noise
                else:
                    if torch.rand(1) < 0.5:
                        x[..., shift:, :] = 0
                    else:
                        x[..., shift:, :] = torch.randn_like(x[..., shift:, :]) * 0.01
        
        return x
    
    def gaussian_noise(self, x):
        """Add Gaussian noise to simulate environmental conditions"""
        if torch.rand(1) < self.noise_prob:
            # Variable noise intensity
            noise_level = torch.FloatTensor(1).uniform_(0.005, 0.03).item()
            noise = torch.randn_like(x) * noise_level
            
            # Apply noise with varying intensity across frequency bands
            # Lower frequencies often have more environmental noise
            freq_dim = x.shape[-2]
            noise_profile = torch.linspace(1.5, 0.5, freq_dim).view(-1, 1)
            if len(x.shape) == 4:  # Batch dimension
                noise_profile = noise_profile.unsqueeze(0).unsqueeze(0)
            else:
                noise_profile = noise_profile.unsqueeze(0)
            
            noise = noise * noise_profile.to(x.device)
            x = x + noise
        
        return x
    
    def spectral_augmentations(self, x):
        """Advanced spectral augmentations"""
        if torch.rand(1) < self.spec_augment_prob:
            # Spectral rolling (circular shift in frequency)
            if torch.rand(1) < 0.3:
                roll_amount = torch.randint(1, x.shape[-2] // 4, (1,)).item()
                x = torch.roll(x, roll_amount, dims=-2)
            
            # Spectral dropout (randomly zero out frequency bins)
            if torch.rand(1) < 0.3:
                dropout_prob = torch.FloatTensor(1).uniform_(0.05, 0.15).item()
                dropout_mask = torch.rand_like(x) < dropout_prob
                x = x * (~dropout_mask)
            
            # Contrast adjustment
            if torch.rand(1) < 0.4:
                contrast_factor = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
                mean_val = x.mean()
                x = (x - mean_val) * contrast_factor + mean_val
        
        return x
    
    def volume_augmentation(self, x):
        """Simulate different recording volumes"""
        if torch.rand(1) < 0.3:
            # Random volume scaling
            volume_factor = torch.FloatTensor(1).uniform_(0.7, 1.3).item()
            x = x * volume_factor
        
        return x
    
    def mixup(self, x1, x2, alpha=0.2):
        """Mixup augmentation between two samples"""
        if torch.rand(1) < self.mixup_prob:
            lam = torch.distributions.Beta(alpha, alpha).sample()
            x = lam * x1 + (1 - lam) * x2
            return x, lam
        return x1, 1.0
    
    def cutmix(self, x1, x2):
        """CutMix augmentation for spectrograms"""
        if torch.rand(1) < 0.1:  # Lower probability for cutmix
            h, w = x1.shape[-2:]
            
            # Random rectangular region
            cut_h = torch.randint(h // 4, h // 2, (1,)).item()
            cut_w = torch.randint(w // 4, w // 2, (1,)).item()
            
            start_h = torch.randint(0, h - cut_h, (1,)).item()
            start_w = torch.randint(0, w - cut_w, (1,)).item()
            
            x1_copy = x1.clone()
            x1_copy[..., start_h:start_h+cut_h, start_w:start_w+cut_w] = \
                x2[..., start_h:start_h+cut_h, start_w:start_w+cut_w]
            
            return x1_copy, (cut_h * cut_w) / (h * w)
        
        return x1, 0.0

class ImprovedDiarizationDataset:
    """Enhanced dataset class with advanced augmentations"""
    
    def __init__(self, root, segment_length=2.0, training=True, augmentation_strength=1.0):
        from pathlib import Path
        
        self.root = Path(root)
        self.files = list(self.root.glob("*.pt"))
        self.segment_length = segment_length
        self.training = training
        
        # Enhanced augmentations
        self.augmenter = AdvancedAudioAugmenter(
            time_mask_prob=0.4 * augmentation_strength,
            freq_mask_prob=0.3 * augmentation_strength,
            freq_shift_prob=0.3 * augmentation_strength,
            noise_prob=0.2 * augmentation_strength,
            mixup_prob=0.15 * augmentation_strength,
            spec_augment_prob=0.3 * augmentation_strength
        )
        
        self.file_ids = [f.stem.split("_")[0] for f in self.files]
        
        print("--- ENHANCED DIARIZATION DATASET ---")
        print(f"Found {len(self.files)} audio segments.")
        print(f"Training mode: {training}")
        print(f"Augmentation strength: {augmentation_strength}")
        print("----------------------------------")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        x = torch.load(self.files[idx])
        
        # Pad or crop to fixed size
        target_width = 501
        current_width = x.shape[-1]
        
        if current_width > target_width:
            # Smart cropping - avoid cutting in the middle of important features
            if self.training and torch.rand(1) < 0.7:
                # Random crop for augmentation
                start_idx = torch.randint(0, current_width - target_width + 1, (1,)).item()
            else:
                # Center crop for validation/testing
                start_idx = (current_width - target_width) // 2
            x = x[..., start_idx:start_idx + target_width]
        else:
            # Symmetric padding when possible
            pad_width = target_width - current_width
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            x = F.pad(x, (pad_left, pad_right))
        
        # Enhanced normalization
        x = self._enhanced_normalize(x)
        
        x = x.unsqueeze(0)  # [1, n_mels, time]
        
        # Generate two augmented views for contrastive learning
        if self.training:
            x1 = self.augmenter(x.clone(), training=True)
            x2 = self.augmenter(x.clone(), training=True)
            
            # Occasionally apply cross-sample augmentations
            if len(self.files) > 1 and torch.rand(1) < 0.1:
                # Load another sample for mixup/cutmix
                other_idx = torch.randint(0, len(self.files), (1,)).item()
                if other_idx != idx:
                    x_other = torch.load(self.files[other_idx])
                    x_other = self._prepare_other_sample(x_other, target_width)
                    x_other = self._enhanced_normalize(x_other).unsqueeze(0)
                    
                    # Apply mixup to one of the views
                    if torch.rand(1) < 0.5:
                        x2, _ = self.augmenter.mixup(x2, x_other)
        else:
            # No augmentation for validation/testing
            x1 = x.clone()
            x2 = x.clone()
        
        return x1, x2, idx
    
    def _enhanced_normalize(self, x):
        """Enhanced normalization strategy"""
        # Compute statistics
        mean = x.mean()
        std = x.std()
        
        if std > 1e-6:
            # Z-score normalization
            x_norm = (x - mean) / std
            
            # Optional: Robust normalization using percentiles
            if self.training and torch.rand(1) < 0.3:
                q25 = x.quantile(0.25)
                q75 = x.quantile(0.75)
                iqr = q75 - q25
                if iqr > 1e-6:
                    x_norm = (x - x.median()) / iqr
            
            # Clip extreme values
            x_norm = torch.clamp(x_norm, -3, 3)
            
        else:
            x_norm = x - mean
        
        return x_norm
    
    def _prepare_other_sample(self, x_other, target_width):
        """Prepare another sample for cross-sample augmentations"""
        current_width = x_other.shape[-1]
        
        if current_width > target_width:
            start_idx = torch.randint(0, current_width - target_width + 1, (1,)).item()
            x_other = x_other[..., start_idx:start_idx + target_width]
        else:
            pad_width = target_width - current_width
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            x_other = F.pad(x_other, (pad_left, pad_right))
        
        return x_other

class TestTimeAugmentation:
    """Test-time augmentation for improved inference"""
    
    def __init__(self, num_augmentations=5, augmentation_strength=0.3):
        self.num_augmentations = num_augmentations
        self.augmenter = AdvancedAudioAugmenter(
            time_mask_prob=0.1 * augmentation_strength,
            freq_mask_prob=0.1 * augmentation_strength,  
            freq_shift_prob=0.2 * augmentation_strength,
            noise_prob=0.05 * augmentation_strength,
            mixup_prob=0.0,  # No mixup at test time
            spec_augment_prob=0.1 * augmentation_strength
        )
    
    def __call__(self, x):
        """Apply test-time augmentation"""
        augmented_samples = [x]  # Include original
        
        for _ in range(self.num_augmentations):
            aug_x = self.augmenter(x.clone(), training=True)
            augmented_samples.append(aug_x)
        
        return torch.stack(augmented_samples)

if __name__ == "__main__":
    print("Testing Enhanced Augmentations...")
    
    # Test augmenter
    augmenter = AdvancedAudioAugmenter()
    x = torch.randn(1, 128, 501)  # [channels, mel_bins, time]
    
    print(f"Original shape: {x.shape}")
    
    # Test different augmentations
    x_aug = augmenter(x, training=True)
    print(f"Augmented shape: {x_aug.shape}")
    
    # Test dataset
    # dataset = ImprovedDiarizationDataset("cache_mels/", training=True)
    # print(f"Dataset length: {len(dataset)}")
    
    # Test TTA
    tta = TestTimeAugmentation(num_augmentations=3)
    x_tta = tta(x)
    print(f"TTA output shape: {x_tta.shape}")
    
    print("✅ Enhanced augmentations working correctly!")