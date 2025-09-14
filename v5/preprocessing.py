# preprocessing.py
import random
import hashlib
import numpy as np
import torchaudio
import librosa
import logging
from pathlib import Path

class BirdPreprocessor:
    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 512,
        fmin: int = 20,
        fmax: int = 16000,
        duration_strategy: str = "segments",  # "fixed", "adaptive", or "segments"
        min_duration: float = 3.0,
        max_duration: float = 10.0,
        fixed_duration: float = 5.0,
        segment_overlap: float = 0.5,
        out_dir: str = "processed_rfcx",
        augment_prob: float = 0.7,
        mixup_prob: float = 0.3,  # Probability of applying mixup
        mixup_alpha: float = 0.2,  # Beta distribution parameter for mixup
        # Enhanced noise parameters
        noise_snr_range: tuple = (15, 30),  # SNR range for Gaussian noise
        # Enhanced SpecAugment parameters
        freq_mask_num: int = 2,  # Number of frequency masks
        time_mask_num: int = 2,  # Number of time masks
        freq_mask_max: int = 20,  # Maximum frequency mask size
        time_mask_max: int = 30   # Maximum time mask size
    ):
        self.sr = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.duration_strategy = duration_strategy
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.fixed_duration = fixed_duration
        self.segment_overlap = segment_overlap
        self.out_dir = Path(out_dir)
        self.augment_prob = augment_prob
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        
        # Enhanced augmentation parameters
        self.noise_snr_range = noise_snr_range
        self.freq_mask_num = freq_mask_num
        self.time_mask_num = time_mask_num
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for MixUp - store processed audio samples
        self.mixup_cache = []
        self.mixup_cache_labels = []
        self.max_cache_size = 100

    # -------------------------------
    # Audio Loading
    # -------------------------------
    def load_audio(self, filepath: str):
        try:
            wav, sr = torchaudio.load(filepath)
            wav = wav.mean(dim=0)  # stereo → mono
            wav = torchaudio.functional.resample(wav, sr, self.sr)
            wav = wav.numpy()
        except (RuntimeError, FileNotFoundError) as e:
            print(f"[WARNING] torchaudio.load failed for {filepath}: {e}. Attempting with librosa.load...")
            logging.warning(f"[WARNING] torchaudio.load failed for {filepath}: {e}. Attempting with librosa.load...")
            try:
                wav, sr = librosa.load(filepath, sr=self.sr, mono=True)
            except Exception as e_librosa:
                raise IOError(f"Failed to load audio with both torchaudio and librosa for {filepath}: {e_librosa}") from e_librosa
        return wav

    def process_duration(self, wav: np.ndarray):
        """
        Process audio duration based on strategy:
        - fixed: Pad/trim to fixed duration
        - adaptive: Keep natural length within min/max bounds
        - segments: Split long audio into overlapping segments
        """
        wav_duration = len(wav) / self.sr
        
        if self.duration_strategy == "fixed":
            return [self._pad_or_trim_fixed(wav, self.fixed_duration)]
        
        elif self.duration_strategy == "adaptive":
            if wav_duration < self.min_duration:
                # Pad short audio to minimum duration
                return [self._pad_or_trim_fixed(wav, self.min_duration)]
            elif wav_duration > self.max_duration:
                # Trim long audio to maximum duration (take middle portion)
                return [self._extract_middle_segment(wav, self.max_duration)]
            else:
                # Keep natural length
                return [wav]
        
        elif self.duration_strategy == "segments":
            return self._split_into_segments(wav)
        
        else:
            raise ValueError(f"Unknown duration strategy: {self.duration_strategy}")
    
    def _pad_or_trim_fixed(self, wav: np.ndarray, target_duration: float):
        """Traditional fixed duration approach"""
        target_len = int(self.sr * target_duration)
        if len(wav) < target_len:
            wav = np.pad(wav, (0, target_len - len(wav)))
        else:
            wav = wav[:target_len]
        return wav
    
    def _extract_middle_segment(self, wav: np.ndarray, target_duration: float):
        """Extract middle segment to preserve main vocalization"""
        target_len = int(self.sr * target_duration)
        if len(wav) <= target_len:
            return wav
        
        # Find the center and extract around it
        center = len(wav) // 2
        start = max(0, center - target_len // 2)
        end = start + target_len
        
        # Adjust if we go beyond the end
        if end > len(wav):
            end = len(wav)
            start = end - target_len
            
        return wav[start:end]
    
    def _split_into_segments(self, wav: np.ndarray):
        """Split long audio into overlapping segments"""
        segment_len = int(self.sr * self.max_duration)
        step_len = int(segment_len * (1 - self.segment_overlap))
        
        if len(wav) <= segment_len:
            return [self._pad_or_trim_fixed(wav, self.max_duration)]
        
        segments = []
        start = 0
        while start < len(wav):
            end = min(start + segment_len, len(wav))
            segment = wav[start:end]
            
            # Pad last segment if it's too short
            if len(segment) < segment_len:
                segment = np.pad(segment, (0, segment_len - len(segment)))
            
            segments.append(segment)
            
            # Break if this was the last segment
            if end == len(wav):
                break
                
            start += step_len
        
        return segments

    # Legacy method for backward compatibility
    def pad_or_trim(self, wav: np.ndarray):
        """Legacy method - use process_duration instead"""
        return self._pad_or_trim_fixed(wav, self.fixed_duration)

    # -------------------------------
    # Augmentations
    # -------------------------------
    def mixup_augmentation(self, wav1: np.ndarray, wav2: np.ndarray, alpha: float = 0.2):
        """
        MixUp augmentation for audio signals.
        Mixes two audio samples with a random mixing ratio.
        
        Args:
            wav1, wav2: Audio arrays to mix
            alpha: Beta distribution parameter (lower = more extreme mixing)
            
        Returns:
            mixed_wav: Mixed audio
            lam: Mixing ratio (for label mixing later)
        """
        # Ensure both audio arrays have the same length
        min_len = min(len(wav1), len(wav2))
        wav1_trim = wav1[:min_len]
        wav2_trim = wav2[:min_len]
        
        # Sample mixing ratio from Beta distribution
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        
        # Mix the audio signals
        mixed_wav = lam * wav1_trim + (1 - lam) * wav2_trim
        
        return mixed_wav, lam
    
    def _add_to_mixup_cache(self, wav: np.ndarray, label: str):
        """Add audio sample to MixUp cache"""
        if len(self.mixup_cache) >= self.max_cache_size:
            # Remove oldest entry
            self.mixup_cache.pop(0)
            self.mixup_cache_labels.pop(0)
        
        self.mixup_cache.append(wav.copy())
        self.mixup_cache_labels.append(label)
    
    def _get_mixup_sample(self, current_label: str):
        """Get a random sample from cache for MixUp, preferably different species"""
        if len(self.mixup_cache) == 0:
            return None, None
        
        # Try to find a different species first
        different_species = [(wav, label) for wav, label in zip(self.mixup_cache, self.mixup_cache_labels) 
                           if label != current_label]
        
        if different_species:
            wav, label = random.choice(different_species)
        else:
            # Fall back to any sample
            idx = random.randint(0, len(self.mixup_cache) - 1)
            wav, label = self.mixup_cache[idx], self.mixup_cache_labels[idx]
        
        return wav, label
    
    def add_gaussian_noise(self, wav: np.ndarray, snr_range=(15, 30)):
        """
        Add Gaussian noise with controlled Signal-to-Noise Ratio (SNR).
        
        Args:
            wav: Input audio signal
            snr_range: Range of SNR values in dB (higher = less noise)
            
        Returns:
            Audio with added noise
        """
        # Calculate signal power
        signal_power = np.mean(wav ** 2)
        
        # Skip if signal is too weak
        if signal_power < 1e-10:
            return wav
        
        # Random SNR from range
        snr_db = random.uniform(*snr_range)
        
        # Calculate noise power from SNR
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate and add noise
        noise = np.random.normal(0, np.sqrt(noise_power), len(wav))
        return wav + noise

    def augment(self, wav: np.ndarray):
        original_wav = wav.copy()
        try:
            if random.random() < 0.3:
                shift = int(0.1 * len(wav))  # up to 10% shift
                # wav = np.roll(wav, random.randint(-shift, shift))

            if random.random() < 0.3:
                rate = random.uniform(0.8, 1.25)
                wav = librosa.effects.time_stretch(y=wav, rate=rate)

            if random.random() < 0.3:
                steps = random.uniform(-2, 2)  # semitones
                wav = librosa.effects.pitch_shift(wav, sr=self.sr, n_steps=steps)

            # Enhanced Gaussian noise with proper SNR control
            if random.random() < 0.4:  # Increased probability
                wav = self.add_gaussian_noise(wav, snr_range=self.noise_snr_range)

            if random.random() < 0.3:
                factor = random.uniform(0.5, 1.5)
                wav = wav * factor
                
        except Exception as e:
            print(f"[WARNING] Augmentation failed: {e}. Returning original waveform.")
            logging.warning(f"[WARNING] Augmentation failed: {e}. Returning original waveform.")
            return original_wav
        return wav

    # Advanced SpecAugment (time/freq masking on spectrogram)
    def spec_augment(self, spec: np.ndarray, freq_mask_num=2, time_mask_num=2, 
                     freq_mask_max=20, time_mask_max=30, mask_value=None):
        """
        Advanced SpecAugment with multiple masks and better masking strategy.
        
        Args:
            spec: Spectrogram (n_mels, time)
            freq_mask_num: Number of frequency masks to apply
            time_mask_num: Number of time masks to apply  
            freq_mask_max: Maximum frequency mask size
            time_mask_max: Maximum time mask size
            mask_value: Value to use for masking (None = use mean)
        """
        original_spec = spec.copy()
        try:
            spec = spec.copy()
            n_mels, time_steps = spec.shape
            
            # Use mean value for masking (more realistic than zero)
            if mask_value is None:
                mask_value = spec.mean()

            # Apply multiple frequency masks
            for _ in range(freq_mask_num):
                if random.random() < 0.5:  # 50% chance for each mask
                    f_mask_size = random.randint(1, min(freq_mask_max, n_mels // 4))
                    f_start = random.randint(0, n_mels - f_mask_size)
                    spec[f_start:f_start + f_mask_size, :] = mask_value

            # Apply multiple time masks
            for _ in range(time_mask_num):
                if random.random() < 0.5:  # 50% chance for each mask
                    t_mask_size = random.randint(1, min(time_mask_max, time_steps // 4))
                    t_start = random.randint(0, time_steps - t_mask_size)
                    spec[:, t_start:t_start + t_mask_size] = mask_value
                    
            # Additional advanced augmentations
            
            # Random frequency band emphasis/suppression
            if random.random() < 0.3:
                band_start = random.randint(0, n_mels // 2)
                band_end = random.randint(band_start + 1, min(band_start + 20, n_mels))
                emphasis_factor = random.uniform(0.5, 1.5)
                spec[band_start:band_end, :] *= emphasis_factor
            
            # Random time segment emphasis/suppression  
            if random.random() < 0.3:
                seg_start = random.randint(0, time_steps // 2)
                seg_end = random.randint(seg_start + 1, min(seg_start + 30, time_steps))
                emphasis_factor = random.uniform(0.7, 1.3)
                spec[:, seg_start:seg_end] *= emphasis_factor
                
        except Exception as e:
            print(f"[WARNING] Advanced SpecAugment failed: {e}. Returning original spectrogram.")
            logging.warning(f"[WARNING] Advanced SpecAugment failed: {e}. Returning original spectrogram.")
            return original_spec
        return spec

    # -------------------------------
    # Feature Extraction
    # -------------------------------
    def to_log_mel(self, wav: np.ndarray, normalize_freq=True):
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # Frequency axis normalization (per frequency bin)
        if normalize_freq:
            log_mel = (log_mel - log_mel.mean(axis=1, keepdims=True)) / (log_mel.std(axis=1, keepdims=True) + 1e-8)
        
        return log_mel

    # -------------------------------
    # Full Pipeline
    # -------------------------------
    def process_and_save(self, filepath: str, label: str):
        """
        Process audio file and save spectrogram(s).
        Returns list of output paths (multiple if segmentation is used).
        """
        try:
            wav = self.load_audio(filepath)
        except IOError as e:
            print(f"[ERROR] Failed to load audio for {filepath}: {e}")
            logging.error(f"[ERROR] Failed to load audio for {filepath}: {e}")
            return [] # Return empty list if audio loading fails

        if random.random() < self.augment_prob:
            wav = self.augment(wav)

        # Process duration based on strategy (may return multiple segments)
        wav_segments = self.process_duration(wav)
        
        output_paths = []
        
        for i, wav_segment in enumerate(wav_segments):
            try:
                # Add current segment to MixUp cache for future use
                self._add_to_mixup_cache(wav_segment, label)
                
                # Apply MixUp augmentation with probability
                mixed_label = label  # Default label
                if random.random() < self.mixup_prob:
                    mixup_wav, mixup_label = self._get_mixup_sample(label)
                    if mixup_wav is not None:
                        wav_segment, mix_ratio = self.mixup_augmentation(wav_segment, mixup_wav, self.mixup_alpha)
                        # Create mixed label (for now, keep dominant label)
                        # In future, you can return mixing info for label combination
                        if mix_ratio < 0.5:
                            mixed_label = f"{label}_mixup_{mixup_label}"
                        else:
                            mixed_label = f"{label}_mixup_{mixup_label}"
                
                log_mel = self.to_log_mel(wav_segment)

                if random.random() < self.augment_prob:
                    log_mel = self.spec_augment(
                        log_mel,
                        freq_mask_num=self.freq_mask_num,
                        time_mask_num=self.time_mask_num,
                        freq_mask_max=self.freq_mask_max,
                        time_mask_max=self.time_mask_max
                    )

                if log_mel.shape[0] == 0 or log_mel.shape[1] == 0:
                    raise ValueError(f"Empty spectrogram for {filepath}, segment {i}")

                # Unique hash filename with segment index
                h = hashlib.md5(filepath.encode()).hexdigest()[:10]
                if len(wav_segments) > 1:
                    out_path = self.out_dir / f"{mixed_label}_{h}_seg{i:02d}.npy"
                else:
                    out_path = self.out_dir / f"{mixed_label}_{h}.npy"
                    
                np.save(out_path, log_mel.astype(np.float32))
                output_paths.append(out_path)
            except Exception as e:
                print(f"[ERROR] Failed to process segment {i} from {filepath}: {e}")
                logging.error(f"[ERROR] Failed to process segment {i} from {filepath}: {e}")
                continue # Continue to next segment if one fails
        
        return output_paths

    def process_and_save_legacy(self, filepath: str, label: str):
        """Legacy method for backward compatibility"""
        wav = self.load_audio(filepath)
        if random.random() < self.augment_prob:
            wav = self.augment(wav)

        wav = self.pad_or_trim(wav)  # Uses fixed duration

        log_mel = self.to_log_mel(wav)

        if random.random() < self.augment_prob:
            log_mel = self.spec_augment(
                log_mel,
                freq_mask_num=self.freq_mask_num,
                time_mask_num=self.time_mask_num,
                freq_mask_max=self.freq_mask_max,
                time_mask_max=self.time_mask_max
            )

        if log_mel.shape[0] == 0 or log_mel.shape[1] == 0:
            raise ValueError(f"Empty spectrogram for {filepath}")

        # Unique hash filename
        h = hashlib.md5(filepath.encode()).hexdigest()[:10]
        out_path = self.out_dir / f"{label}_{h}.npy"
        np.save(out_path, log_mel.astype(np.float32))
        return out_path
