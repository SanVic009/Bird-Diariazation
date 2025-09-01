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
        duration_strategy: str = "adaptive",  # "fixed", "adaptive", or "segments"
        min_duration: float = 3.0,
        max_duration: float = 10.0,
        fixed_duration: float = 5.0,
        segment_overlap: float = 0.5,
        out_dir: str = "processed",
        augment_prob: float = 0.7
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
        self.out_dir.mkdir(parents=True, exist_ok=True)

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

            if random.random() < 0.3:
                noise = np.random.randn(len(wav)) * 0.005
                wav = wav + noise

            if random.random() < 0.3:
                factor = random.uniform(0.5, 1.5)
                wav = wav * factor
        except Exception as e:
            print(f"[WARNING] Augmentation failed: {e}. Returning original waveform.")
            logging.warning(f"[WARNING] Augmentation failed: {e}. Returning original waveform.")
            return original_wav
        return wav

    # SpecAugment (time/freq masking on spectrogram)
    def spec_augment(self, spec: np.ndarray):
        original_spec = spec.copy()
        try:
            spec = spec.copy()
            n_mels, t = spec.shape

            # freq mask
            if random.random() < 0.3:
                f = random.randint(0, n_mels // 8)
                f0 = random.randint(0, n_mels - f)
                spec[f0:f0 + f, :] = 0

            # time mask
            if random.random() < 0.3:
                t_mask = random.randint(0, t // 8)
                t0 = random.randint(0, t - t_mask)
                spec[:, t0:t0 + t_mask] = 0
        except Exception as e:
            print(f"[WARNING] SpecAugment failed: {e}. Returning original spectrogram.")
            logging.warning(f"[WARNING] SpecAugment failed: {e}. Returning original spectrogram.")
            return original_spec
        return spec

    # -------------------------------
    # Feature Extraction
    # -------------------------------
    def to_log_mel(self, wav: np.ndarray):
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
                log_mel = self.to_log_mel(wav_segment)

                if random.random() < self.augment_prob:
                    log_mel = self.spec_augment(log_mel)

                if log_mel.shape[0] == 0 or log_mel.shape[1] == 0:
                    raise ValueError(f"Empty spectrogram for {filepath}, segment {i}")

                # Unique hash filename with segment index
                h = hashlib.md5(filepath.encode()).hexdigest()[:10]
                if len(wav_segments) > 1:
                    out_path = self.out_dir / f"{label}_{h}_seg{i:02d}.npy"
                else:
                    out_path = self.out_dir / f"{label}_{h}.npy"
                    
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
            log_mel = self.spec_augment(log_mel)

        if log_mel.shape[0] == 0 or log_mel.shape[1] == 0:
            raise ValueError(f"Empty spectrogram for {filepath}")

        # Unique hash filename
        h = hashlib.md5(filepath.encode()).hexdigest()[:10]
        out_path = self.out_dir / f"{label}_{h}.npy"
        np.save(out_path, log_mel.astype(np.float32))
        return out_path
