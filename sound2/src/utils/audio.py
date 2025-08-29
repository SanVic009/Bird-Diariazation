import torch, torchaudio, librosa, numpy as np

def load_audio(path, target_sr=32000):
    # Try torchaudio -> fallback to librosa
    try:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)  # mono
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.numpy(), target_sr
    except Exception:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        return y, sr

def mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    fmin: int = 50,
    fmax: int = 14000,
    frame_sec: float = 1.0,
    hop_sec: float = 0.5
):
    n_fft = int(sr * frame_sec)
    hop_length = int(sr * hop_sec)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    # Shape: [n_mels, T]
    return S_db.astype(np.float32), hop_length, n_fft

def window_indices(total_frames: int, win: int, hop: int):
    # Given mel time frames, compute frame windows; here we align mel frames directly.
    # For per-frame classification, we'll treat each mel column as a "frame".
    idxs = [(i, i+1) for i in range(total_frames)]
    return idxs
