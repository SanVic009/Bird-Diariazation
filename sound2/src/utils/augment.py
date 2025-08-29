import numpy as np, random

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
