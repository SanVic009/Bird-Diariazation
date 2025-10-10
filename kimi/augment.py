# augment.py
import torch
from torch_audiomentations import (
    AddColoredNoise, PitchShift, Compose
)

def get_augmenter(sr: int, p: float = 0.5):
    return Compose([
        AddColoredNoise(
            min_snr_in_db=3.0, max_snr_in_db=15.0, p=p, output_type='tensor')
    ], output_type='tensor')