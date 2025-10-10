import torch
# config.py
SR          = 32_000
DUR         = 2.0           # s
N_MELS      = 64
EMBED_DIM   = 512
BATCH_SIZE  = 256
EPOCHS      = 30
LR          = 3e-4
NUM_WORKERS = 4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AUGMENT     = False
K_MAX       = 8             # max birds to consider (auto-K search 1..K_MAX)