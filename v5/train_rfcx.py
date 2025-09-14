# train_rfcx.py
import random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# -----------------------------
# Config
# -----------------------------
class CFG:
    data_dir = "rfcx-species-audio-detection"                    # path where you unzipped
    train_audio_dir = "rfcx-species-audio-detection/train"       # FLAC files live here
    train_tp_csv = "rfcx-species-audio-detection/train_tp.csv"
    train_fp_csv = "rfcx-species-audio-detection/train_fp.csv"   # optional hard negatives

    # audio / features
    target_sr = 32000
    win_length = 1024            # ~32 ms @32k
    hop_length = 320             # 10 ms @32k
    n_mels = 128
    fmin = 300                   # band-pass (optional)
    fmax = 12000

    # segmentation
    seg_seconds = 2.0
    seg_hop_seconds = 1.0
    iou_pos = 0.5
    min_event_len = 0.25

    # training
    num_epochs = 20
    batch_size = 24
    lr = 1e-3
    weight_decay = 1e-4
    num_workers = 4
    seed = 42
    valid_frac = 0.2

    # thresholds
    pred_thresh = 0.5
    merge_gap = 0.3

random.seed(CFG.seed); np.random.seed(CFG.seed); torch.manual_seed(CFG.seed)

# -----------------------------
# Utilities
# -----------------------------
def load_audio(path, target_sr=32000):
    try:
        y, sr = sf.read(path, always_2d=False)
    except Exception as e:
        print(f"Error loading audio file: {path}")
        raise e
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # light high-pass via pre-emphasis
    y = librosa.effects.preemphasis(y, coef=0.97)
    # normalize
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32), target_sr

def melspec(y, sr):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=CFG.win_length*2, hop_length=CFG.hop_length,
        win_length=CFG.win_length, n_mels=CFG.n_mels, fmin=CFG.fmin, fmax=CFG.fmax, power=2.0
    )
    S = librosa.power_to_db(S, ref=np.max)
    # standardize
    S = (S - S.mean()) / (S.std() + 1e-6)
    return S.astype(np.float32)

def time_to_frame(t, sr):
    hop = CFG.hop_length / sr
    return int(t / hop)

def overlap_iou(a, b):
    # a=(t0,t1), b=(t0,t1) in seconds
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / (union + 1e-9)

# -----------------------------
# Prepare label tables
# -----------------------------
tp = pd.read_csv(CFG.train_tp_csv)
# columns commonly seen: recording_id, species_id, songtype_id, t_min, t_max, f_min, f_max
required_cols = {"recording_id","species_id","t_min","t_max"}
missing = required_cols - set(tp.columns)
if missing:
    raise RuntimeError(f"train_tp.csv missing columns: {missing}")

species_ids = sorted(tp["species_id"].unique().tolist())
sid2idx = {sid:i for i, sid in enumerate(species_ids)}
idx2sid = {i:sid for sid, i in sid2idx.items()}
num_classes = len(species_ids)

# build per-recording event dict
events_by_rec = {}
for _, r in tp.iterrows():
    rid = r["recording_id"]
    events_by_rec.setdefault(rid, []).append((float(r["t_min"]), float(r["t_max"]), int(r["species_id"])))

# optional: list of audio files with labels
audio_files = sorted([p.name for p in Path(CFG.train_audio_dir).glob("*.flac")])
labeled = [f for f in audio_files if Path(CFG.train_audio_dir, f).stem in events_by_rec]

# simple split by recording_id
recs = list(events_by_rec.keys())
random.shuffle(recs)
cut = int(len(recs)*(1.0-CFG.valid_frac))
train_recs, valid_recs = set(recs[:cut]), set(recs[cut:])

# -----------------------------
# Dataset
# -----------------------------
class RFCXDataset(Dataset):
    def __init__(self, rec_ids, augment=True):
        self.rec_ids = list(rec_ids)
        self.augment = augment
        self.index = []  # (audio_path, seg_start, seg_end)
        for rid in self.rec_ids:
            path = Path(CFG.train_audio_dir, f"{rid}.flac")
            # index segments
            # assume many files are ~60s; robust to others
            # We'll measure length lazily when loading
            self.index.append((str(path), rid))

    def __len__(self):
        return len(self.index)

    def window_targets(self, rid, seg_start, seg_end):
        y = np.zeros(num_classes, dtype=np.float32)
        for (t0, t1, sid) in events_by_rec.get(rid, []):
            if overlap_iou((seg_start, seg_end), (t0, t1)) >= CFG.iou_pos:
                y[sid2idx[sid]] = 1.0
        return y

    def spec_augment(self, spec):
        # simple SpecAugment: 1 freq mask, 1 time mask
        if random.random() < 0.8:
            # freq mask
            Fm = random.randint(0, CFG.n_mels//8)
            f0 = random.randint(0, CFG.n_mels - Fm)
            spec[f0:f0+Fm, :] = 0
        if random.random() < 0.8:
            # time mask
            T = spec.shape[1]
            Tm = random.randint(0, T//10)
            t0 = random.randint(0, T - Tm) if Tm>0 else 0
            spec[:, t0:t0+Tm] = 0
        return spec

    def __getitem__(self, i):
        path, rid = self.index[i]
        y_raw, sr = load_audio(path, CFG.target_sr)
        dur = len(y_raw)/sr

        # pick a random segment during training; full coverage during validation handled elsewhere
        if self.augment:
            max_start = max(0.0, dur - CFG.seg_seconds)
            seg_start = random.uniform(0.0, max_start) if max_start>0 else 0.0
        else:
            seg_start = 0.0
        seg_end = min(dur, seg_start + CFG.seg_seconds)
        s0 = int(seg_start*sr); s1 = int(seg_end*sr)
        y_seg = y_raw[s0:s1]

        # pad if short
        need = int(CFG.seg_seconds*sr) - len(y_seg)
        if need > 0:
            y_seg = np.pad(y_seg, (0, need))

        spec = melspec(y_seg, sr)
        if self.augment:
            if random.random() < 0.5:
                # small time shift (circular)
                shift = random.randint(0, spec.shape[1]-1)
                spec = np.roll(spec, shift, axis=1)
            spec = self.spec_augment(spec)

        # 3-channel: add deltas
        d1 = librosa.feature.delta(spec)
        d2 = librosa.feature.delta(spec, order=2)
        x = np.stack([spec, d1, d2], axis=0)  # (3, n_mels, T)

        y = self.window_targets(rid, seg_start, seg_end)
        return torch.tensor(x), torch.tensor(y), rid, torch.tensor([seg_start, seg_end], dtype=torch.float32)

# -----------------------------
# Model: simple CRNN
# -----------------------------
class CRNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2,2)),
        )
        self.dropout = nn.Dropout(0.3)
        # time dimension is axis=3 after conv; we’ll permute and BiGRU over time
        self.gru = nn.GRU(input_size=128*(CFG.n_mels//8), hidden_size=256, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        # x: (B, 3, n_mels, T)
        h = self.conv(x)  # (B, C=128, n_mels/8, T/8)
        B, C, Fm, Tm = h.shape
        h = self.dropout(h)
        h = h.permute(0, 3, 1, 2).contiguous()  # (B, Tm, C, Fm)
        h = h.view(B, Tm, C*Fm)                 # (B, Tm, feat)
        h, _ = self.gru(h)                      # (B, Tm, 512)
        # frame-wise logits -> mean pool over time within segment
        logits = self.fc(h)                     # (B, Tm, n_classes)
        logits = logits.mean(dim=1)             # (B, n_classes)
        return logits

# -----------------------------
# Train / Validate
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CRNN(num_classes).to(device)
opt = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.num_epochs)
criterion = nn.BCEWithLogitsLoss()

train_ds = RFCXDataset([r for r in train_recs], augment=True)
valid_ds = RFCXDataset([r for r in valid_recs], augment=False)

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, drop_last=True)
valid_loader = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

def evaluate():
    model.eval()
    losses, all_t, all_p = [], [], []
    with torch.no_grad():
        for xb, yb, _, _ in valid_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
            probs = torch.sigmoid(logits).cpu().numpy()
            all_p.append(probs); all_t.append(yb.cpu().numpy())
    P = np.vstack(all_p); T = np.vstack(all_t)
    # simple micro-F1 at threshold 0.5 (for sanity)
    preds = (P >= 0.5).astype(int)
    tp = (preds*T).sum(); fp = (preds*(1-T)).sum(); fn = ((1-preds)*T).sum()
    f1 = (2*tp)/(2*tp+fp+fn+1e-9)
    return float(np.mean(losses)), float(f1)

best = {"f1": -1, "state": None}
for epoch in range(1, CFG.num_epochs+1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG.num_epochs}")
    for xb, yb, _, _ in pbar:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        pbar.set_postfix(loss=float(loss.item()))
    vl, vf1 = evaluate(); sched.step()
    print(f"[Valid] loss={vl:.4f} microF1@0.5={vf1:.4f}")
    if vf1 > best["f1"]:
        best["f1"] = vf1
        best["state"] = model.state_dict()
        cfg_to_save = {k: v for k, v in CFG.__dict__.items() if not k.startswith('__') and not callable(v)}
        torch.save({"model":best["state"], "sid2idx":sid2idx, "cfg":cfg_to_save}, "rfcx_crnn_best.pt")
        print("Saved rfcx_crnn_best.pt")

print("Best F1:", best["f1"])
