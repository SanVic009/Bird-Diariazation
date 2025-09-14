# infer_rfcx.py
import json, math
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn

# ---- same CFG values used in training (kept minimal here) ----
class CFG:
    target_sr = 32000
    win_length = 1024
    hop_length = 320
    n_mels = 128
    fmin = 300
    fmax = 12000
    seg_seconds = 2.0
    seg_hop_seconds = 1.0
    pred_thresh = 0.5
    merge_gap = 0.3
    min_event_len = 0.25

def load_audio(path, target_sr=32000):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1: y = np.mean(y, axis=1)
    if sr != target_sr: y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    y = librosa.effects.preemphasis(y, coef=0.97)
    if np.max(np.abs(y))>0: y = y/np.max(np.abs(y))
    return y.astype(np.float32), target_sr

def melspec(y, sr):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=CFG.win_length*2, hop_length=CFG.hop_length,
        win_length=CFG.win_length, n_mels=CFG.n_mels, fmin=CFG.fmin, fmax=CFG.fmax, power=2.0
    )
    S = librosa.power_to_db(S, ref=np.max)
    S = (S - S.mean())/(S.std()+1e-6)
    d1 = librosa.feature.delta(S)
    d2 = librosa.feature.delta(S, order=2)
    X = np.stack([S, d1, d2], axis=0)  # (3, mels, T)
    return X.astype(np.float32)

class CRNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2,2)),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2,2)),
        )
        self.dropout = nn.Dropout(0.3)
        self.gru = nn.GRU(input_size=128*(CFG.n_mels//8), hidden_size=256, num_layers=1,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, 0)  # placeholder; set later

    def set_classes(self, n):
        self.fc = nn.Linear(512, n)

    def forward(self, x):
        h = self.conv(x)
        B, C, Fm, Tm = h.shape
        h = self.dropout(h)
        h = h.permute(0,3,1,2).contiguous().view(B, Tm, C*Fm)
        h, _ = self.gru(h)
        return self.fc(h).mean(dim=1)

def merge_events(times, probs, thresh, hop, min_len=0.25, merge_gap=0.3):
    """times: centers for windows; probs: (N_windows,), binary after thresh"""
    # convert window centers to start/end spans of length seg_seconds
    seg = CFG.seg_seconds
    spans = [(t - seg/2, t + seg/2) for t in times]
    # filter positives
    pos_spans = [s for s, p in zip(spans, probs>=thresh) if p]
    if not pos_spans: return []
    # merge touching/close spans
    pos_spans.sort()
    merged = [list(pos_spans[0])]
    for s,e in pos_spans[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s,e])
    # remove very short
    merged = [(max(0.0, s), max(0.0,e)) for s,e in merged if (e-s)>=min_len]
    return merged

def infer(audio_path, ckpt_path="rfcx_crnn_best.pt", label_map_path=None, nice_names=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt_path, map_location=device)
    sid2idx = state["sid2idx"]; idx2sid = {i:s for s,i in sid2idx.items()}
    model = CRNN(n_classes=len(sid2idx)); model.set_classes(len(sid2idx))
    model.load_state_dict(state["model"]); model.to(device); model.eval()

    y, sr = load_audio(audio_path, CFG.target_sr)
    dur = len(y)/sr

    centers = []
    Xb = []
    # slide windows
    start = 0.0
    while start < dur:
        s0 = int(start*sr)
        s1 = int(min(dur, start+CFG.seg_seconds)*sr)
        seg = y[s0:s1]
        need = int(CFG.seg_seconds*sr)-len(seg)
        if need>0: seg = np.pad(seg, (0,need))
        spec = melspec(seg, sr)
        Xb.append(spec)
        centers.append(start + CFG.seg_seconds/2.0)
        start += CFG.seg_hop_seconds

    X = torch.tensor(np.stack(Xb,0)).to(device)
    with torch.no_grad():
        logits = model(X)
        P = torch.sigmoid(logits).cpu().numpy()  # (Nwin, C)

    species_events = {}
    for ci in range(P.shape[1]):
        probs = P[:,ci]
        ev = merge_events(np.array(centers), probs, CFG.pred_thresh, CFG.seg_hop_seconds,
                          min_len=CFG.min_event_len, merge_gap=CFG.merge_gap)
        if ev:
            species_events[ci] = ev

    # Pretty names
    def label_for(ci):
        sid = idx2sid[ci]
        if nice_names and sid in nice_names:
            return nice_names[sid]
        return f"species_{sid}"

    # Print results to console like the sample
    printed = 0
    for ci, spans in species_events.items():
        name = label_for(ci)
        for (s,e) in spans:
            print(f"{name}:{round(s,2)} to {round(e,2)}s")
            printed += 1
    unique_species = len(species_events)
    print(f"total species: {unique_species}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="path to .flac/.wav")
    ap.add_argument("--ckpt", default="rfcx_crnn_best.pt")
    ap.add_argument("--names", default=None, help="JSON mapping {species_id: 'frog1', ...}")
    args = ap.parse_args()

    nice_names = None
    if args.names and Path(args.names).exists():
        with open(args.names, "r") as f:
            nice_names = json.load(f)

    infer(args.audio, args.ckpt, nice_names=nice_names)
