import os, json, argparse, yaml, torch, numpy as np, pandas as pd
from .utils.audio import load_audio, mel_spectrogram
from .models import MelCNN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    labels_meta = json.load(open(cfg["data"]["labels_json"], "r"))
    labels = labels_meta["labels"]
    num_classes = len(labels)

    ckpt_path = args.checkpoint or os.path.join(cfg["outputs"]["checkpoints"], "best.pt")
    state = torch.load(ckpt_path, map_location=device)
    model = MelCNN(n_mels=cfg["audio"]["n_mels"], n_classes=num_classes, width=cfg["model"]["cnn_width"], dropout=cfg["model"]["dropout"])
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    y, sr = load_audio(args.audio, cfg["audio"]["sample_rate"])
    M, hop_length, n_fft = mel_spectrogram(
        y, sr, n_mels=cfg["audio"]["n_mels"], fmin=cfg["audio"]["fmin"], fmax=cfg["audio"]["fmax"], 
        frame_sec=cfg["audio"]["frame_sec"], hop_sec=cfg["audio"]["hop_sec"]
    )  # [n_mels, T]
    M = (M - M.mean()) / (M.std()+1e-6)

    with torch.no_grad():
        M_t = torch.tensor(M).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(M_t)  # [1, n_mels, T]
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [T, C]

    # Compute timestamps for each frame center
    hop_sec = cfg["audio"]["hop_sec"]
    frame_times = np.arange(probs.shape[0]) * hop_sec

    # Save probabilities CSV
    base = os.path.splitext(os.path.basename(args.audio))[0].replace(os.sep, "_")
    out_csv = os.path.join(args.out, f"probs_{base}.csv")
    df = pd.DataFrame(probs, columns=labels)
    df.insert(0, "time_s", frame_times)
    df.to_csv(out_csv, index=False)
    print(f"Wrote frame probs to {out_csv}")

if __name__ == "__main__":
    main()
