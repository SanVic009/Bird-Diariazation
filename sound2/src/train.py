import os, json, argparse, yaml, math
import torch, numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from preprocessed_dataset import PreprocessedBirdDataset
from models import MelCNN

def bce_logits_loss(logits, targets):
    # logits: [B,T,C], targets: [B,T,C]
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

def set_seed(seed):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    os.makedirs(cfg["outputs"]["checkpoints"], exist_ok=True)

    train_ds = PreprocessedBirdDataset(
        os.path.join(cfg["data"]["splits_dir"], "train.csv"),
        os.path.join(cfg["outputs"]["preprocessed"], "train"),
        cfg["data"]["labels_json"],
        mix_prob=cfg["train"]["mix_prob"],
        mix_snr_db=cfg["train"]["mix_snr_db"],
        train=True
    )
    val_ds = PreprocessedBirdDataset(
        os.path.join(cfg["data"]["splits_dir"], "val.csv"),
        os.path.join(cfg["outputs"]["preprocessed"], "val"),
        cfg["data"]["labels_json"],
        mix_prob=0.0,
        mix_snr_db=cfg["train"]["mix_snr_db"],
        train=False
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])

    num_classes = train_ds.num_classes
    model = MelCNN(n_mels=cfg["audio"]["n_mels"], n_classes=num_classes, width=cfg["model"]["cnn_width"], dropout=cfg["model"]["dropout"]).to(device)
    opt = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    best_val = 1e9
    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        running = 0.0
        for M, tgt in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            M, tgt = M.to(device), tgt.to(device)
            logits = model(M)        # [B,T,C]
            loss = bce_logits_loss(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        avg_train = running / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for M, tgt in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                M, tgt = M.to(device), tgt.to(device)
                logits = model(M)
                loss = bce_logits_loss(logits, tgt)
                val_loss += loss.item()
        avg_val = val_loss / max(1, len(val_loader))

        print(f"Epoch {epoch}: train {avg_train:.4f} | val {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            ckpt = os.path.join(cfg["outputs"]["checkpoints"], f"best.pt")
            torch.save({"model": model.state_dict(), "num_classes": num_classes}, ckpt)
            print(f"Saved best checkpoint to {ckpt}")
    print("Done.")

if __name__ == "__main__":
    main()
