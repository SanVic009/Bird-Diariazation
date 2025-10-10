# train.py
from config import *
from augment import get_augmenter
from dataset import BirdCLEFUnsupervised
from model import CNNEncoder
from loss import nt_xent
import torch, tqdm, os
from torch.utils.data import DataLoader

def train():
    augmenter = get_augmenter(SR) if AUGMENT else None
    ds = BirdCLEFUnsupervised("data", dur=DUR, sr=SR, augment=augmenter)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    encoder = CNNEncoder(EMBED_DIM, N_MELS).to(DEVICE)
    opt = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(EPOCHS):
        losses = []
        for x in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}"):
            x = x.to(DEVICE, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                if augmenter:
                    x1, x2 = augmenter(x), augmenter(x)  # two views with augmentation
                else:
                    x1, x2 = x, x  # two identical views without augmentation
                z1, z2 = encoder(x1), encoder(x2)
                loss = nt_xent(z1, z2)
            
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        print(f"Epoch {epoch+1} | loss {torch.tensor(losses).mean():.4f}")
    os.makedirs("ckpt", exist_ok=True)
    torch.save(encoder.state_dict(), "ckpt/birdclef24_encoder.pt")

if __name__ == "__main__":
    train()
