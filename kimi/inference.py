# inference.py
import torch, librosa, numpy as np
from sklearn.cluster import KMeans
from config import *
from model import CNNEncoder

def segment_and_embed(file, hop=0.5):
    y, _ = librosa.load(file, sr=SR, mono=True)
    win = int(DUR*SR); hop = int(hop*SR)
    encoder = CNNEncoder(EMBED_DIM).to(DEVICE)
    encoder.load_state_dict(torch.load("ckpt/birdclef24_encoder.pt", map_location=DEVICE))
    encoder.eval()
    embeds = []
    with torch.no_grad():
        for start in range(0, len(y)-win, hop):
            chunk = torch.tensor(y[start:start+win], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            z = encoder(chunk).cpu().numpy()
            embeds.append(z)
    return np.vstack(embeds)   # (N, embed_dim)

def auto_kmeans(X, k_max=K_MAX):
    inertias = [KMeans(n_clusters=k, n_init=10).fit(X).inertia_ for k in range(1, k_max+1)]
    # simple knee at 90 % inertia drop
    knee = np.argmax(np.diff(inertias, 2)) + 2
    return KMeans(n_clusters=knee, n_init=10).fit_predict(X)

def to_rttm(labels, out_file, hop=0.5):
    with open(out_file, "w") as f:
        for i, lab in enumerate(labels):
            start = i * hop
            f.write(f"SPEAKER soundscape 1 {start:.2f} {hop:.2f} <NA> <NA> Bird{lab} <NA>\n")

if __name__ == "__main__":
    import glob, os, tqdm
    os.makedirs("rttm", exist_ok=True)
    for wav in tqdm.tqdm(glob.glob("data/test_soundscapes/*.ogg")):
        embeds = segment_and_embed(wav)
        labels = auto_kmeans(embeds)
        out = os.path.join("rttm", os.path.basename(wav).replace(".ogg", ".rttm"))
        to_rttm(labels, out)