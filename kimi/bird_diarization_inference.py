# bird_diarization_inference.py
import torch, librosa, numpy as np
from sklearn.cluster import KMeans
from config import *
from model import CNNEncoder
import matplotlib.pyplot as plt

def identify_bird_segments(audio_file, output_plot=True):
    """
    Identify different bird segments in audio (but not species names)
    """
    # Load trained encoder
    encoder = CNNEncoder(EMBED_DIM, N_MELS).to(DEVICE)
    encoder.load_state_dict(torch.load("ckpt/birdclef24_encoder.pt", map_location=DEVICE))
    encoder.eval()
    
    # Load audio
    y, _ = librosa.load(audio_file, sr=SR, mono=True)
    
    # Segment audio and extract embeddings
    win = int(DUR * SR)
    hop = int(0.5 * SR)  # 0.5 second hop
    embeds = []
    times = []
    
    with torch.no_grad():
        for start in range(0, len(y) - win, hop):
            chunk = y[start:start + win]
            chunk = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            embedding = encoder(chunk).cpu().numpy()
            embeds.append(embedding.flatten())
            times.append(start / SR)
    
    embeds = np.array(embeds)
    
    # Cluster to find different birds
    n_birds = min(5, len(embeds))  # Max 5 different birds
    kmeans = KMeans(n_clusters=n_birds, random_state=42)
    labels = kmeans.fit_predict(embeds)
    
    # Plot results
    if output_plot:
        plt.figure(figsize=(12, 6))
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (time, label) in enumerate(zip(times, labels)):
            plt.scatter(time, 0, c=colors[label % 5], s=50, alpha=0.7)
        plt.xlabel('Time (seconds)')
        plt.title('Bird Diarization - Different Colors = Different Birds')
        plt.yticks([])
        plt.show()
    
    # Return segments
    segments = []
    for i, (time, label) in enumerate(zip(times, labels)):
        segments.append({
            'start_time': time,
            'end_time': time + 0.5,
            'bird_id': f'Bird_{label}',
            'embedding': embeds[i]
        })
    
    return segments

# Usage example:
if __name__ == "__main__":
    segments = identify_bird_segments("your_audio_file.wav")
    print(f"Found {len(set([s['bird_id'] for s in segments]))} different birds")
    for seg in segments[:5]:  # Show first 5
        print(f"Time {seg['start_time']:.1f}s: {seg['bird_id']}")