# similarity_inference.py
import torch
import librosa
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from config import *
from model import CNNEncoder

def build_species_database():
    """
    Build a database of species embeddings from training data
    """
    encoder = CNNEncoder(EMBED_DIM, N_MELS).to(DEVICE)
    encoder.load_state_dict(torch.load("ckpt/birdclef24_encoder.pt", map_location=DEVICE))
    encoder.eval()
    
    species_embeddings = {}
    
    print("Building species embedding database...")
    species_folders = sorted(os.listdir("data"))
    
    for species in species_folders[:20]:  # Process first 20 species
        species_path = f"data/{species}"
        audio_files = [f for f in os.listdir(species_path) if f.endswith('.ogg')][:3]  # 3 files per species
        
        embeddings = []
        for audio_file in audio_files:
            try:
                audio_path = os.path.join(species_path, audio_file)
                y, _ = librosa.load(audio_path, sr=SR, mono=True)
                
                # Process audio
                if len(y) < SR * DUR:
                    y = np.pad(y, (0, int(SR * DUR - len(y))))
                else:
                    start = (len(y) - int(SR * DUR)) // 2
                    y = y[start:start + int(SR * DUR)]
                
                x = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    embedding = encoder(x).cpu().numpy().flatten()
                    embeddings.append(embedding)
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        if embeddings:
            # Average embeddings for this species
            species_embeddings[species] = np.mean(embeddings, axis=0)
            print(f"Processed {species}: {len(embeddings)} files")
    
    # Save database
    np.savez("ckpt/species_database.npz", **species_embeddings)
    return species_embeddings

def identify_by_similarity(audio_file, database_path="ckpt/species_database.npz"):
    """
    Identify species by similarity to database
    """
    # Load encoder
    encoder = CNNEncoder(EMBED_DIM, N_MELS).to(DEVICE)
    encoder.load_state_dict(torch.load("ckpt/birdclef24_encoder.pt", map_location=DEVICE))
    encoder.eval()
    
    # Load species database
    database = np.load(database_path)
    species_names = list(database.keys())
    species_embeddings = [database[name] for name in species_names]
    species_embeddings = np.array(species_embeddings)
    
    # Process input audio
    y, _ = librosa.load(audio_file, sr=SR, mono=True)
    if len(y) < SR * DUR:
        y = np.pad(y, (0, int(SR * DUR - len(y))))
    else:
        start = (len(y) - int(SR * DUR)) // 2
        y = y[start:start + int(SR * DUR)]
    
    x = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # Get embedding
    with torch.no_grad():
        query_embedding = encoder(x).cpu().numpy().flatten()
    
    # Calculate similarities
    similarities = cosine_similarity([query_embedding], species_embeddings)[0]
    
    # Get top matches
    top_indices = np.argsort(similarities)[::-1][:5]
    
    results = []
    for idx in top_indices:
        results.append({
            'species': species_names[idx],
            'similarity': similarities[idx]
        })
    
    return results

# Usage
if __name__ == "__main__":
    # Build database (run once)
    print("Building species database...")
    build_species_database()
    
    # Identify species
    print("\\nIdentifying species...")
    results = identify_by_similarity("your_audio_file.wav")
    
    print("\\nTop 5 similar species:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['species']}: {result['similarity']:.3f}")