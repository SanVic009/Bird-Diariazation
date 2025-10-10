# species_classifier.py
import torch
import torch.nn as nn
import librosa
import numpy as np
import os
from config import *
from model import CNNEncoder
import json

class BirdSpeciesClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = CNNEncoder(EMBED_DIM, N_MELS)
        self.classifier = nn.Linear(EMBED_DIM, num_classes)
    
    def forward(self, x):
        embeddings = self.encoder(x)
        return self.classifier(embeddings)

def train_species_classifier():
    """
    Train a classifier on top of the pretrained encoder
    """
    # Get species names from data folder
    species = sorted(os.listdir("data"))
    species_to_id = {species[i]: i for i in range(len(species))}
    
    print(f"Training classifier for {len(species)} species...")
    
    # Load pretrained encoder
    model = BirdSpeciesClassifier(len(species)).to(DEVICE)
    model.encoder.load_state_dict(torch.load("ckpt/birdclef24_encoder.pt", map_location=DEVICE))
    
    # Freeze encoder, only train classifier
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Simple training loop (you'd want a proper dataset here)
    model.train()
    for epoch in range(10):
        total_loss = 0
        count = 0
        
        for species_name in species[:20]:  # Train on first 20 species for demo
            species_path = f"data/{species_name}"
            audio_files = [f for f in os.listdir(species_path) if f.endswith('.ogg')][:5]  # 5 files per species
            
            for audio_file in audio_files:
                try:
                    # Load and process audio
                    audio_path = os.path.join(species_path, audio_file)
                    y, _ = librosa.load(audio_path, sr=SR, mono=True)
                    
                    if len(y) < SR * DUR:
                        y = np.pad(y, (0, int(SR * DUR - len(y))))
                    else:
                        start = np.random.randint(0, len(y) - int(SR * DUR))
                        y = y[start:start + int(SR * DUR)]
                    
                    x = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
                    y_true = torch.tensor([species_to_id[species_name]]).to(DEVICE)
                    
                    # Forward pass
                    pred = model(x)
                    loss = criterion(pred, y_true)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    count += 1
                    
                except Exception as e:
                    continue
        
        if count > 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss/count:.4f}")
    
    # Save the full model
    torch.save({
        'model_state_dict': model.state_dict(),
        'species_to_id': species_to_id,
        'id_to_species': {v: k for k, v in species_to_id.items()}
    }, "ckpt/species_classifier.pt")
    
    return model, species_to_id

def predict_species(audio_file, model_path="ckpt/species_classifier.pt"):
    """
    Predict bird species from audio file
    """
    # Load model and species mapping
    checkpoint = torch.load(model_path, map_location=DEVICE)
    id_to_species = checkpoint['id_to_species']
    
    model = BirdSpeciesClassifier(len(id_to_species)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and process audio
    y, _ = librosa.load(audio_file, sr=SR, mono=True)
    if len(y) < SR * DUR:
        y = np.pad(y, (0, int(SR * DUR - len(y))))
    else:
        # Take middle section
        start = (len(y) - int(SR * DUR)) // 2
        y = y[start:start + int(SR * DUR)]
    
    x = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 5)  # Top 5 predictions
    
    # Format results
    results = []
    for i in range(5):
        species_id = top_indices[0][i].item()
        prob = top_probs[0][i].item()
        species_name = id_to_species[species_id]
        results.append({
            'species': species_name,
            'confidence': prob
        })
    
    return results

# Usage
if __name__ == "__main__":
    # First train the classifier
    print("Training species classifier...")
    model, species_map = train_species_classifier()
    
    # Then use it for prediction
    print("\\nMaking predictions...")
    results = predict_species("your_audio_file.wav")
    
    print("\\nTop 5 predictions:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['species']}: {result['confidence']:.3f}")