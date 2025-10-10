#!/usr/bin/env python3
"""
train.py – Bird Diarization Training (Unsupervised)
Learns embeddings to separate different birds without knowing species names.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import random
import os

# ----------------------------
# Dataset for Diarization (No Labels Needed!)
# ----------------------------
class DiarizationDataset(Dataset):
    def __init__(self, root, segment_length=2.0):
        self.root = Path(root)
        self.files = list(self.root.glob("*.pt"))
        self.segment_length = segment_length
        
        # We don't use species labels - just file names for tracking
        self.file_ids = [f.stem.split("_")[0] for f in self.files]
        
        print("--- DIARIZATION DATASET ---")
        print(f"Found {len(self.files)} audio segments.")
        print("No species labels used - learning embeddings for clustering!")
        print("---------------------------")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = torch.load(self.files[idx])
        
        # Pad or crop to fixed size
        target_width = 501
        current_width = x.shape[-1]
        
        if current_width > target_width:
            # Random crop for augmentation
            start_idx = torch.randint(0, current_width - target_width + 1, (1,)).item()
            x = x[..., start_idx:start_idx + target_width]
        else:
            pad_width = target_width - current_width
            x = F.pad(x, (0, pad_width))

        # Normalize per sample
        mean = x.mean()
        std = x.std()
        if std > 1e-6:
            x = (x - mean) / std
        else:
            x = x - mean

        x = x.unsqueeze(0)  # [1, n_mels, time]
        
        # Return two augmented views for contrastive learning
        x1 = self._augment(x.clone())
        x2 = self._augment(x.clone())
        
        return x1, x2, idx  # Return index for tracking

    def _augment(self, x):
        """Simple augmentation for contrastive learning"""
        # Time masking
        if torch.rand(1) < 0.3:
            mask_size = torch.randint(1, 20, (1,)).item()
            mask_start = torch.randint(0, x.shape[-1] - mask_size, (1,)).item()
            x[..., mask_start:mask_start + mask_size] = 0
        
        # Frequency masking  
        if torch.rand(1) < 0.3:
            freq_mask_size = torch.randint(1, 10, (1,)).item()
            freq_start = torch.randint(0, x.shape[-2] - freq_mask_size, (1,)).item()
            x[..., freq_start:freq_start + freq_mask_size, :] = 0
            
        return x


# ----------------------------
# Simple Embedding Model for Diarization
# ----------------------------
class DiarizationEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        
        # Simple CNN encoder
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Embedding layer
        self.embed = nn.Linear(128, embed_dim)
        
    def forward(self, x):
        # Extract features
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        
        # Generate embeddings
        embeddings = self.embed(x)
        
        # L2 normalize for better clustering
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

# ----------------------------
# Contrastive Loss for Diarization (FIXED)
# ----------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z1, z2):
        batch_size = z1.size(0)
        
        # Concatenate z1 and z2
        z = torch.cat([z1, z2], dim=0)  # Shape: [2*batch_size, embed_dim]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Remove diagonal (self-similarity) - use large negative number instead of -inf
        diag_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(diag_mask, -1e9)
        
        # Labels: first half should match second half
        # z1[0] matches z2[0] -> positions 0 and batch_size
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),  # z1 matches z2
            torch.arange(0, batch_size, device=z.device)  # z2 matches z1
        ])
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


# ----------------------------
# Early stopping
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=7, delta=0.0, path="models/best_model.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = float("inf")
        self.best_score = float("inf")  # For compatibility
        self.counter = 0
        self.early_stop = False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_score = val_loss  # Keep best_score in sync
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f"New best loss: {val_loss:.4f} - Model saved to {self.path}!")
        else:
            self.counter += 1
            print(f"No improvement for {self.counter} epochs (patience: {self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True


# ----------------------------
# Training loop
# ----------------------------
# ----------------------------
# Diarization Training Function (With Early Stopping)
# ----------------------------
def train_diarization(train_loader, device, embed_dim=128, epochs=30):
    model = DiarizationEncoder(embed_dim).to(device)
    criterion = ContrastiveLoss(temperature=0.5)  # Higher temperature for stability
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)  # Lower LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Early stopping for unsupervised learning (based on loss stabilization)
    early_stopper = EarlyStopping(patience=10, delta=0.001, path="models/best_diarization_model.pt")
    
    print("Training diarization encoder...")
    print(f"Models will be saved to: models/")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        valid_batches = 0
        
        for batch_idx, (x1, x2, indices) in enumerate(train_loader):
            x1, x2 = x1.to(device), x2.to(device)
            
            # Get embeddings
            z1 = model(x1)
            z2 = model(x2)
            
            # Check for NaN/inf in embeddings
            if torch.isnan(z1).any() or torch.isnan(z2).any():
                print(f"Warning: NaN detected in embeddings at epoch {epoch+1}, batch {batch_idx}")
                continue
                
            if torch.isinf(z1).any() or torch.isinf(z2).any():
                print(f"Warning: Inf detected in embeddings at epoch {epoch+1}, batch {batch_idx}")
                continue
            
            # Compute contrastive loss
            loss = criterion(z1, z2)
            
            # Check for NaN/inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at epoch {epoch+1}, batch {batch_idx}: {loss.item()}")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            valid_batches += 1
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
                print(f"  Batch {batch_idx}/{len(train_loader)}: Avg Loss = {avg_loss:.4f}")
        
        scheduler.step()
        avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
        
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}, Valid batches = {valid_batches}/{num_batches}")
        
        # Early stopping check (using training loss for unsupervised learning)
        early_stopper(avg_loss, model)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best loss: {early_stopper.best_score:.4f}")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"models/diarization_encoder_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Load best model
    model.load_state_dict(torch.load("models/best_diarization_model.pt"))
    print("Loaded best model from early stopping")
    
    return model

# ----------------------------
# Diarization Inference
# ----------------------------
def perform_diarization(model, audio_segments, device, max_speakers=6):
    """
    Cluster audio segments to identify different speakers/birds
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for segment in audio_segments:
            # Move segment to the same device as the model
            segment = segment.to(device)
            embedding = model(segment.unsqueeze(0))
            embeddings.append(embedding.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    
    # Find optimal number of clusters using silhouette score
    from sklearn.metrics import silhouette_score
    
    best_score = -1
    best_k = 2
    
    for k in range(2, min(max_speakers + 1, len(embeddings))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
    
    # Final clustering
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = kmeans.fit_predict(embeddings)
    
    print(f"Found {best_k} different birds/speakers (silhouette score: {best_score:.3f})")
    
    return final_labels, best_k, embeddings


# ----------------------------
# Main Diarization Pipeline
# ----------------------------
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create diarization dataset (without labels)
    dataset = DiarizationDataset("cache_mels/")
    
    # Create data loader for training
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"Total audio segments: {len(dataset)}")
    
    # Train the diarization encoder
    model = train_diarization(train_loader, device, embed_dim=128, epochs=50)
    
    # Save final model
    torch.save(model.state_dict(), "models/final_diarization_encoder.pt")
    print("Final model saved to models/final_diarization_encoder.pt")
    
    print("\nDiarization encoder training completed!")
    
    # Example of performing diarization on test data
    print("\nTesting diarization on sample data...")
    
    # Load some sample audio segments for testing
    test_segments = []
    for i in range(min(20, len(dataset))):  # Test on first 20 segments
        segment, _, _ = dataset[i]
        test_segments.append(segment)
    
    if test_segments:
        # Perform diarization
        labels, n_speakers, embeddings = perform_diarization(model, test_segments, device)
        
        print(f"Diarization results:")
        print(f"- Number of different speakers/birds detected: {n_speakers}")
        print(f"- Speaker assignments for first 10 segments: {labels[:10]}")
        
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        # Save embeddings for further analysis
        np.save("results/audio_embeddings.npy", embeddings)
        np.save("results/speaker_labels.npy", labels)
        print("Results saved to results/ directory")
        
    print("Diarization pipeline completed!")
