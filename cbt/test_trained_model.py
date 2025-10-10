#!/usr/bin/env python3
"""
test_trained_model.py - Test the trained diarization model
"""

import torch
import numpy as np
from train import DiarizationDataset, DiarizationEncoder, perform_diarization

def test_trained_model():
    """Test the trained diarization model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained model
    model = DiarizationEncoder(embed_dim=128).to(device)
    
    try:
        # Try to load the final model first
        model.load_state_dict(torch.load("final_diarization_encoder.pt", map_location=device))
        print("✓ Loaded final_diarization_encoder.pt")
    except FileNotFoundError:
        try:
            # Try to load the latest checkpoint
            model.load_state_dict(torch.load("diarization_encoder_epoch_50.pt", map_location=device))
            print("✓ Loaded diarization_encoder_epoch_50.pt")
        except FileNotFoundError:
            try:
                # Try to load any available checkpoint
                model.load_state_dict(torch.load("diarization_encoder_epoch_40.pt", map_location=device))
                print("✓ Loaded diarization_encoder_epoch_40.pt")
            except FileNotFoundError:
                print("❌ No trained model found. Please run training first.")
                return
    
    # Load test data
    dataset = DiarizationDataset("cache_mels/")
    
    # Test on a small sample
    print(f"\nTesting on sample data from {len(dataset)} total segments...")
    test_segments = []
    test_indices = np.random.choice(len(dataset), size=min(30, len(dataset)), replace=False)
    
    for i in test_indices:
        segment, _, _ = dataset[i]
        test_segments.append(segment)
    
    # Perform diarization
    print("Performing diarization...")
    labels, n_speakers, embeddings = perform_diarization(model, test_segments, device, max_speakers=8)
    
    print("\n🎉 Diarization Results:")
    print(f"   📊 Number of different birds/speakers detected: {n_speakers}")
    print(f"   🏷️  Speaker assignments: {labels}")
    print(f"   📈 Embedding shape: {embeddings.shape}")
    
    # Analyze the results
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n📋 Speaker Distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"   Speaker {label}: {count} segments ({count/len(labels)*100:.1f}%)")
    
    # Save results
    np.save("test_embeddings.npy", embeddings)
    np.save("test_labels.npy", labels)
    print(f"\n💾 Results saved to test_embeddings.npy and test_labels.npy")
    
    return True

if __name__ == "__main__":
    test_trained_model()