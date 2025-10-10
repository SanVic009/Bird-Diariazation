#!/usr/bin/env python3
"""
test_diarization.py - Test the diarization pipeline
"""

import torch
import numpy as np
from train import DiarizationDataset, DiarizationEncoder, ContrastiveLoss
from torch.utils.data import DataLoader

def test_diarization_pipeline():
    """Test if the diarization pipeline works correctly"""
    print("Testing Bird Diarization Pipeline...")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test dataset creation
    try:
        dataset = DiarizationDataset("cache_mels/")
        print(f"✓ Dataset loaded successfully with {len(dataset)} audio segments")
        
        # Test data loading
        if len(dataset) > 0:
            sample_x1, sample_x2, sample_idx = dataset[0]
            print(f"✓ Sample data shape: {sample_x1.shape}, {sample_x2.shape}")
            
            # Test data loader
            loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
            batch_x1, batch_x2, batch_indices = next(iter(loader))
            print(f"✓ Batch data shape: {batch_x1.shape}, {batch_x2.shape}")
            
            # Test model creation
            model = DiarizationEncoder(embed_dim=128)
            print(f"✓ Model created successfully")
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                embedding1 = model(batch_x1)
                embedding2 = model(batch_x2)
                print(f"✓ Model forward pass successful: {embedding1.shape}")
            
            # Test loss function
            criterion = ContrastiveLoss(temperature=0.1)
            loss = criterion(embedding1, embedding2)
            print(f"✓ Contrastive loss computation successful: {loss.item():.4f}")
            
            print("\n🎉 All tests passed! The diarization pipeline is ready to use.")
            print("\nTo start training, run:")
            print("  python train.py")
            
        else:
            print("❌ No audio files found in cache_mels/ directory")
            print("Please run preprocessing first to create mel spectrograms")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_diarization_pipeline()