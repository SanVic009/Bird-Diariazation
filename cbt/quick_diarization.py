#!/usr/bin/env python3
"""
quick_diarization.py - Simple script to quickly use your saved .pt models

Usage examples:
    python quick_diarization.py                          # Use cached data
    python quick_diarization.py audio.wav               # Diarize specific audio
    python quick_diarization.py --model best_model.pt   # Use specific model
"""

import torch
import numpy as np
import sys
import os
from train import DiarizationDataset, DiarizationEncoder, perform_diarization

def quick_diarize(model_path=None, audio_path=None, max_speakers=8):
    """
    Quick diarization with minimal setup
    
    Args:
        model_path: Path to .pt model file (auto-detects if None)
        audio_path: Path to audio file (uses cached data if None) 
        max_speakers: Maximum number of speakers to detect
    """
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load model
    model = DiarizationEncoder(embed_dim=128).to(device)
    
    if model_path is None:
        # Auto-detect best model
        candidates = [
            "models/best_diarization_model.pt",
            "../best_diarization_model.pt", 
            "models/final_diarization_encoder.pt",
            "../final_diarization_encoder.pt",
            "../diarization_encoder_epoch_50.pt",
            "../diarization_encoder_epoch_40.pt"
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break
        else:
            raise FileNotFoundError("❌ No trained model found!")
    
    print(f"📂 Loading model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    if audio_path is None:
        # Use cached data
        print("🗂️  Using cached mel spectrograms...")
        dataset = DiarizationDataset("cache_mels/" if os.path.exists("cache_mels/") else "../cache_mels/")
        
        # Test on random sample
        n_test = min(50, len(dataset))
        test_segments = []
        
        indices = np.random.choice(len(dataset), n_test, replace=False)
        for i in indices:
            segment, _, _ = dataset[i]
            test_segments.append(segment)
        
        print(f"🔍 Testing on {len(test_segments)} cached segments")
        
    else:
        print(f"🎵 Processing audio file: {audio_path}")
        # TODO: Add audio file processing here
        # For now, fall back to cached data
        print("⚠️  Audio file processing not implemented in quick script")
        print("   Using cached data instead...")
        return quick_diarize(model_path, None, max_speakers)
    
    # Perform diarization
    print("🧠 Running diarization...")
    labels, n_speakers, embeddings = perform_diarization(
        model, test_segments, device, max_speakers=max_speakers
    )
    
    # Display results
    print(f"\n🎉 Results:")
    print(f"   🔢 Number of different birds detected: {n_speakers}")
    print(f"   📊 Total segments analyzed: {len(labels)}")
    
    # Show speaker distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\n📋 Speaker breakdown:")
    for label, count in zip(unique_labels, counts):
        percentage = count / len(labels) * 100
        print(f"   🐦 Bird {label}: {count} segments ({percentage:.1f}%)")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    np.save("results/quick_embeddings.npy", embeddings)
    np.save("results/quick_labels.npy", labels)
    
    print(f"\n💾 Results saved to:")
    print(f"   📁 results/quick_embeddings.npy")
    print(f"   📁 results/quick_labels.npy")
    
    return {
        'labels': labels,
        'n_speakers': n_speakers,
        'embeddings': embeddings,
        'model_used': model_path
    }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick bird diarization using saved models")
    parser.add_argument("audio", nargs="?", help="Audio file to diarize (optional)")
    parser.add_argument("--model", "-m", help="Specific model file to use")
    parser.add_argument("--speakers", "-s", type=int, default=8, help="Max speakers (default: 8)")
    
    args = parser.parse_args()
    
    try:
        results = quick_diarize(
            model_path=args.model,
            audio_path=args.audio, 
            max_speakers=args.speakers
        )
        print(f"\n✅ Diarization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Quick troubleshooting:")
        print("   • Make sure you have trained .pt model files")
        print("   • Check that cache_mels/ directory exists with data")
        print("   • Verify you're in the correct directory")

if __name__ == "__main__":
    main()