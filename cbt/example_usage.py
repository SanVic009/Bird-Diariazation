#!/usr/bin/env python3
"""
Example usage of the bird diarization inference system
"""

from infer_birds import BirdDiarizationInferencer
import numpy as np

def example_usage():
    """Example of how to use the inferencer programmatically"""
    
    # Initialize the inferencer with your trained model
    model_path = "models/best_enhanced_model.pt"
    inferencer = BirdDiarizationInferencer(model_path, device='auto')
    
    # Example 1: Process an audio file
    print("="*60)
    print("Example 1: Processing audio file")
    print("="*60)
    
    # Replace with your actual audio file path
    audio_path = "path/to/your/bird_recording.wav"  # Change this!
    
    try:
        result = inferencer.infer_from_audio(
            audio_path, 
            segment_length=2.0  # 2-second segments
        )
        
        # Display results
        inferencer.print_results(result)
        
        # Save results
        inferencer.save_results(result, "audio_results.json")
        
    except FileNotFoundError:
        print(f"⚠️  Audio file not found: {audio_path}")
        print("   Please update the audio_path variable with a real file")
    except Exception as e:
        print(f"⚠️  Error processing audio: {e}")
    
    # Example 2: Process cached spectrograms
    print("\n" + "="*60)
    print("Example 2: Processing cached spectrograms")
    print("="*60)
    
    try:
        result = inferencer.infer_from_batch(
            "../cache_mels/",  # Your cache directory
            max_files=50       # Process first 50 files
        )
        
        # Display results
        inferencer.print_results(result)
        
        # Access specific information
        print(f"\n🔍 Detailed Analysis:")
        print(f"   Files processed: {result['n_files']}")
        print(f"   Method used: {result['method']}")
        
        # Get speaker assignments for each file
        if 'file_names' in result and 'labels' in result:
            print(f"\n📁 File-to-Speaker Mapping (first 10):")
            for i, (filename, speaker) in enumerate(zip(result['file_names'][:10], result['labels'][:10])):
                print(f"   {filename}: Bird_{speaker}")
        
        # Save results
        inferencer.save_results(result, "batch_results.json")
        
    except Exception as e:
        print(f"⚠️  Error processing batch: {e}")
    
    # Example 3: Process a single spectrogram
    print("\n" + "="*60)
    print("Example 3: Single spectrogram embedding")
    print("="*60)
    
    try:
        # Pick any .pt file from your cache
        spec_files = list(Path("../cache_mels/").glob("*.pt"))
        if spec_files:
            spec_path = spec_files[0]  # Use first file
            
            embedding = inferencer.infer_from_spectrogram(spec_path)
            print(f"Generated embedding for {spec_path.name}:")
            print(f"   Shape: {embedding.shape}")
            print(f"   Mean: {embedding.mean():.4f}")
            print(f"   Std: {embedding.std():.4f}")
            
            # You could save this embedding for later use
            np.save(f"embedding_{spec_path.stem}.npy", embedding)
            print(f"   Saved to: embedding_{spec_path.stem}.npy")
        else:
            print("   No .pt files found in ../cache_mels/")
            
    except Exception as e:
        print(f"⚠️  Error processing single spectrogram: {e}")

def analyze_speaker_patterns(result):
    """Analyze temporal patterns in the diarization result"""
    if 'timeline' not in result:
        return
    
    print(f"\n🔍 Speaker Pattern Analysis:")
    
    timeline = result['timeline']
    
    # Count speaker switches
    switches = len(timeline) - 1
    print(f"   Speaker switches: {switches}")
    
    # Average segment duration per speaker
    speaker_durations = {}
    for segment in timeline:
        speaker = segment['speaker']
        if speaker not in speaker_durations:
            speaker_durations[speaker] = []
        speaker_durations[speaker].append(segment['duration'])
    
    for speaker, durations in speaker_durations.items():
        avg_duration = sum(durations) / len(durations)
        print(f"   {speaker} average segment: {avg_duration:.1f}s ({len(durations)} segments)")
    
    # Find longest continuous segment for each speaker
    print(f"\n   Longest continuous segments:")
    for speaker, durations in speaker_durations.items():
        max_duration = max(durations)
        print(f"   {speaker}: {max_duration:.1f}s")

if __name__ == "__main__":
    # You need to make sure you have a trained model first!
    from pathlib import Path
    
    model_path = Path("models/best_enhanced_model.pt")
    if model_path.exists():
        example_usage()
    else:
        print("❌ No trained model found!")
        print(f"   Expected: {model_path}")
        print(f"   Please run training first: python improved_train.py")