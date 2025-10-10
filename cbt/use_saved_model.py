#!/usr/bin/env python3
"""
use_saved_model.py - Complete guide for using saved .pt diarization models

This script demonstrates how to:
1. Load saved .pt model files
2. Use them for bird diarization on new audio
3. Process single audio files or batches
4. Visualize and analyze results
"""

import torch
import numpy as np
import librosa
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from train import DiarizationDataset, DiarizationEncoder, perform_diarization

class DiarizationModelLoader:
    """Helper class to load and use saved diarization models"""
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the model loader
        
        Args:
            model_path: Path to the .pt model file (if None, will auto-detect)
            device: Device to use (if None, will auto-detect)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = None
        
        # Load model
        if model_path:
            self.load_model(model_path)
        else:
            self.auto_load_best_model()
    
    def auto_load_best_model(self):
        """Automatically find and load the best available model"""
        # Priority order: best model > final model > latest checkpoint
        model_candidates = [
            "models/best_diarization_model.pt",
            "../best_diarization_model.pt",
            "models/final_diarization_encoder.pt", 
            "../final_diarization_encoder.pt",
            "models/diarization_encoder_epoch_50.pt",
            "../diarization_encoder_epoch_50.pt",
            "../diarization_encoder_epoch_40.pt",
            "../diarization_encoder_epoch_30.pt"
        ]
        
        for candidate in model_candidates:
            if os.path.exists(candidate):
                self.load_model(candidate)
                return
        
        raise FileNotFoundError("No trained model found! Available options should be in models/ directory")
    
    def load_model(self, model_path):
        """Load a specific model file"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        self.model = DiarizationEncoder(embed_dim=128).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model_path = model_path
        print(f"✅ Model loaded successfully on {self.device}")
    
    def diarize_audio_file(self, audio_path, segment_length=5.0, hop_length=2.5, max_speakers=8):
        """
        Diarize a single audio file
        
        Args:
            audio_path: Path to the audio file
            segment_length: Length of each segment in seconds
            hop_length: Hop length between segments in seconds
            max_speakers: Maximum number of speakers/birds to detect
            
        Returns:
            dict with results: timestamps, speaker_labels, n_speakers, embeddings
        """
        print(f"\n🎵 Processing audio file: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=22050)
        duration = len(audio) / sr
        print(f"   Duration: {duration:.1f} seconds, Sample rate: {sr} Hz")
        
        # Create segments
        segment_samples = int(segment_length * sr)
        hop_samples = int(hop_length * sr)
        
        segments = []
        timestamps = []
        
        for start_sample in range(0, len(audio) - segment_samples + 1, hop_samples):
            end_sample = start_sample + segment_samples
            segment_audio = audio[start_sample:end_sample]
            
            # Convert to mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=segment_audio,
                sr=sr,
                n_mels=128,
                hop_length=512,
                n_fft=2048
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to expected input size (128, 128)
            if mel_spec_db.shape[1] != 128:
                mel_spec_db = np.resize(mel_spec_db, (128, 128))
            
            segments.append(torch.tensor(mel_spec_db, dtype=torch.float32))
            timestamps.append(start_sample / sr)
        
        print(f"   Created {len(segments)} segments for analysis")
        
        # Perform diarization
        if segments:
            labels, n_speakers, embeddings = perform_diarization(
                self.model, segments, self.device, max_speakers=max_speakers
            )
            
            results = {
                'audio_path': audio_path,
                'timestamps': np.array(timestamps),
                'speaker_labels': labels,
                'n_speakers': n_speakers,
                'embeddings': embeddings,
                'segment_length': segment_length,
                'hop_length': hop_length,
                'total_duration': duration
            }
            
            print(f"   🎯 Detected {n_speakers} different birds/speakers")
            return results
        else:
            print("   ❌ No segments could be created from this audio")
            return None
    
    def diarize_cached_data(self, cache_dir="cache_mels/", n_samples=50, max_speakers=8):
        """
        Diarize pre-processed cached data
        
        Args:
            cache_dir: Directory containing cached mel spectrograms
            n_samples: Number of random samples to test
            max_speakers: Maximum number of speakers to detect
        """
        print(f"\n📁 Processing cached data from: {cache_dir}")
        
        dataset = DiarizationDataset(cache_dir)
        print(f"   Found {len(dataset)} cached segments")
        
        # Sample random segments
        n_test = min(n_samples, len(dataset))
        test_indices = np.random.choice(len(dataset), size=n_test, replace=False)
        
        test_segments = []
        for i in test_indices:
            segment, _, _ = dataset[i]
            test_segments.append(segment)
        
        print(f"   Testing on {len(test_segments)} segments")
        
        # Perform diarization
        labels, n_speakers, embeddings = perform_diarization(
            self.model, test_segments, self.device, max_speakers=max_speakers
        )
        
        results = {
            'source': 'cached_data',
            'cache_dir': cache_dir,
            'n_segments': len(test_segments),
            'speaker_labels': labels,
            'n_speakers': n_speakers,
            'embeddings': embeddings
        }
        
        print(f"   🎯 Detected {n_speakers} different birds/speakers")
        return results
    
    def visualize_results(self, results, save_plots=True):
        """Visualize diarization results"""
        if results is None:
            return
        
        print(f"\n📊 Visualizing results...")
        
        labels = results['speaker_labels']
        embeddings = results['embeddings']
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Bird Diarization Results - {results.get('n_speakers', 'Unknown')} Speakers Detected", 
                     fontsize=16)
        
        # 1. Speaker timeline (if timestamps available)
        if 'timestamps' in results:
            ax1 = axes[0, 0]
            timestamps = results['timestamps']
            ax1.scatter(timestamps, labels, c=labels, cmap='tab10', alpha=0.7)
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Speaker ID')
            ax1.set_title('Speaker Timeline')
            ax1.grid(True, alpha=0.3)
        
        # 2. Speaker distribution
        ax2 = axes[0, 1]
        unique_labels, counts = np.unique(labels, return_counts=True)
        ax2.bar(unique_labels, counts, color=plt.cm.tab10(unique_labels / max(unique_labels)))
        ax2.set_xlabel('Speaker ID')
        ax2.set_ylabel('Number of Segments')
        ax2.set_title('Speaker Distribution')
        
        # 3. t-SNE embedding visualization
        ax3 = axes[1, 0]
        if len(embeddings) > 1:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_2d = tsne.fit_transform(embeddings)
            scatter = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                 c=labels, cmap='tab10', alpha=0.7)
            ax3.set_title('t-SNE Embedding Visualization')
            ax3.set_xlabel('t-SNE 1')
            ax3.set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=ax3)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
Diarization Summary:
• Total segments: {len(labels)}
• Unique speakers: {results['n_speakers']}
• Model used: {os.path.basename(self.model_path)}
• Device: {self.device}

Speaker Statistics:"""
        
        for i, (label, count) in enumerate(zip(unique_labels, counts)):
            percentage = count / len(labels) * 100
            summary_text += f"\n• Speaker {label}: {count} segments ({percentage:.1f}%)"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs("results", exist_ok=True)
            plot_path = "results/diarization_visualization.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   📈 Visualization saved to: {plot_path}")
        
        plt.show()
        return fig

def main():
    """Main demonstration of how to use saved models"""
    print("🐦 Bird Diarization Model Usage Demo")
    print("=" * 50)
    
    try:
        # Initialize model loader
        loader = DiarizationModelLoader()
        
        print("\n" + "="*50)
        print("OPTION 1: Diarize from cached mel spectrograms")
        print("="*50)
        
        # Test on cached data
        results = loader.diarize_cached_data(n_samples=30)
        if results:
            loader.visualize_results(results)
        
        print("\n" + "="*50)
        print("OPTION 2: Diarize a new audio file")
        print("="*50)
        
        # Example: Diarize a specific audio file (if it exists)
        audio_candidates = [
            "../sound/test_audio.wav",
            "../birdclef-2024/train_audio/asbfly/XC134896.ogg",
            "../synthetic_audio/mixed_audio_001.wav"
        ]
        
        for audio_path in audio_candidates:
            if os.path.exists(audio_path):
                print(f"\nFound audio file: {audio_path}")
                results = loader.diarize_audio_file(audio_path, segment_length=5.0)
                if results:
                    loader.visualize_results(results)
                break
        else:
            print("\n📝 To diarize your own audio file, use:")
            print("   loader.diarize_audio_file('path/to/your/audio.wav')")
        
        print("\n" + "="*50)
        print("OPTION 3: Load specific model")
        print("="*50)
        print("\nTo use a specific model:")
        print("   loader = DiarizationModelLoader('path/to/specific/model.pt')")
        print("\nAvailable models in your workspace:")
        for root, dirs, files in os.walk(".."):
            for file in files:
                if file.endswith('.pt') and 'diarization' in file:
                    print(f"   • {os.path.join(root, file)}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Trained models (.pt files)")
        print("   2. Required dependencies installed")
        print("   3. Audio data available")

if __name__ == "__main__":
    main()