#!/usr/bin/env python3
"""
infer_birds.py - Bird Diarization Inference

Usage:
    python infer_birds.py --audio path/to/audio.wav --model models/best_enhanced_model.pt
    python infer_birds.py --spectrogram path/to/spec.pt --model models/best_enhanced_model.pt
    python infer_birds.py --batch path/to/cache_mels/ --model models/best_enhanced_model.pt
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Import our components
from improved_models import ImprovedDiarizationEncoder
from advanced_clustering import perform_advanced_diarization

class BirdDiarizationInferencer:
    """Complete bird diarization inference pipeline"""
    
    def __init__(self, model_path, device='auto'):
        """
        Initialize the inferencer
        
        Args:
            model_path: Path to trained model (.pt file)
            device: 'auto', 'cuda', or 'cpu'
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Loading Bird Diarization Model")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_path}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        
    def _load_model(self, model_path):
        """Load the trained model"""
        try:
            # Load checkpoint (trust our own model files)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Initialize model with same architecture
            model = ImprovedDiarizationEncoder(
                embed_dim=256,  # Should match training config
                num_heads=8,
                dropout=0.1
            ).to(self.device)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"   Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
                if 'score' in checkpoint:
                    print(f"   Model score: {checkpoint['score']:.4f}")
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
            
            return model
            
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    def infer_from_audio(self, audio_path, sr=22050, segment_length=2.0):
        """
        Infer bird speakers from raw audio file
        
        Args:
            audio_path: Path to audio file (.wav, .mp3, etc.)
            sr: Sample rate for processing
            segment_length: Length of each segment in seconds
            
        Returns:
            dict with diarization results
        """
        print(f"🎵 Processing audio: {audio_path}")
        
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr)
        duration = len(audio) / sr
        
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Sample rate: {sr} Hz")
        
        # Convert to spectrograms
        spectrograms = self._audio_to_spectrograms(audio, sr, segment_length)
        
        if len(spectrograms) == 0:
            raise ValueError("No valid spectrograms could be generated from audio")
        
        print(f"   Generated {len(spectrograms)} segments")
        
        # Get embeddings
        embeddings = self._spectrograms_to_embeddings(spectrograms)
        
        # Perform diarization
        result = self._diarize_embeddings(embeddings, segment_length)
        
        # Add metadata
        result['input_file'] = str(audio_path)
        result['duration'] = duration
        result['n_segments'] = len(spectrograms)
        result['segment_length'] = segment_length
        
        return result
    
    def infer_from_spectrogram(self, spec_path):
        """
        Infer from a single spectrogram file (.pt)
        
        Args:
            spec_path: Path to spectrogram tensor file
            
        Returns:
            numpy array: embedding vector
        """
        print(f"📊 Processing spectrogram: {spec_path}")
        
        # Load spectrogram
        spec = torch.load(spec_path, map_location='cpu', weights_only=False)
        
        if isinstance(spec, dict):
            # If it's a checkpoint, extract the spectrogram
            spec = spec.get('spectrogram', spec.get('data', spec))
        
        # Ensure correct shape [1, channels, height, width]
        if spec.dim() == 2:  # [height, width] -> [1, 1, height, width]
            spec = spec.unsqueeze(0).unsqueeze(0)
        elif spec.dim() == 3:  # [channels, height, width] -> [1, channels, height, width]
            spec = spec.unsqueeze(0)
        
        print(f"   Spectrogram shape: {spec.shape}")
        
        # Get embedding
        with torch.no_grad():
            spec = spec.to(self.device)
            embedding = self.model(spec)
            embedding = F.normalize(embedding, dim=1)  # L2 normalize
            
        return embedding.cpu().numpy()
    
    def infer_from_batch(self, cache_dir, max_files=None):
        """
        Process multiple spectrogram files from a directory
        
        Args:
            cache_dir: Directory containing .pt files
            max_files: Maximum number of files to process (None for all)
            
        Returns:
            dict with diarization results
        """
        print(f"📁 Processing batch from: {cache_dir}")
        
        cache_path = Path(cache_dir)
        spec_files = list(cache_path.glob("*.pt"))
        
        if max_files:
            spec_files = spec_files[:max_files]
        
        if len(spec_files) == 0:
            raise ValueError(f"No .pt files found in {cache_dir}")
        
        print(f"   Found {len(spec_files)} spectrogram files")
        
        # Process all spectrograms
        all_embeddings = []
        file_names = []
        
        for spec_file in tqdm(spec_files, desc="Extracting embeddings"):
            try:
                embedding = self.infer_from_spectrogram(spec_file)
                all_embeddings.append(embedding.squeeze())  # Remove batch dimension
                file_names.append(spec_file.name)
            except Exception as e:
                print(f"⚠️  Failed to process {spec_file}: {e}")
                continue
        
        if len(all_embeddings) == 0:
            raise ValueError("No valid spectrograms could be processed")
        
        # Stack embeddings
        embeddings = np.stack(all_embeddings)
        
        print(f"   Successfully processed {len(embeddings)} files")
        print(f"   Embedding shape: {embeddings.shape}")
        
        # Perform diarization
        result = self._diarize_embeddings(embeddings)
        
        # Add file mapping
        result['file_names'] = file_names
        result['input_dir'] = str(cache_dir)
        result['n_files'] = len(file_names)
        
        return result
    
    def _audio_to_spectrograms(self, audio, sr, segment_length):
        """Convert audio to spectrograms"""
        segment_samples = int(segment_length * sr)
        spectrograms = []
        
        # Split into segments
        for start in range(0, len(audio) - segment_samples, segment_samples):
            segment = audio[start:start + segment_samples]
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=segment,
                sr=sr,
                n_mels=128,
                hop_length=256,
                win_length=512,
                n_fft=1024
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [-1, 1]
            mel_spec_norm = 2 * (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min()) - 1
            
            # Ensure consistent width (padding/truncating)
            target_width = 501  # Should match training
            if mel_spec_norm.shape[1] < target_width:
                # Pad
                pad_width = target_width - mel_spec_norm.shape[1]
                mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
            elif mel_spec_norm.shape[1] > target_width:
                # Truncate
                mel_spec_norm = mel_spec_norm[:, :target_width]
            
            spectrograms.append(mel_spec_norm)
        
        return spectrograms
    
    def _spectrograms_to_embeddings(self, spectrograms):
        """Convert spectrograms to embeddings"""
        embeddings = []
        
        with torch.no_grad():
            for spec in tqdm(spectrograms, desc="Generating embeddings"):
                # Convert to tensor and add batch dimension
                spec_tensor = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                spec_tensor = spec_tensor.to(self.device)
                
                # Get embedding
                embedding = self.model(spec_tensor)
                embedding = F.normalize(embedding, dim=1)  # L2 normalize
                
                embeddings.append(embedding.cpu().numpy().squeeze())
        
        return np.stack(embeddings)
    
    def _diarize_embeddings(self, embeddings, segment_length=None):
        """Perform clustering on embeddings"""
        print(f"🎯 Performing diarization on {len(embeddings)} embeddings...")
        
        # Perform clustering
        clustering_result = perform_advanced_diarization(embeddings, max_speakers=8)
        
        if clustering_result is None:
            raise ValueError("Clustering failed")
        
        # Create timeline if segment_length is provided
        timeline = None
        if segment_length:
            timeline = self._create_timeline(clustering_result['labels'], segment_length)
        
        # Prepare final result
        result = {
            'n_speakers': clustering_result['n_speakers'],
            'labels': clustering_result['labels'].tolist(),
            'method': clustering_result['method'],
            'metrics': clustering_result['metrics'],
            'timeline': timeline,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _create_timeline(self, labels, segment_length):
        """Create a timeline showing when each speaker is active"""
        timeline = []
        
        current_speaker = labels[0]
        start_time = 0.0
        
        for i, speaker in enumerate(labels[1:], 1):
            if speaker != current_speaker:
                # End of current segment
                end_time = i * segment_length
                timeline.append({
                    'speaker': f"Bird_{current_speaker}",
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time
                })
                
                # Start of new segment
                current_speaker = speaker
                start_time = end_time
        
        # Add final segment
        end_time = len(labels) * segment_length
        timeline.append({
            'speaker': f"Bird_{current_speaker}",
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        })
        
        return timeline
    
    def save_results(self, result, output_path):
        """Save results to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Make numpy arrays JSON serializable
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Convert all numpy objects
        json_result = json.loads(json.dumps(result, default=convert_numpy))
        
        with open(output_path, 'w') as f:
            json.dump(json_result, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
    
    def print_results(self, result):
        """Print results in a readable format"""
        print(f"\n{'='*60}")
        print(f"🎯 BIRD DIARIZATION RESULTS")
        print(f"{'='*60}")
        
        print(f"📊 Overview:")
        print(f"   Detected Speakers: {result['n_speakers']}")
        print(f"   Clustering Method: {result['method']}")
        
        if 'metrics' in result and 'silhouette_score' in result['metrics']:
            silhouette = result['metrics']['silhouette_score']
            if isinstance(silhouette, (int, float)):
                print(f"   Quality Score: {silhouette:.3f}")
            else:
                print(f"   Quality Score: {silhouette}")
        
        if 'n_segments' in result:
            print(f"   Total Segments: {result['n_segments']}")
        
        if 'duration' in result:
            print(f"   Audio Duration: {result['duration']:.1f}s")
        
        # Timeline
        if result.get('timeline'):
            print(f"\n⏰ Timeline:")
            for segment in result['timeline']:
                print(f"   {segment['start']:.1f}-{segment['end']:.1f}s: {segment['speaker']} ({segment['duration']:.1f}s)")
        
        # Speaker distribution
        if 'labels' in result:
            labels = result['labels']
            unique, counts = np.unique(labels, return_counts=True)
            print(f"\n📈 Speaker Distribution:")
            for speaker, count in zip(unique, counts):
                percentage = (count / len(labels)) * 100
                print(f"   Bird_{speaker}: {count} segments ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Bird Diarization Inference")
    parser.add_argument('--model', required=True, help="Path to trained model (.pt file)")
    parser.add_argument('--audio', help="Path to audio file")
    parser.add_argument('--spectrogram', help="Path to spectrogram file (.pt)")
    parser.add_argument('--batch', help="Path to directory with spectrogram files")
    parser.add_argument('--output', help="Output path for results (.json)")
    parser.add_argument('--max-files', type=int, help="Maximum files to process in batch mode")
    parser.add_argument('--segment-length', type=float, default=2.0, help="Segment length in seconds")
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize inferencer
    inferencer = BirdDiarizationInferencer(args.model, device=args.device)
    
    # Perform inference based on input type
    if args.audio:
        result = inferencer.infer_from_audio(args.audio, segment_length=args.segment_length)
    elif args.spectrogram:
        embedding = inferencer.infer_from_spectrogram(args.spectrogram)
        print(f"Generated embedding shape: {embedding.shape}")
        return  # Single spectrogram doesn't need clustering
    elif args.batch:
        result = inferencer.infer_from_batch(args.batch, max_files=args.max_files)
    else:
        print("❌ Please specify --audio, --spectrogram, or --batch")
        return
    
    # Display results
    inferencer.print_results(result)
    
    # Save results if requested
    if args.output:
        inferencer.save_results(result, args.output)

if __name__ == "__main__":
    main()