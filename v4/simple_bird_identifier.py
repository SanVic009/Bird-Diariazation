#!/usr/bin/env python3
"""
Simple Bird Species Identifier using MobileNet
==============================================

This script uses a pre-trained MobileNet model to identify bird species from audio input.
It can process either a single audio file or listen to the microphone in real-time.

Requirements:
- torch, torchaudio for model inferen    parser.add_argument(
        '--labels',
        type=str,
        default='./labels_22_species.json',
        help='Path to the labels JSON file (default: ./labels_22_species.json)'
    )ibrosa for audio processing
- numpy for numerical operations
- pyaudio for microphone input (optional)

Usage:
    # Process an audio file
    python simple_bird_identifier.py --file path/to/audio.wav
    
    # Listen to microphone (if PyAudio is available)
    python simple_bird_identifier.py --mic
    
    # Specify custom model
    python simple_bird_identifier.py --model checkpoints_mobilenet/best_model.pth --file audio.wav

Author: GitHub Copilot
Date: September 2025
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import librosa

# Add v4 directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mobilenet import MobileNetBird
from preprocessing import BirdPreprocessor


class SimpleBirdIdentifier:
    """Simple bird species identification using MobileNet."""
    
    def __init__(
        self,
        model_path: str,
        labels_path: str = "../labels.json",
        device: str = None
    ):
        """
        Initialize the bird identifier.
        
        Args:
            model_path: Path to the trained MobileNet model (.pth file)
            labels_path: Path to the labels JSON file
            device: Device to use ('cpu' or 'cuda'). If None, auto-detect.
        """
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Load labels
        self.labels = self._load_labels(labels_path)
        self.num_classes = len(self.labels)
        print(f"Loaded {self.num_classes} bird species labels")
        
        # Load model
        self.model = self._load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Initialize preprocessor
        self.preprocessor = BirdPreprocessor()
    
    def _load_labels(self, labels_path: str) -> List[str]:
        """Load bird species labels from JSON file."""
        with open(labels_path, 'r') as f:
            data = json.load(f)
            return data['labels']
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained MobileNet model."""
        # Load checkpoint first to determine number of classes
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Determine number of classes from classifier layer
        classifier_weight_key = 'mobilenet.classifier.1.weight'
        if classifier_weight_key in state_dict:
            model_num_classes = state_dict[classifier_weight_key].shape[0]
            print(f"Model was trained on {model_num_classes} classes")
        else:
            model_num_classes = self.num_classes
            print(f"Could not determine model classes, using {model_num_classes}")
        
        # Initialize model with correct number of classes
        model = MobileNetBird(n_classes=model_num_classes)
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        # Store the actual number of classes the model was trained on
        self.model_num_classes = model_num_classes
        
        return model
    
    def _preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Preprocess audio file for model inference."""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=32000)
        print(f"Loaded audio: {len(audio)/sr:.2f} seconds at {sr} Hz")
        
        # Process to fixed duration chunks
        processed_chunks = self.preprocessor.process_duration(audio)
        
        # Use the first chunk (you could also average predictions across all chunks)
        audio_chunk = processed_chunks[0]
        
        # Convert to mel spectrogram
        log_mel = self.preprocessor.to_log_mel(audio_chunk)
        
        # Convert to tensor and add batch dimension
        spectrogram = torch.from_numpy(log_mel).float()
        spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
        
        return spectrogram.to(self.device)
    
    def _preprocess_audio_array(self, audio: np.ndarray, sample_rate: int = 32000) -> torch.Tensor:
        """Preprocess audio numpy array for model inference."""
        # Resample if necessary
        if sample_rate != 32000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=32000)
        
        # Process to fixed duration chunks
        processed_chunks = self.preprocessor.process_duration(audio)
        
        # Use the first chunk
        audio_chunk = processed_chunks[0]
        
        # Convert to mel spectrogram
        log_mel = self.preprocessor.to_log_mel(audio_chunk)
        
        # Convert to tensor and add batch dimension
        spectrogram = torch.from_numpy(log_mel).float()
        spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
        
        return spectrogram.to(self.device)
    
    def predict(self, spectrogram: torch.Tensor, top_k: int = 5) -> Tuple[str, float, Dict[str, float]]:
        """Make prediction on preprocessed audio."""
        with torch.no_grad():
            logits = self.model(spectrogram)
            probabilities = F.softmax(logits, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(top_k, self.num_classes), dim=1)
            
            # Convert to CPU and numpy
            top_probs = top_probs.cpu().numpy().flatten()
            top_indices = top_indices.cpu().numpy().flatten()
            
            # Get top prediction
            predicted_label = self.labels[top_indices[0]]
            confidence = top_probs[0]
            
            # Create top predictions dictionary
            top_predictions = {
                self.labels[idx]: prob 
                for idx, prob in zip(top_indices, top_probs)
            }
            
            return predicted_label, confidence, top_predictions
    
    def identify_from_file(self, audio_path: str, top_k: int = 5) -> Tuple[str, float, Dict[str, float]]:
        """Identify bird species from an audio file."""
        print(f"\nProcessing audio file: {audio_path}")
        
        # Preprocess audio
        spectrogram = self._preprocess_audio(audio_path)
        
        # Make prediction
        prediction, confidence, top_predictions = self.predict(spectrogram, top_k)
        
        return prediction, confidence, top_predictions
    
    def identify_from_array(self, audio: np.ndarray, sample_rate: int = 32000, top_k: int = 5) -> Tuple[str, float, Dict[str, float]]:
        """Identify bird species from an audio numpy array."""
        # Preprocess audio
        spectrogram = self._preprocess_audio_array(audio, sample_rate)
        
        # Make prediction
        prediction, confidence, top_predictions = self.predict(spectrogram, top_k)
        
        return prediction, confidence, top_predictions
    
    def listen_microphone(self, duration: float = 5.0, threshold: float = 0.3):
        """Listen to microphone and identify bird species in real-time."""
        try:
            import pyaudio
        except ImportError:
            print("Error: PyAudio not installed. Install it with:")
            print("  Linux: sudo apt-get install portaudio19-dev && pip install pyaudio")
            print("  macOS: brew install portaudio && pip install pyaudio")
            print("  Windows: pip install pyaudio")
            return
        
        print(f"\n🎤 Starting microphone recording...")
        print(f"Recording {duration} second chunks with {threshold} confidence threshold")
        print("Press Ctrl+C to stop...\n")
        
        # Audio parameters
        sample_rate = 16000  # Use lower sample rate for better compatibility
        chunk_size = 1024
        audio_format = pyaudio.paFloat32
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        try:
            # Start recording stream
            stream = audio.open(
                format=audio_format,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size
            )
            
            frames_needed = int(sample_rate * duration)
            
            while True:
                print(f"Recording {duration} seconds...")
                frames = []
                
                # Record for specified duration
                for _ in range(0, int(sample_rate / chunk_size * duration)):
                    data = stream.read(chunk_size, exception_on_overflow=False)
                    frames.append(data)
                
                # Convert to numpy array
                audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
                
                # Identify bird species
                try:
                    prediction, confidence, top_predictions = self.identify_from_array(
                        audio_data, sample_rate
                    )
                    
                    # Display results if above threshold
                    if confidence >= threshold:
                        self._display_prediction(prediction, confidence, top_predictions)
                    else:
                        print(f"Low confidence: {prediction} ({confidence:.3f})")
                
                except Exception as e:
                    print(f"Error processing audio: {e}")
                
                time.sleep(0.5)  # Brief pause between recordings
                
        except KeyboardInterrupt:
            print("\nStopping microphone recording...")
        
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            audio.terminate()
    
    def _display_prediction(self, prediction: str, confidence: float, top_predictions: Dict[str, float]):
        """Display prediction results."""
        print(f"\n{'='*60}")
        print(f"🐦 BIRD DETECTED!")
        print(f"{'='*60}")
        print(f"Species: {prediction}")
        print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        print(f"\nTop 5 predictions:")
        for i, (species, prob) in enumerate(top_predictions.items(), 1):
            print(f"  {i}. {species:<30} {prob:.3f} ({prob*100:.1f}%)")
        print(f"{'='*60}\n")


def find_best_mobilenet_model() -> str:
    """Find the best MobileNet model in the checkpoints directory."""
    checkpoint_dirs = [
        '../checkpoints_mobilenet',
        'checkpoints_mobilenet',
        './checkpoints_mobilenet'
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            continue
            
        # Look for 22-class models (September 5th models)
        models_22_class = list(checkpoint_path.glob("*20250905*_acc*.pth"))
        
        if models_22_class:
            # Sort by accuracy (extract from filename)
            def extract_accuracy(filename):
                try:
                    parts = filename.stem.split('_acc')
                    if len(parts) > 1:
                        return float(parts[1])
                    return 0.0
                except:
                    return 0.0
            
            best_model = max(models_22_class, key=extract_accuracy)
            accuracy = extract_accuracy(best_model)
            print(f"Found best 22-class model: {best_model.name} (accuracy: {accuracy:.4f})")
            return str(best_model)
        
        # Fallback to any best model files with accuracy in filename
        best_models = list(checkpoint_path.glob("best_model_*_acc*.pth"))
        
        if best_models:
            def extract_accuracy(filename):
                try:
                    parts = filename.stem.split('_acc')
                    if len(parts) > 1:
                        return float(parts[1])
                    return 0.0
                except:
                    return 0.0
            
            best_model = max(best_models, key=extract_accuracy)
            accuracy = extract_accuracy(best_model)
            print(f"Found best model: {best_model.name} (accuracy: {accuracy:.4f})")
            return str(best_model)
    
    raise FileNotFoundError("No MobileNet model checkpoints found!")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Simple Bird Species Identifier using MobileNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process an audio file
    python simple_bird_identifier.py --file bird_sound.wav
    
    # Listen to microphone
    python simple_bird_identifier.py --mic
    
    # Use custom model and threshold
    python simple_bird_identifier.py --model checkpoints_mobilenet/best_model.pth --file audio.wav --threshold 0.7
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to the trained model file (.pth). If not specified, will auto-detect best model.'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        default='./labels.json',
        help='Path to the labels JSON file (default: ../labels.json)'
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to audio file to process'
    )
    
    parser.add_argument(
        '--mic',
        action='store_true',
        help='Listen to microphone input'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=5.0,
        help='Duration of microphone recording chunks in seconds (default: 5.0)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.3,
        help='Confidence threshold for displaying predictions (default: 0.3)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device to use for inference (default: auto-detect)'
    )
    
    parser.add_argument(
        '--top_k', '-k',
        type=int,
        default=5,
        help='Number of top predictions to show (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Check that either file or mic is specified
    if not args.file and not args.mic:
        print("Error: Please specify either --file or --mic")
        parser.print_help()
        return 1
    
    # Auto-detect model if not specified
    if args.model is None:
        try:
            args.model = find_best_mobilenet_model()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please specify a model path using --model")
            return 1
    
    # Check if model file exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Check if labels file exists
    if not Path(args.labels).exists():
        print(f"Error: Labels file not found: {args.labels}")
        return 1
    
    try:
        # Initialize bird identifier
        identifier = SimpleBirdIdentifier(
            model_path=args.model,
            labels_path=args.labels,
            device=args.device
        )
        
        if args.file:
            # Process audio file
            if not Path(args.file).exists():
                print(f"Error: Audio file not found: {args.file}")
                return 1
            
            prediction, confidence, top_predictions = identifier.identify_from_file(
                args.file, args.top_k
            )
            identifier._display_prediction(prediction, confidence, top_predictions)
        
        elif args.mic:
            # Listen to microphone
            identifier.listen_microphone(args.duration, args.threshold)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
