#!/usr/bin/env python3
"""
Real-time Bird Species Identifier using Microphone Input
========================================================

This script captures audio from the microphone in real-time and identifies bird species
using a pre-trained ResNet-GRU model. It continuously processes audio chunks and displays
the predicted bird species with confidence scores.

Requirements:
- PyAudio for microphone capture
- torch, torchaudio for model inference
- librosa for audio processing
- numpy for numerical operations

Usage:
    python real_time_bird_identifier.py [--model_path PATH] [--duration SECONDS] [--threshold CONFIDENCE]

Author: GitHub Copilot
Date: September 2025
"""

import os
import sys
import json
import time
import queue
import threading
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import pyaudio

# Add v4 directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from resnet_gru import ResNetGRUBird
from mobilenet_gru import MobileNetGRUBird
from mobilenet import MobileNetBird
from preprocessing import BirdPreprocessor


def detect_model_architecture(checkpoint_path: str) -> str:
    """Detect the model architecture from checkpoint keys."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Check for architecture-specific keys
    keys = list(state_dict.keys())
    
    # Check for specific patterns
    if any('mobilenet.features' in key for key in keys):
        # This is a plain MobileNet model (not MobileNet-GRU)
        return 'mobilenet'
    elif any('mobilenet' in key for key in keys):
        return 'mobilenet_gru'
    elif any('feature_extractor' in key for key in keys):
        return 'resnet_gru'
    elif any('resnet' in key for key in keys):
        return 'resnet'
    elif any('lstm' in key for key in keys):
        return 'lstm'
    else:
        # Default fallback
        print(f"Warning: Could not detect architecture from keys: {keys[:5]}...")
        return 'resnet_gru'


def load_model_by_architecture(architecture: str, num_classes: int, checkpoint_path: str, device: torch.device):
    """Load the appropriate model based on detected architecture."""
    
    if architecture == 'mobilenet':
        model = MobileNetBird(n_classes=num_classes)
    elif architecture == 'mobilenet_gru':
        model = MobileNetGRUBird(n_classes=num_classes)
    elif architecture == 'resnet_gru':
        model = ResNetGRUBird(n_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        # Try with strict=False
        model.load_state_dict(state_dict, strict=False)
        print("Loaded with strict=False (some parameters may be missing)")
    
    model.to(device)
    model.eval()
    
    return model


class RealTimeBirdIdentifier:
    """Real-time bird species identification system using microphone input."""
    
    def __init__(
        self,
        model_path: str,
        labels_path: str = "../labels.json",
        sample_rate: int = 32000,
        chunk_duration: float = 5.0,
        overlap_duration: float = 2.5,
        confidence_threshold: float = 0.3,
        device: Optional[str] = None
    ):
        """
        Initialize the real-time bird identifier.
        
        Args:
            model_path: Path to the trained PyTorch model (.pth file)
            labels_path: Path to the labels JSON file
            sample_rate: Audio sample rate (should match training)
            chunk_duration: Duration of audio chunks to process (seconds)
            overlap_duration: Overlap between consecutive chunks (seconds)
            confidence_threshold: Minimum confidence for predictions
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect)
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load labels
        self.labels = self._load_labels(labels_path)
        self.num_classes = len(self.labels)
        print(f"Loaded {self.num_classes} bird species labels")
        
        # Initialize preprocessor
        self.preprocessor = BirdPreprocessor(
            sample_rate=sample_rate,
            duration_strategy="fixed",
            fixed_duration=chunk_duration,
            augment_prob=0.0  # No augmentation for inference
        )
        
        # Load model
        self.model = self._load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Audio capture setup
        self.chunk_size = int(sample_rate * chunk_duration)
        self.overlap_size = int(sample_rate * overlap_duration)
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=self.chunk_size)
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=5)
        
        # Control flags
        self.running = False
        self.processing = False
    
    def _load_labels(self, labels_path: str) -> List[str]:
        """Load bird species labels from JSON file."""
        with open(labels_path, 'r') as f:
            data = json.load(f)
            return data['labels']
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained model with automatic architecture detection."""
        # Detect architecture from checkpoint
        architecture = detect_model_architecture(model_path)
        print(f"Detected model architecture: {architecture}")
        
        # Load model with appropriate architecture
        model = load_model_by_architecture(architecture, self.num_classes, model_path, self.device)
        
        return model
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to queue for processing
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            # If queue is full, remove oldest item and add new one
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(audio_data)
            except queue.Empty:
                pass
        
        return (in_data, pyaudio.paContinue)
    
    def _preprocess_audio(self, audio_chunk: np.ndarray) -> torch.Tensor:
        """Preprocess audio chunk for model inference."""
        # Ensure correct duration
        processed_chunks = self.preprocessor.process_duration(audio_chunk)
        audio_chunk = processed_chunks[0]  # Take first chunk
        
        # Convert to mel spectrogram
        log_mel = self.preprocessor.to_log_mel(audio_chunk)
        
        # Convert to tensor and add batch dimension
        spectrogram = torch.from_numpy(log_mel).float()
        spectrogram = spectrogram.unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
        
        return spectrogram.to(self.device)
    
    def _predict(self, spectrogram: torch.Tensor) -> Tuple[str, float, Dict[str, float]]:
        """Make prediction on preprocessed audio."""
        with torch.no_grad():
            logits = self.model(spectrogram)
            probabilities = F.softmax(logits, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)
            
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
    
    def _smooth_predictions(self, prediction: str, confidence: float) -> Tuple[str, float]:
        """Apply temporal smoothing to predictions."""
        self.prediction_history.append((prediction, confidence))
        
        if len(self.prediction_history) < 3:
            return prediction, confidence
        
        # Count occurrences of each prediction
        prediction_counts = {}
        total_confidence = {}
        
        for pred, conf in self.prediction_history:
            if pred not in prediction_counts:
                prediction_counts[pred] = 0
                total_confidence[pred] = 0.0
            prediction_counts[pred] += 1
            total_confidence[pred] += conf
        
        # Get most frequent prediction
        most_frequent = max(prediction_counts.keys(), key=lambda x: prediction_counts[x])
        avg_confidence = total_confidence[most_frequent] / prediction_counts[most_frequent]
        
        return most_frequent, avg_confidence
    
    def _audio_processing_thread(self):
        """Thread function for processing audio chunks."""
        while self.running:
            try:
                # Get audio data from queue (with timeout)
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Add to buffer
                self.audio_buffer.extend(audio_data)
                
                # Process when we have enough data
                if len(self.audio_buffer) >= self.chunk_size:
                    # Extract chunk
                    audio_chunk = np.array(list(self.audio_buffer)[:self.chunk_size])
                    
                    # Remove overlap amount from buffer
                    for _ in range(min(self.overlap_size, len(self.audio_buffer))):
                        self.audio_buffer.popleft()
                    
                    # Process audio chunk
                    self.processing = True
                    try:
                        # Preprocess
                        spectrogram = self._preprocess_audio(audio_chunk)
                        
                        # Predict
                        prediction, confidence, top_predictions = self._predict(spectrogram)
                        
                        # Apply smoothing
                        smooth_prediction, smooth_confidence = self._smooth_predictions(
                            prediction, confidence
                        )
                        
                        # Display results if above threshold
                        if smooth_confidence >= self.confidence_threshold:
                            self._display_prediction(
                                smooth_prediction, smooth_confidence, top_predictions
                            )
                        else:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                                  f"Low confidence detection: {prediction} ({confidence:.3f})")
                            
                    except Exception as e:
                        print(f"Error processing audio: {e}")
                    finally:
                        self.processing = False
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing thread: {e}")
                break
    
    def _display_prediction(self, prediction: str, confidence: float, top_predictions: Dict[str, float]):
        """Display prediction results."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        print(f"\n{'='*60}")
        print(f"[{timestamp}] 🐦 BIRD DETECTED!")
        print(f"{'='*60}")
        print(f"Species: {prediction}")
        print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        print(f"\nTop 5 predictions:")
        for i, (species, prob) in enumerate(top_predictions.items(), 1):
            print(f"  {i}. {species:<15} {prob:.3f} ({prob*100:.1f}%)")
        print(f"{'='*60}\n")
    
    def start_listening(self):
        """Start real-time audio capture and processing."""
        print("Initializing audio capture...")
        
        # Start PyAudio stream
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self._audio_callback
        )
        
        # Start processing thread
        self.running = True
        processing_thread = threading.Thread(target=self._audio_processing_thread)
        processing_thread.daemon = True
        processing_thread.start()
        
        # Start stream
        self.stream.start_stream()
        
        print(f"\n🎤 Listening for bird sounds...")
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Chunk Duration: {self.chunk_duration}s")
        print(f"Confidence Threshold: {self.confidence_threshold}")
        print(f"Device: {self.device}")
        print("\nPress Ctrl+C to stop...\n")
        
        try:
            while True:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nStopping bird identifier...")
            self.stop_listening()
    
    def stop_listening(self):
        """Stop audio capture and processing."""
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("Bird identifier stopped.")


def find_best_model(checkpoints_dir: str) -> str:
    """Find the best model file in the checkpoints directory."""
    checkpoint_path = Path(checkpoints_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # Look for best model files with accuracy in filename
    best_models = list(checkpoint_path.glob("best_model_*_acc*.pth"))
    
    if best_models:
        # Sort by accuracy (extract from filename)
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
    
    # Fallback to generic best_model.pth
    generic_best = checkpoint_path / "best_model.pth"
    if generic_best.exists():
        print(f"Using generic best model: {generic_best.name}")
        return str(generic_best)
    
    # List all available models
    all_models = list(checkpoint_path.glob("*.pth"))
    if not all_models:
        raise FileNotFoundError(f"No model files found in {checkpoints_dir}")
    
    print(f"Available models in {checkpoints_dir}:")
    for model in all_models:
        print(f"  - {model.name}")
    
    # Use the first one as fallback
    fallback_model = all_models[0]
    print(f"Using fallback model: {fallback_model.name}")
    return str(fallback_model)


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Real-time Bird Species Identifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default ResNet-GRU model
    python real_time_bird_identifier.py
    
    # Specify custom model path
    python real_time_bird_identifier.py --model_path ../checkpoints_resnet_gru/best_model.pth
    
    # Adjust confidence threshold
    python real_time_bird_identifier.py --threshold 0.5
    
    # Use longer audio chunks
    python real_time_bird_identifier.py --duration 10.0
        """
    )
    
    parser.add_argument(
        '--model_path', '-m',
        type=str,
        default=None,
        help='Path to the trained model file (.pth). If not specified, will auto-detect best model.'
    )
    
    parser.add_argument(
        '--labels_path', '-l',
        type=str,
        default='./labels.json',
        help='Path to the labels JSON file (default: ../labels.json)'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=5.0,
        help='Duration of audio chunks to process in seconds (default: 5.0)'
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
        '--sample_rate', '-sr',
        type=int,
        default=32000,
        help='Audio sample rate (default: 32000)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect model if not specified
    if args.model_path is None:
        # Try different checkpoint directories in order of preference
        checkpoint_dirs = [
            '../checkpoints_mobilenet',
            '../checkpoints_mobilenet_gru', 
            '../checkpoints_resnet_gru',
            '../checkpoints_resnet',
            '../checkpoints_lstm'
        ]
        
        model_path = None
        for checkpoint_dir in checkpoint_dirs:
            try:
                model_path = find_best_model(checkpoint_dir)
                break
            except FileNotFoundError:
                continue
        
        if model_path is None:
            print("Error: No model checkpoints found!")
            print("Please specify a model path using --model_path")
            return 1
    else:
        model_path = args.model_path
    
    # Check if model file exists
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        return 1
    
    # Check if labels file exists
    if not Path(args.labels_path).exists():
        print(f"Error: Labels file not found: {args.labels_path}")
        return 1
    
    # Check PyAudio availability
    try:
        import pyaudio
    except ImportError:
        print("Error: PyAudio not installed. Install it with:")
        print("  Linux: sudo apt-get install portaudio19-dev && pip install pyaudio")
        print("  macOS: brew install portaudio && pip install pyaudio")  
        print("  Windows: pip install pyaudio")
        return 1
    
    try:
        # Initialize bird identifier
        identifier = RealTimeBirdIdentifier(
            model_path=model_path,
            labels_path=args.labels_path,
            sample_rate=args.sample_rate,
            chunk_duration=args.duration,
            confidence_threshold=args.threshold,
            device=args.device
        )
        
        # Start listening
        identifier.start_listening()
        
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
