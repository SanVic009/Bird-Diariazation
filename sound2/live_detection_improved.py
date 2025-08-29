#!/usr/bin/env python3
"""
LIVE BIRD DETECTION SYSTEM
==========================
Real-time audio capture and bird species detection using your trained model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time
import yaml
import queue
import warnings
import argparse
warnings.filterwarnings('ignore')

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d((2,1))
        self.drop = nn.Dropout(p)
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        x = self.drop(x)
        return x

class MelCNN(nn.Module):
    def __init__(self, n_mels=128, n_classes=10, width=64, dropout=0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, width, dropout),
            ConvBlock(width, width*2, dropout),
            ConvBlock(width*2, width*2, dropout),
        )
        self.head = nn.Conv2d(width*2, n_classes, kernel_size=1)
    
    def forward(self, x):  # x: [B,1,n_mels,T]
        h = self.backbone(x)            # [B, C, n_mels/8, T]
        h = h.mean(dim=2, keepdim=True) # pool mel -> [B, C, 1, T]
        logits = self.head(h).squeeze(2).transpose(1,2)  # [B, T, C]
        return logits  # frame-wise logits

class BirdDetector:
    def __init__(self, model_path='outputs/checkpoints/full_model.pt', config_path='config.yaml',
                 device_index=None, sample_rate=None):
        """Initialize the live detection system."""
        # Load configs
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Audio settings
        self.device_index = device_index
        self.sample_rate = sample_rate or self.cfg['audio']['sample_rate']
        self.chunk_duration = 2.0  # seconds
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Get available devices and sample rates
        devices = sd.query_devices()
        if device_index is None:
            device_index = sd.default.device[0]
        
        device_info = devices[device_index]
        print(f"\nUsing audio device: {device_info['name']}")
        
        # Check supported sample rates
        try:
            supported_rates = sd.query_devices(device_index)['default_samplerate']
            print(f"Default sample rate: {supported_rates} Hz")
            if self.sample_rate != supported_rates:
                print(f"⚠️  Adjusting sample rate from {self.sample_rate} to {supported_rates} Hz")
                self.sample_rate = int(supported_rates)
                self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        except Exception as e:
            print(f"Warning: Could not query sample rates: {e}")
        
        # Model and processing
        self.model = self._load_model(model_path)
        self.threshold = 0.0001  # Lower threshold for testing
        print("\nModel output shape:", self.model.head.weight.shape)  # Debug info
        
        # Detection state
        self.running = False
        self.audio_queue = queue.Queue()
        self.detected_species = {}  # Dictionary to track detections {species_id: [(timestamp, confidence), ...]}
        self.last_detection_time = {}  # To prevent too frequent detections of the same species
        self.detection_cooldown = 2.0  # Minimum seconds between detections of the same species
        
        print("\n✅ Live Bird Detector initialized!")
        print(f"   🎤 Sample rate: {self.sample_rate} Hz")
        print(f"   ⏱️  Chunk duration: {self.chunk_duration} seconds")
        print(f"   🎯 Detection threshold: {self.threshold}")
        print(f"   🐦 Species database: {self.model.head.out_channels} species")

    def _load_model(self, model_path):
        """Load the bird detection model."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Initialize the model architecture
            n_mels = self.cfg['audio']['n_mels']
            model = MelCNN(n_mels=n_mels, n_classes=74, width=64, dropout=0.2).to('cpu')
            
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model' in checkpoint and isinstance(checkpoint['model'], nn.Module):
                    model = checkpoint['model']
                elif isinstance(checkpoint, (dict, torch.nn.Module)):
                    try:
                        if isinstance(checkpoint, dict):
                            model.load_state_dict(checkpoint)
                        else:
                            model = checkpoint
                    except Exception as e:
                        print(f"Warning: Could not load checkpoint directly: {e}")
            
            model.eval()
            print(f"✅ Model loaded from {model_path}")
            return model
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream to process incoming audio chunks."""
        if status:
            print(f"\nStatus: {status}")
        
        # Check if we're getting audio
        audio_level = np.abs(indata).mean()
        if audio_level < 1e-6:
            print("\rWarning: Very low audio level detected!", end="")
        else:
            print(f"\rAudio input level: {audio_level:.6f}", end="")
            
        self.audio_queue.put(indata.copy())

    def process_audio(self, audio):
        """Process audio chunk and return detections."""
        try:
            # Print audio stats
            print(f"\rAudio shape: {audio.shape}, dtype: {audio.dtype}, range: [{audio.min():.3f}, {audio.max():.3f}]", end="")
            
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.cfg['audio']['n_mels'],
                fmin=self.cfg['audio']['fmin'],
                fmax=self.cfg['audio']['fmax']
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
            
            # Print mel spectrogram stats
            print(f"\rMel shape: {mel_db.shape}, range: [{mel_db.min():.3f}, {mel_db.max():.3f}]", end="")
            
            # Prepare for model
            x = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)
            
            # Get predictions
            with torch.no_grad():
                predictions = torch.sigmoid(self.model(x)).squeeze()
                print(f"\rPredictions shape: {predictions.shape}, range: [{predictions.min():.3f}, {predictions.max():.3f}]", end="")
            
            return predictions.numpy()
        except Exception as e:
            print(f"\nError in process_audio: {str(e)}")
            return np.zeros(74)  # Return zeros in case of error

    def start_detection(self):
        """Start real-time bird detection."""
        try:
            self.running = True
            
            # Configure and start the audio stream
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples,
                callback=self.audio_callback
            )
            
            self.stream.start()
            print("\n🎤 Starting live bird detection...")
            
            while self.running:
                try:
                    audio_chunk = self.audio_queue.get(timeout=1)
                    predictions = self.process_audio(audio_chunk)
                    
                    # Calculate audio level for visualization
                    audio_level = np.abs(audio_chunk).mean()
                    max_level = 0.1  # Adjust this based on your audio input
                    level_bars = int((audio_level / max_level) * 20)
                    level_bars = min(20, level_bars)
                    
                    # Create level meter
                    meter = "▐" + "█" * level_bars + "░" * (20 - level_bars) + "▌"
                    
                    # Clear previous line and show audio level
                    print("\033[K", end="")  # Clear line
                    print(f"\r🎤 Audio Level: {meter}", end=" ")
                    
                    # Update detections
                    detections = np.where(predictions > self.threshold)[0]
                    if len(detections) > 0:
                        timestamp = time.strftime("%H:%M:%S")
                        print("\n", end="")  # New line for detections
                        for idx in detections:
                            confidence = predictions[idx]
                            conf_bars = int(confidence * 20)
                            conf_meter = "▐" + "█" * conf_bars + "░" * (20 - conf_bars) + "▌"
                            print(f"🔍 {timestamp} - Species {idx:3d} | Confidence: {conf_meter} ({confidence:.3f})")
                        print("\n", end="")  # Add space after detections
                    
                    # Update maximum predictions for visualization
                    top_predictions = np.argsort(predictions)[-3:][::-1]  # Top 3 predictions
                    print("\033[K", end="")  # Clear line
                    print(f"\r📊 Top predictions:", end=" ")
                    for idx in top_predictions:
                        if predictions[idx] > 0.1:  # Only show if confidence > 0.1
                            print(f"Species {idx:3d}: {predictions[idx]:.3f}", end=" | ")
                    print("", end="\r")
                
                except queue.Empty:
                    # Show a simple spinner when waiting for audio
                    spinner = "|/-\\"
                    print(f"\r⏳ Listening... {spinner[int(time.time() * 2) % 4]}", end="")
                    continue
                except Exception as e:
                    print(f"Error processing audio: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"❌ Error starting detection: {str(e)}")
        finally:
            self.stop_detection()

    def stop_detection(self):
        """Stop the detection system."""
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("\n\n✅ Detection stopped!")
        
        # Print session summary if there were any detections
        if hasattr(self, 'detected_species') and self.detected_species:
            print("\n📊 Session Summary:")
            print("-" * 50)
            for species_id, detections in self.detected_species.items():
                count = len(detections)
                avg_conf = sum(conf for _, conf in detections) / count
                print(f"Species {species_id:3d}: {count:3d} detections (avg conf: {avg_conf:.3f})")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="Live Bird Detection System")
    parser.add_argument("--model", default="outputs/checkpoints/full_model.pt", help="Path to model file")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--device", type=int, help="Audio device index")
    parser.add_argument("--sample-rate", type=int, help="Sample rate in Hz")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    
    args = parser.parse_args()
    
    if args.list_devices:
        print("\nAvailable audio devices:")
        print("-" * 50)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"Device {i}: {dev['name']}")
                print(f"    Max input channels: {dev['max_input_channels']}")
                print(f"    Default sample rate: {dev['default_samplerate']} Hz")
        return
    
    detector = BirdDetector(
        model_path=args.model,
        config_path=args.config,
        device_index=args.device,
        sample_rate=args.sample_rate
    )
    
    try:
        detector.start_detection()
    except KeyboardInterrupt:
        print("\n⚠️  Stopping detection...")
    finally:
        detector.stop_detection()

if __name__ == "__main__":
    main()
