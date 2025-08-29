#!/usr/bin/env python3
"""
DEBUG VERSION - LIVE BIRD DETECTION SYSTEM
========================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sounddevice as sd
import librosa
import matplotlib.pyplot as plt
import time
import yaml
import queue
import warnings
import argparse
import sys
from pathlib import Path
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
    
    def forward(self, x):
        h = self.backbone(x)
        h = h.mean(dim=2, keepdim=True)
        logits = self.head(h).squeeze(2).transpose(1,2)
        return logits

class BirdDetector:
    def __init__(self, model_path='outputs/checkpoints/full_model.pt', config_path='config.yaml',
                 device_index=None, sample_rate=None):
        print("\n🔍 Initializing Bird Detector with debugging...")
        
        # Load configs
        print("Loading config file...")
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        print("Config loaded successfully")
        
        # Audio settings
        self.device_index = device_index
        # Get default audio device info first
        devices = sd.query_devices()
        if device_index is None:
            device_index = sd.default.device[0]
        device_info = devices[device_index]
        
        # Use device's default sample rate if none specified
        self.sample_rate = sample_rate or int(device_info['default_samplerate'])
        self.chunk_duration = 2.0
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        device_info = devices[device_index]
        print(f"\n🎤 Audio Device Information:")
        print(f"Name: {device_info['name']}")
        print(f"Channels: {device_info['max_input_channels']}")
        print(f"Default Sample Rate: {device_info['default_samplerate']} Hz")
        print(f"Selected Sample Rate: {self.sample_rate} Hz")
        
        # Model loading
        print("\n📂 Loading model...")
        self.model = self._load_model(model_path)
        print("Model architecture:")
        print(self.model)
        
        self.threshold = 0.001
        self.running = False
        self.audio_queue = queue.Queue()
        
        print("\n✅ Initialization complete!")
        print(f"🎯 Detection threshold: {self.threshold}")
        print(f"🔊 Audio chunk size: {self.chunk_samples} samples")
        print(f"⏱️ Chunk duration: {self.chunk_duration} seconds")

    def _load_model(self, model_path):
        try:
            print(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
            
            n_mels = self.cfg['audio']['n_mels']
            model = MelCNN(n_mels=n_mels, n_classes=74, width=64, dropout=0.2).to('cpu')
            
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    print("Loading state_dict from checkpoint...")
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model' in checkpoint and isinstance(checkpoint['model'], nn.Module):
                    print("Loading model directly from checkpoint...")
                    model = checkpoint['model']
                elif isinstance(checkpoint, dict):
                    print("Attempting to load checkpoint as state_dict...")
                    try:
                        model.load_state_dict(checkpoint)
                    except Exception as e:
                        print(f"Warning: Could not load checkpoint as state_dict: {e}")
            
            model.eval()
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"\n⚠️ Audio callback status: {status}")
        
        # Debug audio input
        if indata is not None:
            rms_level = np.sqrt(np.mean(indata**2))
            peak_level = np.max(np.abs(indata))
            meter = "█" * int(peak_level * 50)
            print(f"\r🎤 Level: {meter:<50} RMS: {rms_level:.3f} Peak: {peak_level:.3f}", end="", flush=True)
            
            if rms_level < 1e-6:
                print("\r⚠️ Warning: Very low audio input level", flush=True)
            
            self.audio_queue.put(indata.copy())
        else:
            print("\r⚠️ Warning: Received None as audio input", flush=True)

    def process_audio(self, audio):
        """Process audio chunk with detailed debugging."""
        print("\n\n" + "="*50, flush=True)
        print("🔄 PROCESSING AUDIO CHUNK - " + time.strftime('%H:%M:%S'), flush=True)
        print("="*50, flush=True)
        
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            print(f"Converted to mono. New shape: {audio.shape}")
        
        try:
            # Compute mel spectrogram
            print("\nComputing mel spectrogram...")
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.cfg['audio']['n_mels'],
                fmin=self.cfg['audio']['fmin'],
                fmax=self.cfg['audio']['fmax']
            )
            print(f"Mel spectrogram shape: {mel_spec.shape}")
            
            # Convert to dB and normalize
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
            print(f"Normalized mel-db range: [{mel_db.min():.3f}, {mel_db.max():.3f}]")
            
            # Prepare tensor for model
            x = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)
            print(f"Model input tensor shape: {x.shape}")
            
            # Get predictions
            with torch.no_grad():
                logits = self.model(x)  # Shape: [B, T, C]
                predictions = torch.sigmoid(logits)
                
                # Average predictions over time
                predictions = predictions.mean(dim=1).squeeze()  # Shape: [C]
                
            print("\n📊 PREDICTION SUMMARY")
            print("-"*50)
            top_k = torch.topk(predictions, min(3, len(predictions)))
            for i, (conf, idx) in enumerate(zip(top_k.values, top_k.indices)):
                confidence = conf.item()
                confidence_bar = "█" * int(confidence * 40)
                print(f"Top {i+1}: Species {idx.item():3d} | {confidence_bar} {confidence:.3f}", flush=True)
                print(f"Level: |{confidence_bar:<40}|")
                print("-"*50)
            
            return predictions.cpu().numpy()
            
        except Exception as e:
            print(f"\n❌ Error in audio processing: {str(e)}")
            raise

    def start_detection(self):
        try:
            print("\n🎯 Starting detection with debugging enabled...")
            self.running = True
            
            print("\nConfiguring audio stream:")
            print(f"Device index: {self.device_index}")
            print(f"Sample rate: {self.sample_rate}")
            print(f"Channels: 1")
            print(f"Block size: {self.chunk_samples}")
            
            self.stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples,
                callback=self.audio_callback
            )
            
            print("\n🎤 Starting audio stream...")
            self.stream.start()
            
            print("\n👂 Listening for birds...")
            while self.running:
                try:
                    audio_chunk = self.audio_queue.get(timeout=1)
                    predictions = self.process_audio(audio_chunk)
                    
                    # Show predictions
                    detections = np.where(predictions > self.threshold)[0]
                    if len(detections) > 0:
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"\n🔍 Detections at {timestamp}:")
                        
                        # Sort detections by confidence
                        detection_confidences = [(idx, predictions[idx]) for idx in detections]
                        detection_confidences.sort(key=lambda x: x[1], reverse=True)
                        
                        for idx, confidence in detection_confidences:
                            confidence_bar = "█" * int(confidence * 20)
                            print(f"   Species {idx:3d}: {confidence:.3f} |{confidence_bar:<20}|")
                    else:
                        print("\n💫 No detections above threshold...")
                        print("\n" + "="*50)  # Add a separator line
                
                except queue.Empty:
                    print("\n⏳ Waiting for audio...")
                    time.sleep(0.5)  # Add a small delay to prevent too much output
                    continue
                
        except Exception as e:
            print(f"\n❌ Error in detection: {str(e)}")
        finally:
            self.stop_detection()

    def stop_detection(self):
        print("\n\n🛑 Stopping detection...")
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("✅ Detection stopped!")

def main():
    parser = argparse.ArgumentParser(description="Live Bird Detection System (Debug Version)")
    parser.add_argument("--model", default="outputs/checkpoints/full_model.pt", help="Path to model file")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--device", type=int, help="Audio device index")
    parser.add_argument("--sample-rate", type=int, help="Sample rate in Hz")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    
    args = parser.parse_args()
    
    if args.list_devices:
        print("\n🎤 Available Audio Devices:")
        print("-" * 50)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"Device {i}: {dev['name']}")
                print(f"    Max input channels: {dev['max_input_channels']}")
                print(f"    Default sample rate: {dev['default_samplerate']} Hz\n")
        return
    
    try:
        detector = BirdDetector(
            model_path=args.model,
            config_path=args.config,
            device_index=args.device,
            sample_rate=args.sample_rate
        )
        
        detector.start_detection()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
    finally:
        if 'detector' in locals():
            detector.stop_detection()

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Live Bird Detection with Debug Output')
        parser.add_argument('--model', type=str, default='outputs/checkpoints/full_model.pt',
                          help='Path to the model checkpoint')
        parser.add_argument('--config', type=str, default='config.yaml',
                          help='Path to the configuration file')
        parser.add_argument('--device', type=int, help='Audio input device index')
        parser.add_argument('--list-devices', action='store_true', help='List available audio devices')
        args = parser.parse_args()
        
        if args.list_devices:
            print("\nAvailable audio input devices:")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"{i}: {device['name']}")
                    print(f"   Channels: {device['max_input_channels']}")
                    print(f"   Sample Rate: {device['default_samplerate']}")
            sys.exit(0)
        
        detector = BirdDetector(model_path=args.model, config_path=args.config, device_index=args.device)
        detector.start_detection()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
    finally:
        if 'detector' in locals():
            detector.stop_detection()
