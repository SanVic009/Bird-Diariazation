#!/usr/bin/env python3
"""
LIVE BIRD DETECTION SYSTEM
==========================
Real-time audio capture and bird species detection using your trained model.
"""

import torch
import torch.nn as nn
import numpy as np
import pyaudio
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time
import yaml
import queue
import warnings
warnings.filterwarnings('ignore')

def list_audio_devices():
    """List all available audio input devices."""
    p = pyaudio.PyAudio()
    info = []
    print("\nAvailable Audio Input Devices:")
    print("-" * 50)
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # if it has input channels
            info.append(dev_info)
            print(f"Device {i}: {dev_info['name']}")
            print(f"    Input channels: {dev_info['maxInputChannels']}")
            print(f"    Default sample rate: {dev_info['defaultSampleRate']}")
    p.terminate()
    return info

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

class MelCNN(nn.Module):
    def __init__(self, num_classes=74, cnn_width=64):
        super(MelCNN, self).__init__()
        
        self.backbone = nn.ModuleList([
            ConvBlock(1, cnn_width, 3, 1),           
            ConvBlock(cnn_width, cnn_width*2, 3, 1),  
            ConvBlock(cnn_width*2, cnn_width*2, 3, 1) 
        ])
        
        self.head = nn.Conv2d(cnn_width*2, num_classes, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        x = self.head(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x

class LiveBirdDetector:
    def __init__(self, model_path='outputs/checkpoints/full_model.pt', config_path='config.yaml'):
        """Initialize the live detection system."""
        """Initialize the live bird detector"""
        
        print("🔄 Initializing Live Bird Detector...")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Audio parameters
        self.sample_rate = self.config['audio']['sample_rate']  # 32000
        self.chunk_duration = 2.0  # seconds per analysis chunk
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.overlap = 0.5  # 50% overlap between chunks
        
        # Detection parameters
        self.detection_threshold = 0.001  # Based on our analysis
        self.min_confidence = 0.002  # Minimum confidence to report
        
        # Load species names
        self.species_names = list(self.config.get('classes', {}).keys())
        if not self.species_names:
            # Fallback species names
            self.species_names = [
                'asbfly', 'ashdro1', 'asikoe2', 'barswa', 'bcnher', 'bkskit1', 'bkwsti',
                'bladro1', 'blakit1', 'blhori1', 'blnmon1', 'blrwar1', 'brnhao1', 'brnshr',
                'brodro1', 'categr', 'comgre', 'comior1', 'comkin1', 'commoo3', 'commyn',
                'comros', 'comsan', 'comtai1', 'copbar1', 'crseag1', 'eaywag1', 'eucdov',
                'eurcoo', 'gargan', 'gloibi', 'graher1', 'grecou1', 'greegr', 'grewar3',
                'grnsan', 'grnwar1', 'grtdro1', 'grywag', 'gybpri1', 'gyhcaf1', 'hoopoe',
                'houcro1', 'houspa', 'kenplo1', 'labcro1', 'laudov1', 'lirplo', 'litegr',
                'litgre1', 'litspi1', 'nutman', 'piekin1', 'plapri1', 'purher1', 'pursun4',
                'putbab1', 'rerswa1', 'revbul', 'rewbul', 'rewlap1', 'rocpig', 'rorpar',
                'ruftre2', 'spodov', 'stbkin1', 'thbwar1', 'tibfly3', 'wemhar1', 'whbwat1',
                'whiter2', 'whtkin2', 'woosan', 'zitcis1'
            ]
        
        # Load trained model
        self.model = self._load_model(model_path)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.chunk_samples * 3)  # 6 seconds buffer
        self.detection_queue = queue.Queue()
        
        # Results storage
        self.recent_detections = deque(maxlen=10)  # Last 10 detections
        self.species_counts = {}
        
        # Audio stream
        self.audio = None
        self.stream = None
        self.is_running = False
        
        print("✅ Live Bird Detector initialized!")
        print(f"   🎤 Sample rate: {self.sample_rate} Hz")
        print(f"   ⏱️  Chunk duration: {self.chunk_duration} seconds")
        print(f"   🎯 Detection threshold: {self.detection_threshold}")
        print(f"   🐦 Species database: {len(self.species_names)} species")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            model = MelCNN(num_classes=74, cnn_width=64)
            model.load_state_dict(checkpoint['model'])
            model.eval()
            print(f"✅ Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"⚠️ Audio status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to buffer
        self.audio_buffer.extend(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def _process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio and detect birds"""
        try:
            # Create mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_chunk,
                sr=self.sample_rate,
                n_mels=128,
                fmax=14000,
                fmin=50,
                hop_length=512,
                n_fft=2048
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to model input size (128x128)
            if mel_spec_db.shape[1] != 128:
                mel_spec_resized = librosa.util.fix_length(mel_spec_db.T, size=128, axis=0).T
            else:
                mel_spec_resized = mel_spec_db
            
            # Normalize
            mel_spec_norm = (mel_spec_resized - mel_spec_resized.mean()) / (mel_spec_resized.std() + 1e-8)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.sigmoid(outputs).numpy()[0]
            
            # Find detections above threshold
            detections = []
            for i, prob in enumerate(probabilities):
                if prob >= self.detection_threshold and i < len(self.species_names):
                    detections.append({
                        'species': self.species_names[i],
                        'confidence': prob,
                        'timestamp': time.time()
                    })
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detections
            
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            return []
    
    def _detection_worker(self):
        """Worker thread for audio processing"""
        while self.is_running:
            try:
                # Check if we have enough audio data
                if len(self.audio_buffer) >= self.chunk_samples:
                    # Get chunk from buffer
                    audio_chunk = np.array(list(self.audio_buffer)[-self.chunk_samples:])
                    
                    # Process chunk
                    detections = self._process_audio_chunk(audio_chunk)
                    
                    if detections:
                        # Update counts
                        for detection in detections:
                            species = detection['species']
                            if species not in self.species_counts:
                                self.species_counts[species] = 0
                            self.species_counts[species] += 1
                        
                        # Store recent detection
                        self.recent_detections.append({
                            'detections': detections,
                            'timestamp': time.time()
                        })
                        
                        # Print live results
                        self._print_detections(detections)
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                print(f"❌ Detection worker error: {e}")
                time.sleep(1)
    
    def _print_detections(self, detections):
        """Print detection results to console"""
        if not detections:
            return
        
        timestamp = time.strftime("%H:%M:%S")
        print(f"\n🐦 [{timestamp}] BIRD DETECTION:")
        
        high_confidence = [d for d in detections if d['confidence'] >= self.min_confidence]
        
        if high_confidence:
            for i, detection in enumerate(high_confidence[:5]):  # Show top 5
                confidence_pct = detection['confidence'] * 100
                species = detection['species'].upper()
                print(f"   {i+1}. {species} - {confidence_pct:.3f}% confidence")
        else:
            print("   🔍 Low confidence detections (background noise)")
    
    def start_detection(self):
        """Start live detection"""
        try:
            print("\n🎤 Starting live bird detection...")
            
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self._audio_callback
            )
            
            print("✅ Audio stream started!")
            print("🔄 Starting detection worker...")
            
            # Start detection worker thread
            self.is_running = True
            detection_thread = threading.Thread(target=self._detection_worker)
            detection_thread.daemon = True
            detection_thread.start()
            
            # Start audio stream
            self.stream.start_stream()
            
            print("\n🎯 LIVE DETECTION ACTIVE!")
            print("=" * 40)
            print("🎤 Listening for bird sounds...")
            print("⏹️  Press Ctrl+C to stop")
            print("=" * 40)
            
            # Keep running until interrupted
            try:
                while self.is_running:
                    time.sleep(1)
                    
                    # Print summary every 30 seconds
                    if int(time.time()) % 30 == 0:
                        self._print_summary()
                        
            except KeyboardInterrupt:
                print("\n⏹️ Stopping detection...")
                self.stop_detection()
                
        except Exception as e:
            print(f"❌ Error starting detection: {e}")
            self.stop_detection()
    
    def stop_detection(self):
        """Stop live detection"""
        self.is_running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("✅ Detection stopped!")
        self._print_final_summary()
    
    def _print_summary(self):
        """Print periodic summary"""
        if not self.species_counts:
            return
        
        print(f"\n📊 DETECTION SUMMARY (Total species detected: {len(self.species_counts)})")
        
        # Sort by count
        sorted_species = sorted(self.species_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i, (species, count) in enumerate(sorted_species[:10]):  # Top 10
            print(f"   {i+1}. {species.upper()}: {count} detections")
    
    def _print_final_summary(self):
        """Print final detection summary"""
        print(f"\n🏁 FINAL DETECTION SUMMARY")
        print("=" * 30)
        
        if self.species_counts:
            total_detections = sum(self.species_counts.values())
            unique_species = len(self.species_counts)
            
            print(f"📊 Total Statistics:")
            print(f"   🐦 Unique species detected: {unique_species}")
            print(f"   📈 Total detection events: {total_detections}")
            
            print(f"\n🏆 Top Species Detected:")
            sorted_species = sorted(self.species_counts.items(), key=lambda x: x[1], reverse=True)
            
            for i, (species, count) in enumerate(sorted_species[:15]):
                percentage = (count / total_detections) * 100
                print(f"   {i+1}. {species.upper()}: {count} times ({percentage:.1f}%)")
        else:
            print("   🔍 No bird species detected during this session")
        
        print("\n✅ Session completed!")

def main():
    """Main function to run live detection"""
    print("🎯 LIVE BIRD DETECTION SYSTEM")
    print("=" * 35)
    
    # Create detector
    detector = LiveBirdDetector()
    
    if detector.model is None:
        print("❌ Cannot start without trained model")
        return
    
    # Start detection
    detector.start_detection()

if __name__ == "__main__":
    main()
