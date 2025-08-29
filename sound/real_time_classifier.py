#!/usr/bin/env python3
"""
Real-time Bird Sound Classification
==================================

This script provides real-time bird sound classification using the trained model.
It can process live audio from microphone or audio files.

Features:
- Real-time microphone input
- Audio file processing
- Top-K species predictions
- Confidence scores
- Species information display

Author: AI Assistant
Date: 2025
"""

import os
import sys
import time
import numpy as np
import torch
import librosa
import sounddevice as sd
import noisereduce as nr
from bird_classifier import BirdClassifier, AudioDataset
import logging
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealTimeBirdClassifier:
    """Real-time bird sound classifier"""
    
    def __init__(self, model_path):
        """Initialize the real-time classifier"""
        self.classifier = BirdClassifier(".")
        
        # Load the trained model
        logger.info(f"Loading model from {model_path}")
        self.classifier.load_model(model_path)
        logger.info("Model loaded successfully!")
        
        # Audio recording parameters
        self.sample_rate = self.classifier.config['audio']['sample_rate']
        self.duration = self.classifier.config['audio']['max_len']  # seconds
        
    def classify_audio_file(self, audio_path, top_k=5):
        """Classify a single audio file"""
        logger.info(f"Classifying audio file: {audio_path}")
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None
        
        start_time = time.time()
        results = self.classifier.predict_single_audio(audio_path, top_k=top_k)
        inference_time = time.time() - start_time
        
        if results:
            logger.info(f"Classification completed in {inference_time:.3f} seconds")
            self.display_results(results)
        else:
            logger.error("Classification failed")
        
        return results
    
    def record_and_classify(self, duration=None, top_k=5):
        """Record audio from microphone and classify"""
        if duration is None:
            duration = self.duration
        
        logger.info(f"Recording for {duration} seconds...")
        logger.info("Speak near the microphone...")
        
        try:
            # Record audio
            audio = sd.rec(
                int(duration * self.sample_rate), 
                samplerate=self.sample_rate, 
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Wait until recording is finished
            
            logger.info("Recording complete. Processing...")
            
            # Save temporary file
            temp_path = "temp_recording.wav"
            librosa.output.write_wav(temp_path, audio.flatten(), self.sample_rate)
            
            # Classify
            start_time = time.time()
            results = self.classifier.predict_single_audio(temp_path, top_k=top_k)
            inference_time = time.time() - start_time
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if results:
                logger.info(f"Classification completed in {inference_time:.3f} seconds")
                self.display_results(results)
            else:
                logger.error("Classification failed")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during recording: {str(e)}")
            return None
    
    def display_results(self, results):
        """Display classification results in a formatted way"""
        print("\n" + "=" * 80)
        print("🐦 BIRD CLASSIFICATION RESULTS 🐦")
        print("=" * 80)
        
        for result in results:
            print(f"\n#{result['rank']} - {result['confidence']} confidence")
            print(f"Species Code: {result['species_code']}")
            print(f"Common Name: {result['common_name']}")
            print(f"Scientific Name: {result['scientific_name']}")
            print("-" * 50)
        
        # Highlight the top prediction
        if results:
            top_result = results[0]
            print(f"\n🏆 MOST LIKELY BIRD:")
            print(f"   {top_result['common_name']} ({top_result['species_code']})")
            print(f"   Confidence: {top_result['confidence']}")
        
        print("=" * 80)
    
    def continuous_monitoring(self, interval=10, top_k=3):
        """Continuously monitor and classify bird sounds"""
        logger.info(f"Starting continuous monitoring (every {interval} seconds)")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Listening...")
                results = self.record_and_classify(duration=interval, top_k=top_k)
                
                if results and results[0]['probability'] > 0.1:  # Only show if confident enough
                    self.display_results(results[:top_k])
                else:
                    print("No clear bird sounds detected or low confidence predictions")
                
                time.sleep(1)  # Brief pause between recordings
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
    
    def batch_classify_directory(self, directory_path, top_k=3):
        """Classify all audio files in a directory"""
        logger.info(f"Batch classifying files in: {directory_path}")
        
        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return
        
        # Find audio files
        audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
        audio_files = []
        
        for file in os.listdir(directory_path):
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(directory_path, file))
        
        if not audio_files:
            logger.info("No audio files found in the directory")
            return
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        results_summary = []
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {os.path.basename(audio_file)}")
            results = self.classify_audio_file(audio_file, top_k=top_k)
            
            if results:
                results_summary.append({
                    'file': os.path.basename(audio_file),
                    'top_prediction': results[0]
                })
        
        # Display summary
        print(f"\n{'='*80}")
        print("BATCH CLASSIFICATION SUMMARY")
        print(f"{'='*80}")
        
        for result in results_summary:
            print(f"{result['file']:<30} -> {result['top_prediction']['common_name']:<25} ({result['top_prediction']['confidence']})")

def main():
    """Main function for real-time classification"""
    parser = argparse.ArgumentParser(description='Real-time Bird Sound Classification')
    parser.add_argument('--model', default='best_bird_model.pth', help='Path to trained model')
    parser.add_argument('--mode', choices=['file', 'record', 'monitor', 'batch'], default='record',
                       help='Classification mode')
    parser.add_argument('--input', help='Input audio file or directory')
    parser.add_argument('--duration', type=int, default=10, help='Recording duration in seconds')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions to show')
    parser.add_argument('--interval', type=int, default=10, help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        logger.info("Please train the model first using bird_classifier.py")
        return
    
    # Initialize classifier
    classifier = RealTimeBirdClassifier(args.model)
    
    if args.mode == 'file':
        if not args.input:
            logger.error("Please provide input audio file with --input")
            return
        classifier.classify_audio_file(args.input, top_k=args.top_k)
    
    elif args.mode == 'record':
        classifier.record_and_classify(duration=args.duration, top_k=args.top_k)
    
    elif args.mode == 'monitor':
        classifier.continuous_monitoring(interval=args.interval, top_k=args.top_k)
    
    elif args.mode == 'batch':
        if not args.input:
            logger.error("Please provide input directory with --input")
            return
        classifier.batch_classify_directory(args.input, top_k=args.top_k)

if __name__ == "__main__":
    main()
