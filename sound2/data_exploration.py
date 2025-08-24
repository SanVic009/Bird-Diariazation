#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import warnings
import noisereduce as nr
import pickle
from collections import Counter
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_exploration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DataExplorer:
    """Data exploration and cleaning for BirdCLEF dataset"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_df = None
        self.taxonomy_df = None
        self.audio_stats = {}
        self.cleaned_files = []
        self.corrupted_files = []
        
    def load_metadata(self):
        """Load all metadata files"""
        logger.info("Loading metadata files...")
        
        # Load training metadata
        self.train_df = pd.read_csv(os.path.join(self.data_dir, 'train_metadata.csv'))
        logger.info(f"Loaded {len(self.train_df)} training samples")
        
        # Load taxonomy data
        self.taxonomy_df = pd.read_csv(os.path.join(self.data_dir, 'eBird_Taxonomy_v2021.csv'))
        logger.info(f"Loaded {len(self.taxonomy_df)} taxonomy entries")
        
        # Load sample submission to understand target format
        sample_sub = pd.read_csv(os.path.join(self.data_dir, 'sample_submission.csv'))
        logger.info(f"Sample submission has {len(sample_sub.columns)-1} target species")
        
        return self.train_df, self.taxonomy_df
    
    def analyze_dataset_structure(self):
        """Analyze the overall dataset structure"""
        logger.info("Analyzing dataset structure...")
        
        # Basic statistics
        stats = {
            'total_samples': len(self.train_df),
            'unique_species': self.train_df['primary_label'].nunique(),
            'unique_authors': self.train_df['author'].nunique(),
            'rating_range': (self.train_df['rating'].min(), self.train_df['rating'].max()),
            'date_range': 'Audio data from Xeno-canto database',
        }
        
        logger.info("Dataset Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Species distribution
        species_counts = self.train_df['primary_label'].value_counts()
        logger.info(f"Most common species: {species_counts.head(10).to_dict()}")
        logger.info(f"Least common species: {species_counts.tail(10).to_dict()}")
        
        # Visualizations
        self.plot_species_distribution()
        self.plot_rating_distribution()
        self.plot_geographic_distribution()
        
        return stats
    
    def plot_species_distribution(self):
        """Plot species distribution"""
        plt.figure(figsize=(15, 8))
        
        species_counts = self.train_df['primary_label'].value_counts()
        
        # Top 20 species
        plt.subplot(2, 2, 1)
        species_counts.head(20).plot(kind='bar')
        plt.title('Top 20 Most Common Species')
        plt.xlabel('Species Code')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Distribution histogram
        plt.subplot(2, 2, 2)
        plt.hist(species_counts.values, bins=50, alpha=0.7)
        plt.title('Distribution of Samples per Species')
        plt.xlabel('Number of Samples')
        plt.ylabel('Number of Species')
        
        # Rating distribution
        plt.subplot(2, 2, 3)
        self.train_df['rating'].hist(bins=20, alpha=0.7)
        plt.title('Audio Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        
        # Type distribution
        plt.subplot(2, 2, 4)
        type_counts = self.train_df['type'].value_counts()
        type_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Audio Type Distribution')
        
        plt.tight_layout()
        plt.savefig('dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("Dataset overview plot saved as 'dataset_overview.png'")
    
    def plot_rating_distribution(self):
        """Plot rating distribution by species"""
        plt.figure(figsize=(12, 6))
        
        # Box plot of ratings by top species
        top_species = self.train_df['primary_label'].value_counts().head(10).index
        data_subset = self.train_df[self.train_df['primary_label'].isin(top_species)]
        
        sns.boxplot(data=data_subset, x='primary_label', y='rating')
        plt.title('Rating Distribution by Top 10 Species')
        plt.xlabel('Species Code')
        plt.ylabel('Rating')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('rating_by_species.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_geographic_distribution(self):
        """Plot geographic distribution of recordings"""
        plt.figure(figsize=(15, 6))
        
        # Remove invalid coordinates
        valid_coords = self.train_df.dropna(subset=['latitude', 'longitude'])
        valid_coords = valid_coords[
            (valid_coords['latitude'].between(-90, 90)) & 
            (valid_coords['longitude'].between(-180, 180))
        ]
        
        plt.subplot(1, 2, 1)
        plt.scatter(valid_coords['longitude'], valid_coords['latitude'], 
                   alpha=0.5, s=1)
        plt.title('Geographic Distribution of Recordings')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist2d(valid_coords['longitude'], valid_coords['latitude'], 
                  bins=50, cmap='Blues')
        plt.title('Recording Density Heatmap')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.colorbar(label='Number of Recordings')
        
        plt.tight_layout()
        plt.savefig('geographic_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Valid coordinates: {len(valid_coords)}/{len(self.train_df)}")
    
    def analyze_audio_files(self, sample_size=1000):
        """Analyze audio file properties"""
        logger.info(f"Analyzing audio files (sample size: {sample_size})...")
        
        audio_dir = os.path.join(self.data_dir, 'train_audio')
        
        # Sample files for analysis
        sample_files = self.train_df.sample(n=min(sample_size, len(self.train_df)))
        
        durations = []
        sample_rates = []
        channels = []
        file_sizes = []
        spectral_features = []
        
        for _, row in tqdm(sample_files.iterrows(), total=len(sample_files), desc="Analyzing audio"):
            file_path = os.path.join(audio_dir, row['filename'])
            
            try:
                if os.path.exists(file_path):
                    # Get basic file info
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    file_sizes.append(file_size)
                    
                    # Load audio
                    audio, sr = librosa.load(file_path, sr=None)
                    duration = len(audio) / sr
                    
                    durations.append(duration)
                    sample_rates.append(sr)
                    channels.append(1)  # librosa loads as mono by default
                    
                    # Spectral features
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
                    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
                    
                    spectral_features.append({
                        'spectral_centroid': spectral_centroid,
                        'spectral_rolloff': spectral_rolloff,
                        'zero_crossing_rate': zero_crossing_rate,
                        'rms_energy': np.mean(librosa.feature.rms(y=audio))
                    })
                
                else:
                    self.corrupted_files.append(file_path)
                    
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {str(e)}")
                self.corrupted_files.append(file_path)
        
        # Store statistics
        self.audio_stats = {
            'durations': durations,
            'sample_rates': sample_rates,
            'file_sizes': file_sizes,
            'spectral_features': spectral_features
        }
        
        # Log statistics
        logger.info("Audio File Statistics:")
        logger.info(f"  Duration - Mean: {np.mean(durations):.2f}s, Std: {np.std(durations):.2f}s")
        logger.info(f"  Duration - Min: {np.min(durations):.2f}s, Max: {np.max(durations):.2f}s")
        logger.info(f"  Sample Rate - Unique values: {set(sample_rates)}")
        logger.info(f"  File Size - Mean: {np.mean(file_sizes):.2f}KB, Std: {np.std(file_sizes):.2f}KB")
        logger.info(f"  Corrupted files: {len(self.corrupted_files)}")
        
        # Plot audio statistics
        self.plot_audio_statistics()
        
        return self.audio_stats
    
    def plot_audio_statistics(self):
        """Plot audio file statistics"""
        plt.figure(figsize=(15, 10))
        
        # Duration distribution
        plt.subplot(2, 3, 1)
        plt.hist(self.audio_stats['durations'], bins=50, alpha=0.7)
        plt.title('Audio Duration Distribution')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Frequency')
        
        # Sample rate distribution
        plt.subplot(2, 3, 2)
        sr_counts = Counter(self.audio_stats['sample_rates'])
        plt.bar(sr_counts.keys(), sr_counts.values())
        plt.title('Sample Rate Distribution')
        plt.xlabel('Sample Rate (Hz)')
        plt.ylabel('Frequency')
        
        # File size distribution
        plt.subplot(2, 3, 3)
        plt.hist(self.audio_stats['file_sizes'], bins=50, alpha=0.7)
        plt.title('File Size Distribution')
        plt.xlabel('File Size (KB)')
        plt.ylabel('Frequency')
        
        # Spectral features
        if self.audio_stats['spectral_features']:
            centroids = [f['spectral_centroid'] for f in self.audio_stats['spectral_features']]
            rolloffs = [f['spectral_rolloff'] for f in self.audio_stats['spectral_features']]
            zcr = [f['zero_crossing_rate'] for f in self.audio_stats['spectral_features']]
            
            plt.subplot(2, 3, 4)
            plt.hist(centroids, bins=30, alpha=0.7)
            plt.title('Spectral Centroid Distribution')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Frequency')
            
            plt.subplot(2, 3, 5)
            plt.hist(rolloffs, bins=30, alpha=0.7)
            plt.title('Spectral Rolloff Distribution')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Frequency')
            
            plt.subplot(2, 3, 6)
            plt.hist(zcr, bins=30, alpha=0.7)
            plt.title('Zero Crossing Rate Distribution')
            plt.xlabel('ZCR')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('audio_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
        logger.info("Audio statistics plot saved as 'audio_statistics.png'")
    
    def detect_and_clean_audio(self, output_dir='cleaned_audio', sample_size=100):
        """Detect issues and clean audio files"""
        logger.info(f"Starting audio cleaning process (sample size: {sample_size})...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        audio_dir = os.path.join(self.data_dir, 'train_audio')
        sample_files = self.train_df.sample(n=min(sample_size, len(self.train_df)))
        
        cleaning_stats = {
            'processed': 0,
            'noise_reduced': 0,
            'normalized': 0,
            'resampled': 0,
            'duration_adjusted': 0,
            'failed': 0
        }
        
        for _, row in tqdm(sample_files.iterrows(), total=len(sample_files), desc="Cleaning audio"):
            file_path = os.path.join(audio_dir, row['filename'])
            output_path = os.path.join(output_dir, row['filename'])
            
            try:
                if os.path.exists(file_path):
                    # Create subdirectory if needed
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Load audio
                    audio, sr = librosa.load(file_path, sr=None)
                    original_audio = audio.copy()
                    
                    cleaning_applied = []
                    
                    # 1. Noise reduction
                    if self.should_apply_noise_reduction(audio, sr):
                        audio = nr.reduce_noise(y=audio, sr=sr)
                        cleaning_applied.append('noise_reduction')
                        cleaning_stats['noise_reduced'] += 1
                    
                    # 2. Normalize audio
                    if np.max(np.abs(audio)) > 0:
                        audio = audio / np.max(np.abs(audio)) * 0.95
                        cleaning_applied.append('normalization')
                        cleaning_stats['normalized'] += 1
                    
                    # 3. Resample if needed
                    target_sr = 22050
                    if sr != target_sr:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                        sr = target_sr
                        cleaning_applied.append('resampling')
                        cleaning_stats['resampled'] += 1
                    
                    # 4. Adjust duration
                    target_duration = 10  # seconds
                    target_length = target_duration * sr
                    
                    if len(audio) > target_length:
                        # Trim audio
                        audio = audio[:target_length]
                        cleaning_applied.append('trimming')
                        cleaning_stats['duration_adjusted'] += 1
                    elif len(audio) < target_length:
                        # Pad audio
                        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
                        cleaning_applied.append('padding')
                        cleaning_stats['duration_adjusted'] += 1
                    
                    # Save cleaned audio
                    sf.write(output_path, audio, sr)
                    
                    self.cleaned_files.append({
                        'original_path': file_path,
                        'cleaned_path': output_path,
                        'cleaning_applied': cleaning_applied,
                        'species': row['primary_label']
                    })
                    
                    cleaning_stats['processed'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to clean {file_path}: {str(e)}")
                cleaning_stats['failed'] += 1
        
        # Log cleaning statistics
        logger.info("Audio Cleaning Statistics:")
        for key, value in cleaning_stats.items():
            logger.info(f"  {key}: {value}")
        
        # Save cleaning report
        cleaning_report = {
            'stats': cleaning_stats,
            'cleaned_files': self.cleaned_files,
            'corrupted_files': self.corrupted_files
        }
        
        with open('cleaning_report.json', 'w') as f:
            json.dump(cleaning_report, f, indent=2)
        
        logger.info("Cleaning report saved as 'cleaning_report.json'")
        
        return cleaning_stats
    
    def should_apply_noise_reduction(self, audio, sr):
        """Determine if noise reduction should be applied"""
        # Simple heuristic: apply if the signal has low SNR
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(audio[:sr] ** 2)  # Assume first second might be mostly noise
        
        if signal_power > 0 and noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return snr < 15  # Apply noise reduction if SNR < 15 dB
        
        return False
    
    def generate_species_report(self):
        """Generate detailed species report"""
        logger.info("Generating species report...")
        
        species_data = []
        
        for species_code in self.train_df['primary_label'].unique():
            species_samples = self.train_df[self.train_df['primary_label'] == species_code]
            
            # Get taxonomy info
            taxonomy_info = self.taxonomy_df[
                self.taxonomy_df['SPECIES_CODE'] == species_code
            ].iloc[0] if len(self.taxonomy_df[self.taxonomy_df['SPECIES_CODE'] == species_code]) > 0 else {}
            
            species_info = {
                'species_code': species_code,
                'common_name': taxonomy_info.get('PRIMARY_COM_NAME', 'Unknown'),
                'scientific_name': taxonomy_info.get('SCI_NAME', 'Unknown'),
                'family': taxonomy_info.get('FAMILY', 'Unknown'),
                'order': taxonomy_info.get('ORDER1', 'Unknown'),
                'sample_count': len(species_samples),
                'avg_rating': species_samples['rating'].mean(),
                'rating_std': species_samples['rating'].std(),
                'unique_authors': species_samples['author'].nunique(),
                'call_types': species_samples['type'].unique().tolist()
            }
            
            species_data.append(species_info)
        
        # Create DataFrame and save
        species_df = pd.DataFrame(species_data)
        species_df = species_df.sort_values('sample_count', ascending=False)
        species_df.to_csv('species_report.csv', index=False)
        
        logger.info(f"Species report saved as 'species_report.csv' with {len(species_df)} species")
        
        return species_df
    
    def save_exploration_results(self):
        """Save all exploration results"""
        results = {
            'audio_stats': self.audio_stats,
            'cleaned_files': self.cleaned_files,
            'corrupted_files': self.corrupted_files,
            'dataset_stats': {
                'total_samples': len(self.train_df),
                'unique_species': self.train_df['primary_label'].nunique(),
                'unique_authors': self.train_df['author'].nunique()
            }
        }
        
        with open('exploration_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        logger.info("Exploration results saved as 'exploration_results.pkl'")

def main():
    """Main exploration function"""
    logger.info("=" * 60)
    logger.info("BirdCLEF-2024 Data Exploration and Cleaning")
    logger.info("=" * 60)
    
    # Initialize explorer
    explorer = DataExplorer(".")
    
    # Load metadata
    train_df, taxonomy_df = explorer.load_metadata()
    
    # Analyze dataset structure
    dataset_stats = explorer.analyze_dataset_structure()
    
    # Analyze audio files
    audio_stats = explorer.analyze_audio_files(sample_size=500)  # Analyze 500 files
    
    # Clean audio files
    cleaning_stats = explorer.detect_and_clean_audio(sample_size=100)  # Clean 100 files
    
    # Generate species report
    species_report = explorer.generate_species_report()
    
    # Save results
    explorer.save_exploration_results()
    
    logger.info("Data exploration completed successfully!")
    logger.info(f"Total species: {dataset_stats['unique_species']}")
    logger.info(f"Audio files analyzed: {len(audio_stats['durations'])}")
    logger.info(f"Files cleaned: {cleaning_stats['processed']}")

if __name__ == "__main__":
    main()
