#!/usr/bin/env python3
"""
visualize_results.py - Visualization for Bird Diarization Results

Features:
- Audio waveform and spectrogram visualization
- Speaker timeline and diarization results
- Clustering visualization with embeddings
- Quality metrics and statistics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class DiarizationVisualizer:
    """Comprehensive visualization for diarization results"""
    
    def __init__(self, figsize=(15, 10), dpi=300):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", n_colors=10)
        
    def visualize_audio_results(self, audio_path, results_path, output_dir="visualizations"):
        """
        Create comprehensive visualization for audio diarization results
        
        Args:
            audio_path: Path to the original audio file
            results_path: Path to the JSON results file
            output_dir: Directory to save visualizations
        """
        print(f"🎨 Creating visualizations for {audio_path}")
        
        # Load results
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=22050)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate all visualizations
        self._plot_comprehensive_analysis(audio, sr, results, output_path, audio_path)
        self._plot_speaker_timeline(results, output_path)
        self._plot_speaker_statistics(results, output_path)
        
        print(f"✅ Visualizations saved to {output_path}/")
        return output_path
    
    def _plot_comprehensive_analysis(self, audio, sr, results, output_path, audio_path):
        """Create comprehensive analysis plot"""
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        fig.suptitle(f'🎯 Bird Diarization Analysis\n{Path(audio_path).name}', fontsize=16, fontweight='bold')
        
        # 1. Waveform
        time_axis = librosa.frames_to_time(np.arange(len(audio)), sr=sr)
        axes[0].plot(time_axis, audio, color='steelblue', alpha=0.7, linewidth=0.5)
        axes[0].set_title('🎵 Audio Waveform', fontweight='bold')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, len(audio)/sr)
        
        # Add speaker regions to waveform
        if results.get('timeline'):
            for i, segment in enumerate(results['timeline']):
                speaker_color = self.colors[i % len(self.colors)]
                axes[0].axvspan(segment['start'], segment['end'], 
                               alpha=0.2, color=speaker_color, 
                               label=f"{segment['speaker']}")
        
        # 2. Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                                      ax=axes[1], cmap='viridis')
        axes[1].set_title('🔊 Spectrogram', fontweight='bold')
        axes[1].set_ylabel('Frequency (Hz)')
        
        # Add speaker regions to spectrogram
        if results.get('timeline'):
            for i, segment in enumerate(results['timeline']):
                speaker_color = self.colors[i % len(self.colors)]
                axes[1].axvspan(segment['start'], segment['end'], 
                               alpha=0.3, color=speaker_color)
        
        # 3. Mel Spectrogram (what the model sees)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel',
                                ax=axes[2], cmap='magma')
        axes[2].set_title('🎭 Mel Spectrogram (Model Input)', fontweight='bold')
        axes[2].set_ylabel('Mel Frequency')
        
        # Add speaker regions to mel spectrogram
        if results.get('timeline'):
            for i, segment in enumerate(results['timeline']):
                speaker_color = self.colors[i % len(self.colors)]
                axes[2].axvspan(segment['start'], segment['end'], 
                               alpha=0.3, color=speaker_color)
        
        # 4. Speaker Timeline
        self._plot_timeline_on_axis(results, axes[3])
        
        # Add colorbar for spectrograms
        plt.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(output_path / 'comprehensive_analysis.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def _plot_timeline_on_axis(self, results, ax):
        """Plot speaker timeline on given axis"""
        if not results.get('timeline'):
            ax.text(0.5, 0.5, 'No timeline data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('🗣️ Speaker Timeline', fontweight='bold')
            return
        
        # Create timeline visualization
        speakers = list(set(seg['speaker'] for seg in results['timeline']))
        speaker_to_y = {speaker: i for i, speaker in enumerate(speakers)}
        
        for segment in results['timeline']:
            speaker = segment['speaker']
            start = segment['start']
            end = segment['end']
            y_pos = speaker_to_y[speaker]
            
            # Color by speaker
            speaker_idx = int(speaker.split('_')[1]) if '_' in speaker else 0
            color = self.colors[speaker_idx % len(self.colors)]
            
            # Plot segment
            ax.barh(y_pos, end - start, left=start, height=0.8, 
                   color=color, alpha=0.8, edgecolor='white', linewidth=2)
            
            # Add duration text
            if end - start > 2:  # Only show text for segments longer than 2s
                ax.text(start + (end - start)/2, y_pos, f'{end-start:.1f}s',
                       ha='center', va='center', fontweight='bold', color='white')
        
        ax.set_yticks(range(len(speakers)))
        ax.set_yticklabels(speakers)
        ax.set_xlabel('Time (seconds)')
        ax.set_title('🗣️ Speaker Timeline', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Set x-axis limits
        if results.get('duration'):
            ax.set_xlim(0, results['duration'])
    
    def _plot_speaker_timeline(self, results, output_path):
        """Create detailed speaker timeline plot"""
        if not results.get('timeline'):
            return
            
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle('🎭 Detailed Speaker Timeline', fontsize=16, fontweight='bold')
        
        # Get speakers and assign colors
        speakers = list(set(seg['speaker'] for seg in results['timeline']))
        speaker_colors = {speaker: self.colors[i % len(self.colors)] 
                         for i, speaker in enumerate(speakers)}
        
        # Plot timeline
        for segment in results['timeline']:
            speaker = segment['speaker']
            start = segment['start']
            end = segment['end']
            duration = segment['duration']
            
            # Plot segment
            ax.barh(speaker, duration, left=start, height=0.6,
                   color=speaker_colors[speaker], alpha=0.8, 
                   edgecolor='white', linewidth=2)
            
            # Add text annotation
            ax.text(start + duration/2, speaker, f'{duration:.1f}s',
                   ha='center', va='center', fontweight='bold', 
                   color='white' if duration > 5 else 'black')
        
        # Formatting
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Speaker', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add summary information
        n_speakers = results.get('n_speakers', 0)
        duration = results.get('duration', 0)
        method = results.get('method', 'unknown')
        
        info_text = f"Speakers: {n_speakers} | Duration: {duration:.1f}s | Method: {method}"
        if 'metrics' in results and isinstance(results['metrics'], dict):
            quality = results['metrics'].get('silhouette_score')
            if isinstance(quality, (int, float)):
                info_text += f" | Quality: {quality:.3f}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / 'speaker_timeline.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_speaker_statistics(self, results, output_path):
        """Create speaker statistics visualization"""
        if not results.get('timeline'):
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('📊 Speaker Statistics', fontsize=16, fontweight='bold')
        
        # Collect statistics
        speaker_durations = {}
        speaker_segments = {}
        
        for segment in results['timeline']:
            speaker = segment['speaker']
            duration = segment['duration']
            
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0
                speaker_segments[speaker] = 0
            
            speaker_durations[speaker] += duration
            speaker_segments[speaker] += 1
        
        speakers = list(speaker_durations.keys())
        colors = [self.colors[i % len(self.colors)] for i in range(len(speakers))]
        
        # 1. Total duration per speaker (pie chart)
        durations = list(speaker_durations.values())
        wedges, texts, autotexts = ax1.pie(durations, labels=speakers, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title('🕐 Total Speaking Time', fontweight='bold')
        
        # 2. Number of segments per speaker (bar chart)
        segments = list(speaker_segments.values())
        bars = ax2.bar(speakers, segments, color=colors, alpha=0.8)
        ax2.set_title('📈 Number of Segments', fontweight='bold')
        ax2.set_ylabel('Segment Count')
        
        # Add value labels on bars
        for bar, count in zip(bars, segments):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Average segment duration per speaker
        avg_durations = [speaker_durations[speaker] / speaker_segments[speaker] 
                        for speaker in speakers]
        bars = ax3.bar(speakers, avg_durations, color=colors, alpha=0.8)
        ax3.set_title('⏱️ Average Segment Duration', fontweight='bold')
        ax3.set_ylabel('Duration (seconds)')
        
        # Add value labels
        for bar, avg_dur in zip(bars, avg_durations):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{avg_dur:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 4. Timeline overview (simplified)
        timeline_data = []
        for segment in results['timeline']:
            timeline_data.append({
                'speaker': segment['speaker'],
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['duration']
            })
        
        # Create a simple timeline
        y_positions = {speaker: i for i, speaker in enumerate(speakers)}
        
        for segment in results['timeline']:
            speaker = segment['speaker']
            start = segment['start']
            end = segment['end']
            y_pos = y_positions[speaker]
            speaker_idx = speakers.index(speaker)
            
            ax4.barh(y_pos, end - start, left=start, height=0.6,
                    color=colors[speaker_idx], alpha=0.8)
        
        ax4.set_yticks(range(len(speakers)))
        ax4.set_yticklabels(speakers)
        ax4.set_xlabel('Time (seconds)')
        ax4.set_title('🎯 Timeline Overview', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path / 'speaker_statistics.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def visualize_batch_results(self, results_path, output_dir="visualizations"):
        """Visualize batch processing results"""
        print(f"📊 Creating batch visualization for {results_path}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create batch analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📁 Batch Diarization Analysis', fontsize=16, fontweight='bold')
        
        # Extract information
        n_speakers = results.get('n_speakers', 0)
        n_files = results.get('n_files', 0)
        method = results.get('method', 'unknown')
        file_names = results.get('file_names', [])
        labels = results.get('labels', [])
        
        # 1. Speaker distribution
        if labels:
            unique_labels, counts = np.unique(labels, return_counts=True)
            speakers = [f'Speaker_{label}' for label in unique_labels]
            
            wedges, texts, autotexts = ax1.pie(counts, labels=speakers, autopct='%1.1f%%', 
                                              startangle=90, colors=self.colors[:len(speakers)])
            ax1.set_title(f'🎭 Speaker Distribution\n({n_files} files, {n_speakers} speakers)', 
                         fontweight='bold')
        
        # 2. Files per speaker
        if labels and file_names:
            speaker_files = {}
            for i, label in enumerate(labels):
                speaker = f'Speaker_{label}'
                if speaker not in speaker_files:
                    speaker_files[speaker] = []
                if i < len(file_names):
                    speaker_files[speaker].append(file_names[i])
            
            speakers = list(speaker_files.keys())
            file_counts = [len(files) for files in speaker_files.values()]
            
            bars = ax2.bar(speakers, file_counts, color=self.colors[:len(speakers)], alpha=0.8)
            ax2.set_title('📈 Files per Speaker', fontweight='bold')
            ax2.set_ylabel('Number of Files')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, file_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Species distribution (from filenames)
        if file_names:
            species = []
            for filename in file_names:
                # Extract species from filename (assumes format: species_XC######.pt)
                parts = filename.split('_')
                if len(parts) >= 2:
                    species.append(parts[0])
                else:
                    species.append('unknown')
            
            species_counts = {}
            for sp in species:
                species_counts[sp] = species_counts.get(sp, 0) + 1
            
            # Plot top 10 species
            sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            species_names, species_count_vals = zip(*sorted_species) if sorted_species else ([], [])
            
            if species_names:
                bars = ax3.barh(species_names, species_count_vals, 
                               color=self.colors[2], alpha=0.8)
                ax3.set_title('🐦 Top Species in Batch', fontweight='bold')
                ax3.set_xlabel('Number of Files')
                
                # Add value labels
                for bar, count in zip(bars, species_count_vals):
                    width = bar.get_width()
                    ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                            f'{count}', ha='left', va='center', fontweight='bold')
        
        # 4. Processing summary
        summary_text = f"""
        📊 Batch Processing Summary
        
        • Total Files: {n_files}
        • Detected Speakers: {n_speakers}
        • Clustering Method: {method}
        • Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        Quality Metrics:
        """
        
        if isinstance(results.get('metrics'), dict):
            metrics = results['metrics']
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not key.endswith('_score'):
                    summary_text += f"\n        • {key.replace('_', ' ').title()}: {value}"
                elif isinstance(value, (int, float)):
                    summary_text += f"\n        • {key.replace('_', ' ').title()}: {value:.3f}"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'batch_analysis.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Batch visualization saved to {output_path}/batch_analysis.png")
        return output_path
    
    def create_summary_report(self, results_files, output_dir="visualizations"):
        """Create a summary report from multiple result files"""
        print("📋 Creating summary report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Collect data from all results
        all_results = []
        for results_file in results_files:
            try:
                with open(results_file, 'r') as f:
                    result = json.load(f)
                    result['source_file'] = Path(results_file).stem
                    all_results.append(result)
            except Exception as e:
                print(f"⚠️  Could not load {results_file}: {e}")
        
        if not all_results:
            print("❌ No valid results found")
            return
        
        # Create summary visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📈 Diarization Summary Report', fontsize=16, fontweight='bold')
        
        # 1. Speaker count distribution
        speaker_counts = [r.get('n_speakers', 0) for r in all_results]
        ax1 = axes[0, 0]
        
        counts, bins = np.histogram(speaker_counts, bins=range(1, max(speaker_counts) + 2))
        ax1.bar(bins[:-1], counts, width=0.8, color=self.colors[0], alpha=0.8)
        ax1.set_title('🎭 Speaker Count Distribution', fontweight='bold')
        ax1.set_xlabel('Number of Speakers')
        ax1.set_ylabel('Number of Recordings')
        ax1.set_xticks(range(1, max(speaker_counts) + 1))
        
        # 2. Duration distribution
        durations = [r.get('duration', 0) for r in all_results if r.get('duration')]
        if durations:
            ax2 = axes[0, 1]
            ax2.hist(durations, bins=10, color=self.colors[1], alpha=0.8, edgecolor='black')
            ax2.set_title('⏱️ Duration Distribution', fontweight='bold')
            ax2.set_xlabel('Duration (seconds)')
            ax2.set_ylabel('Number of Recordings')
        
        # 3. Method usage
        methods = [r.get('method', 'unknown') for r in all_results]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        ax3 = axes[1, 0]
        methods_list = list(method_counts.keys())
        counts_list = list(method_counts.values())
        
        bars = ax3.bar(methods_list, counts_list, color=self.colors[2], alpha=0.8)
        ax3.set_title('🔧 Clustering Methods Used', fontweight='bold')
        ax3.set_ylabel('Usage Count')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Quality scores (if available)
        quality_scores = []
        for r in all_results:
            if isinstance(r.get('metrics'), dict):
                score = r['metrics'].get('silhouette_score')
                if isinstance(score, (int, float)):
                    quality_scores.append(score)
        
        ax4 = axes[1, 1]
        if quality_scores:
            ax4.hist(quality_scores, bins=10, color=self.colors[3], alpha=0.8, edgecolor='black')
            ax4.set_title('📊 Quality Score Distribution', fontweight='bold')
            ax4.set_xlabel('Silhouette Score')
            ax4.set_ylabel('Number of Recordings')
            ax4.axvline(np.mean(quality_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(quality_scores):.3f}')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No quality scores available', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('📊 Quality Score Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'summary_report.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Summary report saved to {output_path}/summary_report.png")
        return output_path

def main():
    parser = argparse.ArgumentParser(description="Visualize Bird Diarization Results")
    parser.add_argument('--audio', help="Path to audio file")
    parser.add_argument('--results', help="Path to results JSON file")
    parser.add_argument('--batch-results', help="Path to batch results JSON file")
    parser.add_argument('--summary', nargs='+', help="Paths to multiple result files for summary")
    parser.add_argument('--output', default="visualizations", help="Output directory")
    parser.add_argument('--dpi', type=int, default=300, help="DPI for saved images")
    
    args = parser.parse_args()
    
    visualizer = DiarizationVisualizer(dpi=args.dpi)
    
    if args.audio and args.results:
        visualizer.visualize_audio_results(args.audio, args.results, args.output)
    elif args.batch_results:
        visualizer.visualize_batch_results(args.batch_results, args.output)
    elif args.summary:
        visualizer.create_summary_report(args.summary, args.output)
    else:
        print("❌ Please specify --audio and --results, --batch-results, or --summary")

if __name__ == "__main__":
    main()