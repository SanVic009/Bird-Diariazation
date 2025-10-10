#!/usr/bin/env python3
"""
temporal_diarization.py - Show WHEN each bird speaks (the main goal of diarization)

This script creates temporal visualizations showing:
- Timeline of which bird is speaking when
- Temporal patterns and activity periods  
- Speaker switching patterns over time
- Duration analysis for each bird's segments
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
from datetime import datetime, timedelta

class TemporalDiarizationAnalyzer:
    """Analyze and visualize temporal patterns in bird diarization"""
    
    def __init__(self, segment_length=5.0, hop_length=2.5):
        """
        Initialize temporal analyzer
        
        Args:
            segment_length: Length of each audio segment in seconds
            hop_length: Time between segment starts in seconds
        """
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.labels = None
        self.embeddings = None
        self.timestamps = None
        self.load_results()
        
    def load_results(self):
        """Load diarization results and create timestamps"""
        try:
            # Load results
            result_files = [
                ("results/quick_embeddings.npy", "results/quick_labels.npy"),
                ("quick_embeddings.npy", "quick_labels.npy"),
                ("results/audio_embeddings.npy", "results/speaker_labels.npy"),
                ("audio_embeddings.npy", "speaker_labels.npy")
            ]
            
            for emb_file, label_file in result_files:
                if os.path.exists(emb_file) and os.path.exists(label_file):
                    self.embeddings = np.load(emb_file)
                    self.labels = np.load(label_file)
                    print(f"✅ Loaded results from {emb_file} and {label_file}")
                    break
            
            if self.labels is None:
                raise FileNotFoundError("No diarization results found!")
            
            # Create timestamps based on segment parameters
            self.timestamps = np.array([i * self.hop_length for i in range(len(self.labels))])
            self.total_duration = self.timestamps[-1] + self.segment_length
            
            print(f"📊 Analysis setup:")
            print(f"   • Total segments: {len(self.labels)}")
            print(f"   • Segment length: {self.segment_length}s")
            print(f"   • Hop length: {self.hop_length}s") 
            print(f"   • Total audio duration: ~{self.total_duration:.1f}s")
            print(f"   • Detected birds: {len(np.unique(self.labels))}")
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            raise

    def create_temporal_timeline(self):
        """Create the main temporal timeline showing when each bird speaks"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('🐦 Bird Diarization Timeline - "Which Bird Spoke When?"', 
                     fontsize=16, fontweight='bold')
        
        unique_labels = np.unique(self.labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        # 1. Main Timeline (Gantt-style chart)
        ax1 = axes[0]
        
        # Draw timeline segments
        for i, (timestamp, label) in enumerate(zip(self.timestamps, self.labels)):
            color = color_map[label]
            
            # Create rectangle for this segment
            rect = patches.Rectangle(
                (timestamp, label - 0.4), 
                self.segment_length, 0.8,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.8
            )
            ax1.add_patch(rect)
        
        ax1.set_xlim(0, self.total_duration)
        ax1.set_ylim(-0.5, len(unique_labels) - 0.5)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Bird Speaker ID')
        ax1.set_title('🕒 Timeline: When Each Bird Speaks (Each Bar = Audio Segment)')
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks(unique_labels)
        
        # Add time markers
        time_markers = np.arange(0, self.total_duration + 1, 10)  # Every 10 seconds
        ax1.set_xticks(time_markers)
        
        # 2. Continuous timeline (like a spectrogram view)
        ax2 = axes[1]
        
        # Create a continuous representation
        timeline_data = []
        for timestamp in self.timestamps:
            timeline_data.append(self.labels[np.argmin(np.abs(self.timestamps - timestamp))])
        
        # Plot as colored line segments
        for i in range(len(self.timestamps) - 1):
            ax2.plot([self.timestamps[i], self.timestamps[i+1]], 
                    [self.labels[i], self.labels[i]], 
                    color=color_map[self.labels[i]], linewidth=8, alpha=0.8, solid_capstyle='round')
        
        ax2.set_xlim(0, self.total_duration)
        ax2.set_ylim(-0.5, len(unique_labels) - 0.5)
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Active Bird')
        ax2.set_title('🎵 Continuous Activity Timeline')
        ax2.grid(True, alpha=0.3)
        ax2.set_yticks(unique_labels)
        ax2.set_xticks(time_markers)
        
        # 3. Activity summary bars
        ax3 = axes[2]
        
        # Calculate time each bird is active
        activity_times = {}
        for label in unique_labels:
            mask = self.labels == label
            activity_times[label] = np.sum(mask) * self.hop_length  # Approximate time
        
        bars = ax3.bar(activity_times.keys(), activity_times.values(), 
                      color=[color_map[label] for label in activity_times.keys()],
                      alpha=0.8, edgecolor='black')
        
        ax3.set_xlabel('Bird Speaker ID')
        ax3.set_ylabel('Total Active Time (seconds)')
        ax3.set_title('📊 Total Speaking Time per Bird')
        ax3.grid(True, alpha=0.3)
        
        # Add time labels on bars
        for bar, (label, time) in zip(bars, activity_times.items()):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{time:.1f}s\n({time/self.total_duration*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        
        # Save timeline
        os.makedirs("graphs", exist_ok=True)
        plt.savefig("graphs/temporal_timeline.png", dpi=300, bbox_inches='tight')
        print("🕒 Temporal timeline saved to graphs/temporal_timeline.png")
        
        return fig, activity_times
    
    def create_detailed_temporal_report(self, activity_times):
        """Create detailed temporal analysis report"""
        print("\n" + "="*70)
        print("🐦 DETAILED TEMPORAL DIARIZATION REPORT")
        print("="*70)
        
        unique_labels = np.unique(self.labels)
        
        # Overall statistics
        print(f"\n📊 OVERALL ANALYSIS:")
        print(f"   • Total audio duration: ~{self.total_duration:.1f} seconds ({self.total_duration/60:.1f} minutes)")
        print(f"   • Number of birds detected: {len(unique_labels)}")
        print(f"   • Analysis segments: {len(self.labels)} segments")
        print(f"   • Temporal resolution: {self.hop_length}s between segments")
        
        # Individual bird analysis
        print(f"\n🐦 INDIVIDUAL BIRD ACTIVITY:")
        print("-" * 50)
        
        for label in sorted(unique_labels):
            mask = self.labels == label
            segments = np.sum(mask)
            total_time = activity_times[label]
            percentage = (total_time / self.total_duration) * 100
            
            # Find when this bird is active
            active_indices = np.where(mask)[0]
            active_times = self.timestamps[active_indices]
            
            if len(active_times) > 0:
                first_appearance = active_times[0]
                last_appearance = active_times[-1]
                
                # Calculate activity periods (consecutive segments)
                periods = []
                current_start = active_times[0]
                current_end = active_times[0] + self.segment_length
                
                for i in range(1, len(active_times)):
                    if active_times[i] - active_times[i-1] <= self.hop_length + 0.1:  # Consecutive
                        current_end = active_times[i] + self.segment_length
                    else:  # Gap found, end current period
                        periods.append((current_start, current_end))
                        current_start = active_times[i]
                        current_end = active_times[i] + self.segment_length
                
                periods.append((current_start, current_end))  # Add final period
                
                print(f"\n🎵 Bird {label}:")
                print(f"   • Active time: {total_time:.1f}s ({percentage:.1f}% of total)")
                print(f"   • Segments: {segments}")
                print(f"   • First heard: {first_appearance:.1f}s")
                print(f"   • Last heard: {last_appearance:.1f}s")
                print(f"   • Activity periods: {len(periods)}")
                
                if len(periods) <= 5:  # Show periods if not too many
                    for i, (start, end) in enumerate(periods, 1):
                        duration = end - start
                        print(f"     Period {i}: {start:.1f}s - {end:.1f}s ({duration:.1f}s)")
                else:
                    # Show summary of periods
                    period_durations = [end - start for start, end in periods]
                    avg_duration = np.mean(period_durations)
                    max_duration = np.max(period_durations)
                    print(f"     Avg period length: {avg_duration:.1f}s")
                    print(f"     Longest period: {max_duration:.1f}s")
        
        # Temporal patterns
        print(f"\n⏰ TEMPORAL PATTERNS:")
        print("-" * 30)
        
        # Speaker transitions
        transitions = 0
        for i in range(len(self.labels) - 1):
            if self.labels[i] != self.labels[i + 1]:
                transitions += 1
        
        avg_segment_length = len(self.labels) / transitions if transitions > 0 else len(self.labels)
        
        print(f"   • Speaker changes: {transitions}")
        print(f"   • Average segments per bird turn: {avg_segment_length:.1f}")
        print(f"   • Speaker change frequency: {transitions/self.total_duration*60:.1f} changes/minute")
        
        # Most/least active periods
        window_size = max(5, len(self.labels) // 10)  # Analyze in windows
        max_activity_diversity = 0
        max_activity_time = 0
        min_activity_diversity = len(unique_labels) + 1
        min_activity_time = 0
        
        for i in range(len(self.labels) - window_size + 1):
            window_labels = self.labels[i:i + window_size]
            diversity = len(np.unique(window_labels))
            
            if diversity > max_activity_diversity:
                max_activity_diversity = diversity
                max_activity_time = self.timestamps[i + window_size // 2]
            
            if diversity < min_activity_diversity:
                min_activity_diversity = diversity
                min_activity_time = self.timestamps[i + window_size // 2]
        
        print(f"   • Most diverse period: {max_activity_diversity} birds around {max_activity_time:.1f}s")
        print(f"   • Least diverse period: {min_activity_diversity} birds around {min_activity_time:.1f}s")
        
        return {
            'total_duration': self.total_duration,
            'bird_count': len(unique_labels),
            'activity_times': activity_times,
            'transitions': transitions,
            'segments_per_change': avg_segment_length
        }
    
    def create_activity_heatmap(self):
        """Create heatmap showing bird activity over time"""
        # Create time bins (e.g., every 10 seconds)
        time_bins = np.arange(0, self.total_duration + 10, 10)
        unique_labels = np.unique(self.labels)
        
        # Create activity matrix
        activity_matrix = np.zeros((len(unique_labels), len(time_bins) - 1))
        
        for i, label in enumerate(unique_labels):
            mask = self.labels == label
            active_times = self.timestamps[mask]
            
            for time in active_times:
                bin_idx = np.digitize(time, time_bins) - 1
                if 0 <= bin_idx < activity_matrix.shape[1]:
                    activity_matrix[i, bin_idx] += 1
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 8))
        
        im = ax.imshow(activity_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        ax.set_xlabel('Time Bins (10-second intervals)')
        ax.set_ylabel('Bird Speaker ID')
        ax.set_title('🔥 Bird Activity Heatmap - Intensity Over Time')
        
        # Set labels
        ax.set_yticks(range(len(unique_labels)))
        ax.set_yticklabels([f'Bird {label}' for label in unique_labels])
        
        # Time labels
        time_labels = [f'{int(t)}s' for t in time_bins[:-1]]
        ax.set_xticks(range(0, len(time_labels), max(1, len(time_labels)//10)))
        ax.set_xticklabels([time_labels[i] for i in range(0, len(time_labels), max(1, len(time_labels)//10))],
                          rotation=45)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activity Intensity (segments per time bin)')
        
        plt.tight_layout()
        plt.savefig("graphs/activity_heatmap.png", dpi=300, bbox_inches='tight')
        print("🔥 Activity heatmap saved to graphs/activity_heatmap.png")
        
        return fig
    
    def analyze_all_temporal_patterns(self):
        """Run complete temporal analysis"""
        print("🕒 Analyzing temporal patterns in bird diarization...")
        print("="*60)
        
        # Create timeline visualization
        fig1, activity_times = self.create_temporal_timeline()
        
        # Create detailed report
        report = self.create_detailed_temporal_report(activity_times)
        
        # Create activity heatmap
        fig2 = self.create_activity_heatmap()
        
        print("\n" + "="*60)
        print("✅ TEMPORAL ANALYSIS COMPLETE!")
        print("="*60)
        print("📁 Generated files:")
        print("   • graphs/temporal_timeline.png - Main timeline visualization")
        print("   • graphs/activity_heatmap.png - Activity intensity heatmap")
        print("\n💡 Key insight: You now know WHEN each bird spoke!")
        
        return report

def main():
    """Main function for temporal analysis"""
    print("🐦 TEMPORAL BIRD DIARIZATION ANALYSIS")
    print("Answer: 'Which Bird Spoke When?'")
    print("="*50)
    
    try:
        analyzer = TemporalDiarizationAnalyzer()
        results = analyzer.analyze_all_temporal_patterns()
        
        print(f"\n🎯 Quick Summary:")
        print(f"   • Analyzed {results['total_duration']:.1f}s of audio")
        print(f"   • Detected {results['bird_count']} different birds")
        print(f"   • {results['transitions']} speaker changes")
        print(f"   • Average {results['segments_per_change']:.1f} segments per bird turn")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to run diarization first to generate results")

if __name__ == "__main__":
    main()