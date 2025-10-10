#!/usr/bin/env python3
"""
bird_timeline_viewer.py - Simple viewer to display when each bird spoke

This creates a clear, easy-to-read timeline showing the temporal diarization results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def create_simple_timeline():
    """Create a simple, clear timeline of bird activity"""
    
    # Load results
    try:
        labels = np.load("results/quick_labels.npy")
        print(f"✅ Loaded {len(labels)} segments")
    except:
        try:
            labels = np.load("quick_labels.npy")
            print(f"✅ Loaded {len(labels)} segments")
        except:
            print("❌ No results found. Run diarization first.")
            return
    
    # Timeline parameters (matching the analysis)
    segment_length = 5.0  # seconds
    hop_length = 2.5     # seconds
    timestamps = np.array([i * hop_length for i in range(len(labels))])
    total_duration = timestamps[-1] + segment_length
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 8))
    
    unique_labels = np.unique(labels)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    print(f"\n🐦 BIRD ACTIVITY TIMELINE ({total_duration:.1f} seconds total)")
    print("=" * 80)
    
    # Draw timeline
    for i, (timestamp, bird_id) in enumerate(zip(timestamps, labels)):
        color = colors[bird_id % len(colors)]
        
        # Draw rectangle for this time segment
        rect = patches.Rectangle(
            (timestamp, bird_id * 2), 
            segment_length, 1.5,
            linewidth=2, edgecolor='black', facecolor=color, alpha=0.7
        )
        ax.add_patch(rect)
        
        # Add text label every few segments to avoid clutter
        if i % 5 == 0:
            ax.text(timestamp + segment_length/2, bird_id * 2 + 0.75, 
                   f'Bird {bird_id}', ha='center', va='center', 
                   fontweight='bold', fontsize=8, color='white')
    
    # Formatting
    ax.set_xlim(0, total_duration + 5)
    ax.set_ylim(-1, len(unique_labels) * 2)
    ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Bird Speaker', fontsize=14, fontweight='bold')
    ax.set_title('🐦 Bird Diarization Timeline - Which Bird Spoke When?', 
                fontsize=16, fontweight='bold')
    
    # Y-axis labels
    ax.set_yticks([i * 2 + 0.75 for i in unique_labels])
    ax.set_yticklabels([f'Bird {i}' for i in unique_labels])
    
    # Time markers every 10 seconds
    time_markers = np.arange(0, total_duration + 1, 10)
    ax.set_xticks(time_markers)
    ax.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [patches.Patch(facecolor=colors[i % len(colors)], 
                                   edgecolor='black', label=f'Bird {bird_id}')
                      for i, bird_id in enumerate(unique_labels)]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig("graphs/simple_bird_timeline.png", dpi=300, bbox_inches='tight')
    print(f"📊 Simple timeline saved to graphs/simple_bird_timeline.png")
    
    # Print detailed timeline
    print(f"\n📋 DETAILED TIMELINE:")
    print("-" * 50)
    
    current_time = 0
    for i, (timestamp, bird_id) in enumerate(zip(timestamps, labels)):
        end_time = timestamp + segment_length
        print(f"⏰ {timestamp:6.1f}s - {end_time:6.1f}s: Bird {bird_id} singing")
        
        # Check for speaker changes
        if i < len(labels) - 1 and labels[i] != labels[i + 1]:
            next_bird = labels[i + 1]
            print(f"   🔄 Speaker change: Bird {bird_id} → Bird {next_bird}")
    
    # Summary statistics
    print(f"\n📈 SUMMARY:")
    print("-" * 30)
    
    for bird_id in unique_labels:
        mask = labels == bird_id
        segments = np.sum(mask)
        total_time = segments * hop_length  # Approximate
        percentage = (total_time / total_duration) * 100
        
        active_times = timestamps[mask]
        first_heard = active_times[0] if len(active_times) > 0 else 0
        last_heard = active_times[-1] + segment_length if len(active_times) > 0 else 0
        
        print(f"🐦 Bird {bird_id}: {segments:2d} segments | {total_time:5.1f}s ({percentage:4.1f}%) | First: {first_heard:5.1f}s | Last: {last_heard:5.1f}s")

def main():
    """Main function"""
    print("🐦 BIRD TIMELINE VIEWER")
    print("Showing exactly when each bird spoke")
    print("=" * 50)
    
    create_simple_timeline()
    
    print(f"\n✅ Timeline analysis complete!")
    print(f"📁 View the timeline: graphs/simple_bird_timeline.png")
    print(f"\n💡 This answers your main question: 'Which bird spoke when?'")

if __name__ == "__main__":
    main()