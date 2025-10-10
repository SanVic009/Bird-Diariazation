#!/usr/bin/env python3
"""
advanced_graphs.py - Generate advanced temporal and statistical analysis graphs

This script creates additional specialized visualizations:
- Temporal activity patterns
- Speaker transition analysis  
- Quality assessment over time
- Advanced clustering validation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, DBSCAN
import os

def create_advanced_analysis():
    """Create advanced analysis graphs"""
    
    # Load results
    try:
        embeddings = None
        labels = None
        
        # Try to load results
        result_files = [
            ("results/quick_embeddings.npy", "results/quick_labels.npy"),
            ("quick_embeddings.npy", "quick_labels.npy"),
            ("results/audio_embeddings.npy", "results/speaker_labels.npy"),
            ("audio_embeddings.npy", "speaker_labels.npy")
        ]
        
        for emb_file, label_file in result_files:
            if os.path.exists(emb_file) and os.path.exists(label_file):
                embeddings = np.load(emb_file)
                labels = np.load(label_file)
                print(f"✅ Loaded data from {emb_file} and {label_file}")
                break
        
        if embeddings is None or labels is None:
            raise FileNotFoundError("No results found")
            
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return
    
    # Create advanced analysis
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Advanced Diarization Analysis', fontsize=16, fontweight='bold')
    
    # 1. Speaker transition matrix
    ax1 = axes[0, 0]
    transition_matrix = create_transition_matrix(labels)
    sns.heatmap(transition_matrix, annot=True, fmt='.2f', cmap='Blues', ax=ax1)
    ax1.set_title('Speaker Transition Probabilities')
    ax1.set_xlabel('Next Speaker')
    ax1.set_ylabel('Current Speaker')
    
    # 2. Clustering comparison
    ax2 = axes[0, 1]
    compare_clustering_methods(embeddings, labels, ax2)
    
    # 3. Embedding quality by speaker
    ax3 = axes[0, 2]
    analyze_speaker_quality(embeddings, labels, ax3)
    
    # 4. Sequential pattern analysis
    ax4 = axes[1, 0]
    analyze_sequential_patterns(labels, ax4)
    
    # 5. Embedding distance distribution
    ax5 = axes[1, 1]
    analyze_embedding_distances(embeddings, labels, ax5)
    
    # 6. Statistical summary
    ax6 = axes[1, 2]
    create_statistical_summary(embeddings, labels, ax6)
    
    plt.tight_layout()
    
    # Save advanced analysis
    os.makedirs("graphs", exist_ok=True)
    plt.savefig("graphs/advanced_analysis.png", dpi=300, bbox_inches='tight')
    print("📈 Advanced analysis saved to graphs/advanced_analysis.png")
    
    return fig

def create_transition_matrix(labels):
    """Create speaker transition probability matrix"""
    unique_labels = np.unique(labels)
    n_speakers = len(unique_labels)
    transition_matrix = np.zeros((n_speakers, n_speakers))
    
    # Count transitions
    for i in range(len(labels) - 1):
        current_speaker = np.where(unique_labels == labels[i])[0][0]
        next_speaker = np.where(unique_labels == labels[i + 1])[0][0]
        transition_matrix[current_speaker, next_speaker] += 1
    
    # Normalize to probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, 
                                 out=np.zeros_like(transition_matrix), 
                                 where=row_sums!=0)
    
    return transition_matrix

def compare_clustering_methods(embeddings, true_labels, ax):
    """Compare different clustering methods"""
    n_clusters = len(np.unique(true_labels))
    
    methods = {
        'K-Means': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=3)
    }
    
    scores = {'Method': [], 'ARI': [], 'NMI': []}
    
    for name, method in methods.items():
        try:
            pred_labels = method.fit_predict(embeddings)
            
            # Handle DBSCAN noise points
            if name == 'DBSCAN' and -1 in pred_labels:
                # Map -1 (noise) to a new cluster
                pred_labels = pred_labels.copy()
                pred_labels[pred_labels == -1] = max(pred_labels) + 1
            
            ari = adjusted_rand_score(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)
            
            scores['Method'].append(name)
            scores['ARI'].append(ari)
            scores['NMI'].append(nmi)
        except Exception as e:
            print(f"⚠️ Warning: {name} failed: {e}")
    
    # Plot comparison
    x = np.arange(len(scores['Method']))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, scores['ARI'], width, label='ARI', alpha=0.8)
    bars2 = ax.bar(x + width/2, scores['NMI'], width, label='NMI', alpha=0.8)
    
    ax.set_xlabel('Clustering Method')
    ax.set_ylabel('Score')
    ax.set_title('Clustering Method Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(scores['Method'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)

def analyze_speaker_quality(embeddings, labels, ax):
    """Analyze embedding quality for each speaker"""
    unique_labels = np.unique(labels)
    speaker_qualities = []
    
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 1:  # Need at least 2 points
            speaker_embeddings = embeddings[mask]
            
            # Calculate intra-cluster distance (compactness)
            from sklearn.metrics.pairwise import pairwise_distances
            distances = pairwise_distances(speaker_embeddings)
            # Get upper triangle (excluding diagonal)
            upper_tri = np.triu(distances, k=1)
            mean_distance = np.mean(upper_tri[upper_tri > 0])
            speaker_qualities.append(mean_distance)
        else:
            speaker_qualities.append(0)
    
    bars = ax.bar(unique_labels, speaker_qualities, 
                  color=plt.cm.Set3(np.linspace(0, 1, len(unique_labels))))
    ax.set_xlabel('Speaker ID')
    ax.set_ylabel('Mean Intra-cluster Distance')
    ax.set_title('Speaker Embedding Quality\n(Lower = More Consistent)')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, quality in zip(bars, speaker_qualities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
               f'{quality:.3f}', ha='center', va='bottom', fontsize=9)

def analyze_sequential_patterns(labels, ax):
    """Analyze sequential patterns in speaker changes"""
    # Calculate run lengths (consecutive segments of same speaker)
    run_lengths = []
    current_run = 1
    
    for i in range(1, len(labels)):
        if labels[i] == labels[i-1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)  # Add final run
    
    # Plot histogram of run lengths
    ax.hist(run_lengths, bins=min(20, max(run_lengths)), 
            alpha=0.7, edgecolor='black')
    ax.set_xlabel('Consecutive Segments (Run Length)')
    ax.set_ylabel('Frequency')
    ax.set_title('Speaker Persistence Patterns')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_run = np.mean(run_lengths)
    median_run = np.median(run_lengths)
    ax.axvline(mean_run, color='red', linestyle='--', alpha=0.7, 
              label=f'Mean: {mean_run:.1f}')
    ax.axvline(median_run, color='orange', linestyle='--', alpha=0.7,
              label=f'Median: {median_run:.1f}')
    ax.legend()

def analyze_embedding_distances(embeddings, labels, ax):
    """Analyze distribution of embedding distances"""
    from sklearn.metrics.pairwise import pairwise_distances
    
    # Calculate pairwise distances
    distances = pairwise_distances(embeddings)
    
    # Separate intra-cluster and inter-cluster distances
    unique_labels = np.unique(labels)
    intra_distances = []
    inter_distances = []
    
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            distance = distances[i, j]
            if labels[i] == labels[j]:
                intra_distances.append(distance)
            else:
                inter_distances.append(distance)
    
    # Plot distributions
    ax.hist(intra_distances, bins=30, alpha=0.7, label='Intra-cluster', density=True)
    ax.hist(inter_distances, bins=30, alpha=0.7, label='Inter-cluster', density=True)
    
    ax.set_xlabel('Embedding Distance')
    ax.set_ylabel('Density')
    ax.set_title('Distance Distribution Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add separation quality metric
    if intra_distances and inter_distances:
        separation_ratio = np.mean(inter_distances) / np.mean(intra_distances)
        ax.text(0.05, 0.95, f'Separation Ratio: {separation_ratio:.2f}\n(Higher = Better)',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def create_statistical_summary(embeddings, labels, ax):
    """Create statistical summary of the diarization"""
    ax.axis('off')
    
    # Calculate comprehensive statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_speakers = len(unique_labels)
    total_segments = len(labels)
    
    # Embedding statistics
    embedding_dim = embeddings.shape[1]
    embedding_mean = np.mean(embeddings)
    embedding_std = np.std(embeddings)
    
    # Speaker balance
    speaker_entropy = -np.sum((counts / total_segments) * np.log2(counts / total_segments))
    max_entropy = np.log2(n_speakers)
    balance_score = speaker_entropy / max_entropy if max_entropy > 0 else 0
    
    # Create summary text
    summary_text = f"""
COMPREHENSIVE DIARIZATION STATISTICS

🔢 Basic Metrics:
   • Total audio segments: {total_segments}
   • Detected speakers: {n_speakers}
   • Embedding dimensions: {embedding_dim}
   
📊 Distribution Analysis:
   • Most active speaker: {np.max(counts)} segments
   • Least active speaker: {np.min(counts)} segments
   • Average per speaker: {np.mean(counts):.1f} ± {np.std(counts):.1f}
   • Speaker balance: {balance_score:.2f} (0-1, higher=better)
   
🎯 Embedding Properties:
   • Mean embedding value: {embedding_mean:.3f}
   • Embedding std dev: {embedding_std:.3f}
   • Value range: [{np.min(embeddings):.2f}, {np.max(embeddings):.2f}]
   
💡 Quality Indicators:
   • Entropy score: {speaker_entropy:.2f} / {max_entropy:.2f}
   • Activity ratio: {np.max(counts)/np.min(counts):.1f}:1
   • Segments per speaker: {total_segments/n_speakers:.1f}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

def main():
    """Main function to generate advanced graphs"""
    print("🔬 Generating Advanced Diarization Analysis...")
    print("=" * 50)
    
    try:
        create_advanced_analysis()
        print("✅ Advanced analysis completed!")
        print("📁 Graph saved to: graphs/advanced_analysis.png")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to run diarization first to generate results")

if __name__ == "__main__":
    main()