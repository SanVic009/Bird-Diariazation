#!/usr/bin/env python3
"""
generate_graphs.py - Generate comprehensive graphs from diarization results

This script creates various visualizations from your saved diarization results:
- Speaker distribution charts
- Embedding scatter plots with t-SNE
- Clustering quality metrics
- Temporal analysis (if timestamps available)
- Confusion matrices and similarity heatmaps
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import os
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class DiarizationGraphGenerator:
    """Generate comprehensive graphs from diarization results"""
    
    def __init__(self, results_dir="results/"):
        """Initialize with results directory"""
        self.results_dir = results_dir
        self.embeddings = None
        self.labels = None
        self.load_results()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_results(self):
        """Load saved diarization results"""
        try:
            # Try different result file names
            embedding_files = [
                "quick_embeddings.npy",
                "audio_embeddings.npy", 
                "test_embeddings.npy"
            ]
            
            label_files = [
                "quick_labels.npy",
                "speaker_labels.npy",
                "test_labels.npy"
            ]
            
            # Load embeddings
            for emb_file in embedding_files:
                emb_path = os.path.join(self.results_dir, emb_file)
                if os.path.exists(emb_path):
                    self.embeddings = np.load(emb_path)
                    print(f"✅ Loaded embeddings from {emb_file}")
                    break
            else:
                # Try loading from root directory
                for emb_file in embedding_files:
                    if os.path.exists(emb_file):
                        self.embeddings = np.load(emb_file)
                        print(f"✅ Loaded embeddings from {emb_file}")
                        break
            
            # Load labels
            for label_file in label_files:
                label_path = os.path.join(self.results_dir, label_file)
                if os.path.exists(label_path):
                    self.labels = np.load(label_path)
                    print(f"✅ Loaded labels from {label_file}")
                    break
            else:
                # Try loading from root directory  
                for label_file in label_files:
                    if os.path.exists(label_file):
                        self.labels = np.load(label_file)
                        print(f"✅ Loaded labels from {label_file}")
                        break
                        
            if self.embeddings is None or self.labels is None:
                raise FileNotFoundError("No diarization results found!")
                
            print(f"📊 Loaded data: {len(self.embeddings)} segments, {len(np.unique(self.labels))} speakers")
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            print("💡 Make sure to run diarization first to generate results")
            raise
    
    def create_speaker_distribution_plots(self):
        """Create various speaker distribution visualizations"""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        n_speakers = len(unique_labels)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Speaker Distribution Analysis ({n_speakers} Birds Detected)', fontsize=16, fontweight='bold')
        
        # 1. Bar chart
        ax1 = axes[0, 0]
        bars = ax1.bar(unique_labels, counts, color=plt.cm.Set3(np.linspace(0, 1, len(unique_labels))))
        ax1.set_xlabel('Bird/Speaker ID')
        ax1.set_ylabel('Number of Segments')
        ax1.set_title('Segments per Speaker')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Pie chart
        ax2 = axes[0, 1]
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        wedges, texts, autotexts = ax2.pie(counts, labels=[f'Bird {i}' for i in unique_labels], 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Speaker Time Distribution')
        
        # 3. Horizontal bar chart with percentages
        ax3 = axes[1, 0]
        percentages = counts / np.sum(counts) * 100
        bars = ax3.barh(unique_labels, percentages, color=colors)
        ax3.set_xlabel('Percentage of Total Time (%)')
        ax3.set_ylabel('Bird/Speaker ID')
        ax3.set_title('Speaker Activity Percentage')
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            width = bar.get_width()
            ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                    f'{pct:.1f}%', ha='left', va='center', fontweight='bold')
        
        # 4. Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics
        total_segments = len(self.labels)
        mean_segments = np.mean(counts)
        std_segments = np.std(counts)
        max_speaker = unique_labels[np.argmax(counts)]
        min_speaker = unique_labels[np.argmin(counts)]
        
        stats_text = f"""
📊 DIARIZATION STATISTICS

🔢 Total Analysis:
   • Total segments: {total_segments}
   • Unique speakers: {n_speakers}
   • Avg segments per speaker: {mean_segments:.1f}
   • Std deviation: {std_segments:.1f}

🏆 Speaker Rankings:
   • Most active: Bird {max_speaker} ({max(counts)} segments)
   • Least active: Bird {min_speaker} ({min(counts)} segments)
   • Activity ratio: {max(counts)/min(counts):.1f}:1

📈 Distribution Quality:
   • Uniformity: {(1 - std_segments/mean_segments):.2f}
   • Balance score: {min(counts)/max(counts):.2f}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs("graphs", exist_ok=True)
        plt.savefig("graphs/speaker_distribution.png", dpi=300, bbox_inches='tight')
        print("📊 Speaker distribution plots saved to graphs/speaker_distribution.png")
        
        return fig
    
    def create_embedding_visualizations(self):
        """Create embedding space visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Audio Embedding Space Analysis', fontsize=16, fontweight='bold')
        
        # 1. t-SNE visualization
        ax1 = axes[0, 0]
        perplexity = min(30, len(self.embeddings) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=self.labels, cmap='tab10', alpha=0.7, s=50)
        ax1.set_title('t-SNE Embedding Visualization')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('Speaker ID')
        
        # 2. PCA visualization  
        ax2 = axes[0, 1]
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(self.embeddings)
        
        scatter2 = ax2.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
                              c=self.labels, cmap='tab10', alpha=0.7, s=50)
        ax2.set_title(f'PCA Visualization (Var: {pca.explained_variance_ratio_.sum():.2f})')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Speaker ID')
        
        # 3. Embedding space density
        ax3 = axes[1, 0]
        ax3.hexbin(embeddings_2d[:, 0], embeddings_2d[:, 1], gridsize=20, cmap='YlOrRd')
        ax3.set_title('Embedding Density Map')
        ax3.set_xlabel('t-SNE Component 1')
        ax3.set_ylabel('t-SNE Component 2')
        
        # 4. Clustering quality metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate quality metrics
        silhouette = silhouette_score(self.embeddings, self.labels)
        calinski = calinski_harabasz_score(self.embeddings, self.labels)
        
        # Inertia for different k values
        k_values = range(2, min(11, len(np.unique(self.labels)) + 3))
        inertias = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.embeddings)
            inertias.append(kmeans.inertia_)
        
        quality_text = f"""
🎯 CLUSTERING QUALITY METRICS

📈 Current Performance:
   • Silhouette Score: {silhouette:.3f}
     (-1 = poor, +1 = excellent)
   • Calinski-Harabasz: {calinski:.1f}
     (higher = better separation)

🔍 Embedding Properties:
   • Dimensions: {self.embeddings.shape[1]}
   • Data points: {len(self.embeddings)}
   • PCA variance captured: {pca.explained_variance_ratio_.sum():.2f}

💡 Quality Assessment:
   • Separation: {'Excellent' if silhouette > 0.5 else 'Good' if silhouette > 0.2 else 'Fair' if silhouette > 0 else 'Poor'}
   • Compactness: {'High' if calinski > 100 else 'Medium' if calinski > 10 else 'Low'}
        """
        
        ax4.text(0.05, 0.95, quality_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig("graphs/embedding_analysis.png", dpi=300, bbox_inches='tight')
        print("🎯 Embedding visualizations saved to graphs/embedding_analysis.png")
        
        return fig, embeddings_2d
    
    def create_similarity_heatmap(self):
        """Create similarity heatmap between speakers"""
        unique_labels = np.unique(self.labels)
        n_speakers = len(unique_labels)
        
        # Calculate average embeddings per speaker
        speaker_embeddings = []
        for label in unique_labels:
            mask = self.labels == label
            avg_embedding = np.mean(self.embeddings[mask], axis=0)
            speaker_embeddings.append(avg_embedding)
        
        speaker_embeddings = np.array(speaker_embeddings)
        
        # Calculate similarity matrix (cosine similarity)
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(speaker_embeddings)
        
        # Create heatmap
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Speaker Similarity Analysis', fontsize=16, fontweight='bold')
        
        # Similarity heatmap
        ax1 = axes[0]
        sns.heatmap(similarity_matrix, 
                   xticklabels=[f'Bird {i}' for i in unique_labels],
                   yticklabels=[f'Bird {i}' for i in unique_labels],
                   annot=True, fmt='.2f', cmap='RdYlBu_r',
                   ax=ax1, cbar_kws={'label': 'Cosine Similarity'})
        ax1.set_title('Speaker Similarity Matrix')
        
        # Hierarchical clustering dendrogram
        ax2 = axes[1]
        linkage_matrix = linkage(1 - similarity_matrix, method='ward')
        dendrogram(linkage_matrix, 
                  labels=[f'Bird {i}' for i in unique_labels],
                  ax=ax2, orientation='top')
        ax2.set_title('Speaker Clustering Dendrogram')
        ax2.set_ylabel('Distance')
        
        plt.tight_layout()
        plt.savefig("graphs/similarity_analysis.png", dpi=300, bbox_inches='tight')
        print("🔗 Similarity analysis saved to graphs/similarity_analysis.png")
        
        return fig, similarity_matrix
    
    def create_interactive_plots(self, embeddings_2d):
        """Create interactive Plotly visualizations"""
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1], 
            'speaker': [f'Bird {label}' for label in self.labels],
            'segment_id': range(len(self.labels))
        })
        
        # Interactive scatter plot
        fig = px.scatter(df, x='x', y='y', color='speaker',
                        title='Interactive t-SNE Embedding Visualization',
                        labels={'x': 't-SNE Component 1', 'y': 't-SNE Component 2'},
                        hover_data=['segment_id'])
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(
            width=800, height=600,
            title_font_size=16,
            showlegend=True
        )
        
        # Save interactive plot
        fig.write_html("graphs/interactive_embeddings.html")
        print("🌐 Interactive plot saved to graphs/interactive_embeddings.html")
        
        return fig
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive analysis dashboard"""
        print("\n" + "="*60)
        print("🎨 GENERATING COMPREHENSIVE DIARIZATION GRAPHS")
        print("="*60)
        
        # Create output directory
        os.makedirs("graphs", exist_ok=True)
        
        # Generate all visualizations
        print("\n1️⃣ Creating speaker distribution plots...")
        self.create_speaker_distribution_plots()
        
        print("\n2️⃣ Creating embedding visualizations...")
        fig_embed, embeddings_2d = self.create_embedding_visualizations()
        
        print("\n3️⃣ Creating similarity analysis...")
        fig_sim, similarity_matrix = self.create_similarity_heatmap()
        
        print("\n4️⃣ Creating interactive plots...")
        self.create_interactive_plots(embeddings_2d)
        
        # Summary report
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        silhouette = silhouette_score(self.embeddings, self.labels)
        
        print("\n" + "="*60)
        print("📊 ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📁 All graphs saved to: graphs/")
        print(f"🐦 Birds detected: {len(unique_labels)}")
        print(f"📈 Clustering quality: {silhouette:.3f}")
        print(f"🎯 Files generated:")
        print(f"   • graphs/speaker_distribution.png")
        print(f"   • graphs/embedding_analysis.png") 
        print(f"   • graphs/similarity_analysis.png")
        print(f"   • graphs/interactive_embeddings.html")
        
        return {
            'n_speakers': len(unique_labels),
            'silhouette_score': silhouette,
            'speaker_counts': dict(zip(unique_labels, counts)),
            'similarity_matrix': similarity_matrix
        }

def main():
    """Main function to generate all graphs"""
    try:
        generator = DiarizationGraphGenerator()
        results = generator.create_comprehensive_dashboard()
        
        print(f"\n✅ Graph generation completed successfully!")
        print(f"🔍 Open graphs/interactive_embeddings.html in your browser for interactive exploration")
        
    except Exception as e:
        print(f"\n❌ Error generating graphs: {e}")
        print("\n💡 Troubleshooting:")
        print("   • Make sure you have run diarization first")
        print("   • Check that results/ directory contains .npy files")
        print("   • Install required packages: pip install plotly pandas")

if __name__ == "__main__":
    main()