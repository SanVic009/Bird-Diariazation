#!/usr/bin/env python3
"""
view_graphs.py - Simple script to display the generated diarization graphs

Usage:
    python view_graphs.py                    # Show all graphs
    python view_graphs.py --graph speaker    # Show specific graph
    python view_graphs.py --summary          # Show summary only
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse
import webbrowser
from pathlib import Path

def display_graph(graph_path, title=""):
    """Display a single graph"""
    if not os.path.exists(graph_path):
        print(f"❌ Graph not found: {graph_path}")
        return False
        
    try:
        img = mpimg.imread(graph_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        return True
    except Exception as e:
        print(f"❌ Error displaying {graph_path}: {e}")
        return False

def show_all_graphs():
    """Display all generated graphs"""
    graphs_dir = "graphs"
    
    if not os.path.exists(graphs_dir):
        print(f"❌ Graphs directory not found: {graphs_dir}")
        print("💡 Run 'python generate_graphs.py' first to create graphs")
        return
    
    graphs = [
        ("speaker_distribution.png", "🎵 Speaker Distribution Analysis"),
        ("embedding_analysis.png", "🎯 Audio Embedding Space Analysis"), 
        ("similarity_analysis.png", "🔗 Speaker Similarity Analysis")
    ]
    
    print("🎨 Displaying Diarization Analysis Graphs")
    print("="*50)
    
    for filename, title in graphs:
        graph_path = os.path.join(graphs_dir, filename)
        if os.path.exists(graph_path):
            print(f"\n📊 Showing: {title}")
            display_graph(graph_path, title)
        else:
            print(f"⚠️  Graph not found: {filename}")
    
    # Open interactive graph in browser
    interactive_path = os.path.join(graphs_dir, "interactive_embeddings.html")
    if os.path.exists(interactive_path):
        print(f"\n🌐 Opening interactive visualization in browser...")
        try:
            webbrowser.open(f"file://{os.path.abspath(interactive_path)}")
            print(f"✅ Interactive graph opened in browser")
        except Exception as e:
            print(f"⚠️  Could not open browser. Open manually: {os.path.abspath(interactive_path)}")

def print_graph_summary():
    """Print summary of what each graph shows"""
    print("\n" + "="*70)
    print("📊 DIARIZATION GRAPHS SUMMARY")
    print("="*70)
    
    summaries = [
        {
            "name": "🎵 Speaker Distribution (speaker_distribution.png)",
            "description": """
Shows how audio segments are distributed across different bird speakers:
• Bar Chart: Number of segments per bird
• Pie Chart: Percentage of time each bird is active
• Horizontal Bar: Activity percentages
• Statistics: Overall distribution quality metrics
            """.strip()
        },
        {
            "name": "🎯 Embedding Analysis (embedding_analysis.png)", 
            "description": """
Visualizes the audio embedding space and clustering quality:
• t-SNE Plot: 2D visualization of high-dimensional embeddings
• PCA Plot: Principal component analysis showing main variation
• Density Map: Areas of high embedding concentration
• Quality Metrics: Silhouette score and clustering performance
            """.strip()
        },
        {
            "name": "🔗 Similarity Analysis (similarity_analysis.png)",
            "description": """
Shows relationships and similarities between different speakers:
• Similarity Heatmap: How similar each pair of birds sounds
• Dendrogram: Hierarchical clustering of bird voices
• Values close to 1.0 = very similar, close to 0.0 = very different
            """.strip()
        },
        {
            "name": "🌐 Interactive Visualization (interactive_embeddings.html)",
            "description": """
Interactive Plotly visualization with the following features:
• Hover over points to see segment details
• Zoom and pan the embedding space
• Toggle speakers on/off in the legend
• Full interactivity for detailed exploration
            """.strip()
        }
    ]
    
    for i, graph in enumerate(summaries, 1):
        print(f"\n{i}. {graph['name']}")
        print("-" * len(graph['name']))
        print(graph['description'])
    
    print(f"\n" + "="*70)
    print("💡 INTERPRETATION TIPS")
    print("="*70)
    print("""
🎯 Clustering Quality:
   • Silhouette Score: -1 to +1 (higher = better separation)
   • Good clustering: > 0.2, Excellent: > 0.5

🔍 What to Look For:
   • Clear clusters in t-SNE plot = good diarization
   • Similar colors grouped together = consistent speakers
   • High similarity in heatmap = potentially same bird

🐦 Speaker Analysis:
   • Uneven distribution = some birds more active
   • Many small clusters = potential over-segmentation
   • Few large clusters = potential under-segmentation
    """)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="View diarization analysis graphs")
    parser.add_argument("--graph", "-g", 
                       choices=["speaker", "embedding", "similarity", "interactive"],
                       help="Show specific graph type")
    parser.add_argument("--summary", "-s", action="store_true",
                       help="Show graph descriptions only")
    
    args = parser.parse_args()
    
    if args.summary:
        print_graph_summary()
        return
    
    graphs_dir = "graphs"
    
    if args.graph:
        # Show specific graph
        graph_files = {
            "speaker": ("speaker_distribution.png", "Speaker Distribution Analysis"),
            "embedding": ("embedding_analysis.png", "Audio Embedding Analysis"),
            "similarity": ("similarity_analysis.png", "Speaker Similarity Analysis"),
            "interactive": ("interactive_embeddings.html", "Interactive Visualization")
        }
        
        if args.graph == "interactive":
            interactive_path = os.path.join(graphs_dir, graph_files[args.graph][0])
            if os.path.exists(interactive_path):
                webbrowser.open(f"file://{os.path.abspath(interactive_path)}")
                print(f"🌐 Opened interactive visualization in browser")
            else:
                print(f"❌ Interactive graph not found: {interactive_path}")
        else:
            filename, title = graph_files[args.graph]
            graph_path = os.path.join(graphs_dir, filename)
            display_graph(graph_path, title)
    else:
        # Show all graphs
        show_all_graphs()
        print_graph_summary()

if __name__ == "__main__":
    main()