#!/usr/bin/env python3
"""
view_metrics.py - View and Analyze Diarization Metrics

Usage:
    python view_metrics.py                           # Show all result files
    python view_metrics.py hoopoe_results.json      # Show specific file
    python view_metrics.py --compare                 # Compare all files
"""

import json
import glob
import argparse
from pathlib import Path
# import pandas as pd  # Not required for basic functionality

def load_results(file_path):
    """Load results from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def format_metrics(metrics):
    """Format metrics for display"""
    if not metrics:
        return "No metrics available"
    
    formatted = []
    formatted.append(f"  Clustering Quality:")
    formatted.append(f"    Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.4f}")
    formatted.append(f"    Calinski-Harabasz: {metrics.get('calinski_harabasz_score', 'N/A'):.2f}")
    formatted.append(f"    Davies-Bouldin: {metrics.get('davies_bouldin_score', 'N/A'):.4f}")
    
    formatted.append(f"  Cluster Statistics:")
    formatted.append(f"    Number of Clusters: {metrics.get('n_clusters', 'N/A')}")
    formatted.append(f"    Cluster Sizes: {metrics.get('cluster_sizes', 'N/A')}")
    formatted.append(f"    Min Cluster Size: {metrics.get('min_cluster_size', 'N/A')}")
    formatted.append(f"    Max Cluster Size: {metrics.get('max_cluster_size', 'N/A')}")
    formatted.append(f"    Cluster Balance: {metrics.get('cluster_balance', 'N/A'):.4f}")
    
    return "\n".join(formatted)

def format_timeline(timeline):
    """Format timeline for display"""
    if not timeline:
        return "No timeline available"
    
    formatted = []
    formatted.append(f"  Speaker Timeline:")
    total_duration = 0
    
    for segment in timeline:
        speaker = segment.get('speaker', 'Unknown')
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        duration = segment.get('duration', 0)
        total_duration += duration
        
        formatted.append(f"    {speaker}: {start:.1f}s - {end:.1f}s ({duration:.1f}s)")
    
    formatted.append(f"  Total Active Time: {total_duration:.1f}s")
    return "\n".join(formatted)

def interpret_metrics(metrics):
    """Provide interpretation of metrics"""
    if not metrics:
        return "No metrics to interpret"
    
    interpretations = []
    interpretations.append("  Quality Interpretation:")
    
    # Silhouette Score
    sil = metrics.get('silhouette_score', 0)
    if sil > 0.7:
        sil_qual = "Excellent"
    elif sil > 0.5:
        sil_qual = "Good"
    elif sil > 0.2:
        sil_qual = "Fair"
    else:
        sil_qual = "Poor"
    interpretations.append(f"    Silhouette ({sil:.3f}): {sil_qual} cluster separation")
    
    # Calinski-Harabasz
    ch = metrics.get('calinski_harabasz_score', 0)
    if ch > 100:
        ch_qual = "Good"
    elif ch > 50:
        ch_qual = "Fair"
    else:
        ch_qual = "Poor"
    interpretations.append(f"    Calinski-Harabasz ({ch:.1f}): {ch_qual} cluster definition")
    
    # Davies-Bouldin
    db = metrics.get('davies_bouldin_score', float('inf'))
    if db < 1.0:
        db_qual = "Good"
    elif db < 2.0:
        db_qual = "Fair"
    else:
        db_qual = "Poor"
    interpretations.append(f"    Davies-Bouldin ({db:.3f}): {db_qual} inter-cluster separation")
    
    # Cluster Balance
    balance = metrics.get('cluster_balance', 0)
    if balance < 0.5:
        balance_qual = "Well-balanced"
    elif balance < 1.0:
        balance_qual = "Moderately balanced"
    else:
        balance_qual = "Imbalanced"
    interpretations.append(f"    Cluster Balance ({balance:.3f}): {balance_qual} speaker distribution")
    
    return "\n".join(interpretations)

def display_single_result(file_path):
    """Display results from a single file"""
    print(f"=" * 80)
    print(f"DIARIZATION RESULTS: {file_path}")
    print(f"=" * 80)
    
    results = load_results(file_path)
    if not results:
        return
    
    # Basic information
    print(f"Input File: {results.get('input_file', 'N/A')}")
    print(f"Duration: {results.get('duration', 'N/A'):.1f} seconds")
    print(f"Segments: {results.get('n_segments', 'N/A')}")
    print(f"Segment Length: {results.get('segment_length', 'N/A')}s")
    print(f"Method: {results.get('method', 'N/A')}")
    print(f"Detected Speakers: {results.get('n_speakers', 'N/A')}")
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    print()
    
    # Metrics
    print("CLUSTERING METRICS:")
    print(format_metrics(results.get('metrics', {})))
    print()
    
    # Interpretation
    print("QUALITY INTERPRETATION:")
    print(interpret_metrics(results.get('metrics', {})))
    print()
    
    # Timeline
    print("SPEAKER TIMELINE:")
    print(format_timeline(results.get('timeline', [])))
    print()

def compare_all_results():
    """Compare metrics across all result files"""
    print(f"=" * 100)
    print(f"COMPARISON OF ALL DIARIZATION RESULTS")
    print(f"=" * 100)
    
    # Find all JSON result files
    json_files = glob.glob("*_results.json")
    
    if not json_files:
        print("No result files found (looking for *_results.json)")
        return
    
    # Load all results
    all_results = []
    for file_path in json_files:
        results = load_results(file_path)
        if results:
            # Extract key info
            row = {
                'File': Path(file_path).stem.replace('_results', ''),
                'Input': Path(results.get('input_file', '')).name,
                'Duration': results.get('duration', 0),
                'Speakers': results.get('n_speakers', 0),
                'Method': results.get('method', 'N/A'),
                'Silhouette': results.get('metrics', {}).get('silhouette_score', 0),
                'Calinski-H': results.get('metrics', {}).get('calinski_harabasz_score', 0),
                'Davies-B': results.get('metrics', {}).get('davies_bouldin_score', float('inf')),
                'Balance': results.get('metrics', {}).get('cluster_balance', 0)
            }
            all_results.append(row)
    
    if not all_results:
        print("No valid results found")
        return
    
    # Create DataFrame for nice formatting
    df = pd.DataFrame(all_results)
    
    print("SUMMARY TABLE:")
    print("-" * 100)
    print(f"{'File':<12} {'Input':<20} {'Dur':<6} {'Spk':<3} {'Method':<10} {'Silhouette':<10} {'Calinski':<9} {'Davies':<7} {'Balance':<7}")
    print("-" * 100)
    
    for _, row in df.iterrows():
        print(f"{row['File']:<12} {row['Input']:<20} {row['Duration']:<6.1f} {row['Speakers']:<3} {row['Method']:<10} {row['Silhouette']:<10.3f} {row['Calinski']:<9.1f} {row['Davies']:<7.3f} {row['Balance']:<7.3f}")
    
    print("-" * 100)
    
    # Statistics
    print("\nSTATISTICS:")
    print(f"  Total Files: {len(all_results)}")
    print(f"  Average Speakers: {df['Speakers'].mean():.1f}")
    print(f"  Average Silhouette: {df['Silhouette'].mean():.3f}")
    print(f"  Average Duration: {df['Duration'].mean():.1f}s")
    
    # Best/Worst
    best_sil = df.loc[df['Silhouette'].idxmax()]
    worst_sil = df.loc[df['Silhouette'].idxmin()]
    
    print(f"\nBEST CLUSTERING (Highest Silhouette):")
    print(f"  {best_sil['File']}: {best_sil['Silhouette']:.3f} ({best_sil['Speakers']} speakers)")
    
    print(f"\nWORST CLUSTERING (Lowest Silhouette):")
    print(f"  {worst_sil['File']}: {worst_sil['Silhouette']:.3f} ({worst_sil['Speakers']} speakers)")

def main():
    parser = argparse.ArgumentParser(description="View bird diarization metrics")
    parser.add_argument('file', nargs='?', help='JSON result file to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare all result files')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_all_results()
    elif args.file:
        if Path(args.file).exists():
            display_single_result(args.file)
        else:
            print(f"File not found: {args.file}")
    else:
        # Show all available files
        json_files = glob.glob("*_results.json")
        
        if not json_files:
            print("No result files found (looking for *_results.json)")
            print("\nTo generate results, run:")
            print("python infer_birds.py --audio /path/to/audio.wav --model models/best_enhanced_model.pt --output results.json")
            return
        
        print("Available result files:")
        for file_path in json_files:
            results = load_results(file_path)
            if results:
                speakers = results.get('n_speakers', 'N/A')
                duration = results.get('duration', 0)
                sil_score = results.get('metrics', {}).get('silhouette_score', 0)
                print(f"  {file_path}: {speakers} speakers, {duration:.1f}s, silhouette={sil_score:.3f}")
        
        print(f"\nUsage:")
        print(f"  python view_metrics.py <filename>     # View specific file")
        print(f"  python view_metrics.py --compare      # Compare all files")

if __name__ == "__main__":
    main()