#!/usr/bin/env python3
"""
advanced_clustering.py - Enhanced Clustering for Bird Diarization

Features:
- Multiple clustering algorithms (KMeans, DBSCAN, GMM, Hierarchical)
- Ensemble clustering with voting
- Advanced cluster evaluation metrics
- Automatic optimal cluster number detection
- Post-processing and temporal smoothing
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                           davies_bouldin_score, adjusted_rand_score,
                           normalized_mutual_info_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import medfilt
import warnings
warnings.filterwarnings('ignore')

class AdvancedClusteringStrategy:
    """Advanced clustering with multiple algorithms and ensemble methods"""
    
    def __init__(self, min_speakers=2, max_speakers=8, fast_mode=False):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.fast_mode = fast_mode
        
        # Initialize clustering methods
        if self.fast_mode:
            # Fast mode: only use fastest methods
            self.clustering_methods = {
                'kmeans': self._kmeans_clustering,
                'hierarchical': self._hierarchical_clustering
            }
            print("🚀 Fast clustering mode enabled (kmeans + hierarchical only)")
        else:
            # Full mode: use all methods
            self.clustering_methods = {
                'kmeans': self._kmeans_clustering,
                'gmm': self._gmm_clustering,
                'hierarchical': self._hierarchical_clustering,
                'dbscan': self._dbscan_clustering,
                'spectral': self._spectral_clustering
            }
        
        self.best_results = {}
        
    def find_optimal_clusters(self, embeddings, method='comprehensive'):
        """
        Find optimal number of clusters using multiple evaluation metrics
        
        Args:
            embeddings: numpy array of shape [n_samples, embed_dim]
            method: 'comprehensive', 'silhouette', 'elbow', or 'gap_statistic'
        
        Returns:
            dict with results from all clustering methods
        """
        print(f"🔍 Finding optimal clusters using {method} evaluation...")
        
        # Preprocess embeddings
        embeddings_processed = self._preprocess_embeddings(embeddings)
        
        all_results = {}
        
        for method_name, clustering_func in self.clustering_methods.items():
            print(f"   Testing {method_name}...")
            try:
                result = clustering_func(embeddings_processed)
                all_results[method_name] = result
            except Exception as e:
                print(f"   ⚠️  {method_name} failed: {e}")
                continue
        
        # Find best method based on comprehensive evaluation
        best_method = self._select_best_method(all_results, embeddings_processed)
        
        print(f"✅ Best method: {best_method['method']} with {best_method['n_clusters']} clusters")
        print(f"   Silhouette score: {best_method['silhouette']:.3f}")
        
        return best_method, all_results
    
    def _preprocess_embeddings(self, embeddings):
        """Preprocess embeddings for better clustering"""
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Optional: Apply PCA for dimensionality reduction if embeddings are very high-dimensional
        if embeddings.shape[1] > 512:
            pca = PCA(n_components=min(256, embeddings.shape[0] - 1))
            embeddings_scaled = pca.fit_transform(embeddings_scaled)
            print(f"   Applied PCA: {embeddings.shape[1]} -> {embeddings_scaled.shape[1]} dims")
        
        return embeddings_scaled
    
    def _kmeans_clustering(self, embeddings):
        """Enhanced K-means clustering with multiple initializations"""
        best_score = -1
        best_result = None
        
        for n_clusters in range(self.min_speakers, min(self.max_speakers + 1, len(embeddings))):
            # Multiple random initializations for robustness
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=42, 
                n_init=20,  # More initializations
                max_iter=500,
                algorithm='lloyd'
            )
            
            labels = kmeans.fit_predict(embeddings)
            
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(embeddings, labels)
                calinski = calinski_harabasz_score(embeddings, labels)
                davies = davies_bouldin_score(embeddings, labels)
                
                # Composite score (lower davies_bouldin is better)
                composite_score = silhouette + (calinski / 1000) - (davies / 10)
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_result = {
                        'method': 'kmeans',
                        'n_clusters': n_clusters,
                        'labels': labels,
                        'silhouette': silhouette,
                        'calinski_harabasz': calinski,
                        'davies_bouldin': davies,
                        'composite_score': composite_score,
                        'cluster_centers': kmeans.cluster_centers_,
                        'inertia': kmeans.inertia_
                    }
        
        return best_result
    
    def _gmm_clustering(self, embeddings):
        """Gaussian Mixture Model clustering with model selection"""
        best_score = -1
        best_result = None
        
        for n_components in range(self.min_speakers, min(self.max_speakers + 1, len(embeddings))):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42,
                    max_iter=200
                )
                
                labels = gmm.fit_predict(embeddings)
                
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(embeddings, labels)
                    bic = gmm.bic(embeddings)
                    aic = gmm.aic(embeddings)
                    
                    # Use silhouette with BIC penalty (lower BIC is better)
                    composite_score = silhouette - (bic / 10000)
                    
                    if composite_score > best_score:
                        best_score = composite_score
                        best_result = {
                            'method': 'gmm',
                            'n_clusters': n_components,
                            'labels': labels,
                            'silhouette': silhouette,
                            'bic': bic,
                            'aic': aic,
                            'composite_score': composite_score,
                            'log_likelihood': gmm.score(embeddings),
                            'converged': gmm.converged_
                        }
            except Exception as e:
                continue
        
        return best_result
    
    def _hierarchical_clustering(self, embeddings):
        """Hierarchical clustering with different linkage methods"""
        best_score = -1
        best_result = None
        
        linkage_methods = ['ward', 'complete', 'average']
        
        for linkage_method in linkage_methods:
            for n_clusters in range(self.min_speakers, min(self.max_speakers + 1, len(embeddings))):
                try:
                    hierarchical = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        linkage=linkage_method
                    )
                    
                    labels = hierarchical.fit_predict(embeddings)
                    
                    if len(np.unique(labels)) > 1:
                        silhouette = silhouette_score(embeddings, labels)
                        
                        if silhouette > best_score:
                            best_score = silhouette
                            best_result = {
                                'method': f'hierarchical_{linkage_method}',
                                'n_clusters': n_clusters,
                                'labels': labels,
                                'silhouette': silhouette,
                                'linkage_method': linkage_method,
                                'composite_score': silhouette
                            }
                except Exception as e:
                    continue
        
        return best_result
    
    def _dbscan_clustering(self, embeddings):
        """DBSCAN clustering with parameter optimization"""
        best_score = -1
        best_result = None
        
        # Try different epsilon values
        eps_values = np.linspace(0.1, 2.0, 20)
        min_samples_values = [3, 5, 7, 10]
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
                    labels = dbscan.fit_predict(embeddings)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    if n_clusters >= self.min_speakers and n_clusters <= self.max_speakers and n_clusters > 1:
                        # Remove noise points for silhouette calculation
                        if -1 in labels:
                            mask = labels != -1
                            if mask.sum() > 1 and len(np.unique(labels[mask])) > 1:
                                silhouette = silhouette_score(embeddings[mask], labels[mask])
                            else:
                                continue
                        else:
                            silhouette = silhouette_score(embeddings, labels)
                        
                        noise_ratio = (labels == -1).sum() / len(labels)
                        
                        # Penalize high noise ratio
                        composite_score = silhouette * (1 - noise_ratio * 0.5)
                        
                        if composite_score > best_score:
                            best_score = composite_score
                            best_result = {
                                'method': 'dbscan',
                                'n_clusters': n_clusters,
                                'labels': labels,
                                'silhouette': silhouette,
                                'noise_ratio': noise_ratio,
                                'eps': eps,
                                'min_samples': min_samples,
                                'composite_score': composite_score
                            }
                except Exception as e:
                    continue
        
        return best_result
    
    def _spectral_clustering(self, embeddings):
        """Spectral clustering with different affinity methods (optimized for speed)"""
        best_score = -1
        best_result = None
        
        # Skip spectral clustering for very large datasets (too slow)
        if len(embeddings) > 500:
            print("   Skipping spectral (too many samples)")
            return None
        
        # Only test nearest_neighbors (faster than rbf)
        affinity_methods = ['nearest_neighbors']
        
        # Test fewer cluster numbers (focus on likely range)
        likely_clusters = [2, 3, 4, 5]  # Most common for bird recordings
        cluster_range = [c for c in likely_clusters if self.min_speakers <= c <= min(self.max_speakers, len(embeddings)-1)]
        
        for affinity in affinity_methods:
            for n_clusters in cluster_range:
                try:
                    spectral = SpectralClustering(
                        n_clusters=n_clusters,
                        affinity=affinity,
                        random_state=42,
                        n_init=3,  # Reduced from 10 to 3
                        assign_labels='cluster_qr'  # Faster assignment method
                    )
                    
                    labels = spectral.fit_predict(embeddings)
                    
                    if len(np.unique(labels)) > 1:
                        silhouette = silhouette_score(embeddings, labels)
                        
                        if silhouette > best_score:
                            best_score = silhouette
                            best_result = {
                                'method': f'spectral_{affinity}',
                                'n_clusters': n_clusters,
                                'labels': labels,
                                'silhouette': silhouette,
                                'affinity': affinity,
                                'composite_score': silhouette
                            }
                except Exception as e:
                    continue
        
        return best_result
    
    def _select_best_method(self, all_results, embeddings):
        """Select best clustering method based on comprehensive evaluation"""
        if not all_results:
            raise ValueError("No clustering methods succeeded")
        
        # Rank methods by composite score
        method_scores = []
        
        for method_name, result in all_results.items():
            if result is not None:
                # Additional evaluation criteria
                score = result['composite_score']
                
                # Bonus for reasonable number of clusters
                n_clusters = result['n_clusters']
                if 3 <= n_clusters <= 8:  # Sweet spot for bird diarization
                    score += 0.05
                
                # Penalty for too many or too few clusters
                if n_clusters < 2 or n_clusters > 12:
                    score -= 0.1
                
                method_scores.append((score, method_name, result))
        
        # Sort by score (descending)
        method_scores.sort(key=lambda x: x[0], reverse=True)
        
        if not method_scores:
            raise ValueError("All clustering methods failed")
        
        return method_scores[0][2]  # Return best result
    
    def ensemble_clustering(self, embeddings, top_k=3):
        """Ensemble clustering using multiple methods with voting"""
        print("🗳️  Performing ensemble clustering...")
        
        _, all_results = self.find_optimal_clusters(embeddings)
        
        # Get top-k methods
        valid_results = [(r['composite_score'], name, r) for name, r in all_results.items() if r is not None]
        valid_results.sort(key=lambda x: x[0], reverse=True)
        top_methods = valid_results[:top_k]
        
        if len(top_methods) < 2:
            print("⚠️  Not enough methods for ensemble, using single best method")
            return top_methods[0][2] if top_methods else None
        
        # Collect all label predictions
        all_labels = [result[2]['labels'] for result in top_methods]
        n_samples = len(all_labels[0])
        
        # Ensemble voting using consensus clustering
        final_labels = self._consensus_clustering(all_labels, embeddings)
        
        # Evaluate ensemble result
        if len(np.unique(final_labels)) > 1:
            silhouette = silhouette_score(embeddings, final_labels)
        else:
            silhouette = -1
        
        ensemble_result = {
            'method': 'ensemble',
            'n_clusters': len(np.unique(final_labels)),
            'labels': final_labels,
            'silhouette': silhouette,
            'composite_score': silhouette,
            'constituent_methods': [method[1] for method in top_methods]
        }
        
        print(f"✅ Ensemble result: {ensemble_result['n_clusters']} clusters, silhouette: {silhouette:.3f}")
        
        return ensemble_result
    
    def _consensus_clustering(self, all_labels, embeddings):
        """Create consensus labels from multiple clustering results"""
        n_samples = len(all_labels[0])
        n_methods = len(all_labels)
        
        # Create co-association matrix
        co_assoc_matrix = np.zeros((n_samples, n_samples))
        
        for labels in all_labels:
            for i in range(n_samples):
                for j in range(n_samples):
                    if labels[i] == labels[j]:
                        co_assoc_matrix[i, j] += 1
        
        # Normalize by number of methods
        co_assoc_matrix /= n_methods
        
        # Convert to distance matrix
        distance_matrix = 1 - co_assoc_matrix
        
        # Apply hierarchical clustering to the consensus matrix
        try:
            # Try different numbers of clusters and pick the best
            best_silhouette = -1
            best_labels = None
            
            for n_clusters in range(self.min_speakers, min(self.max_speakers + 1, n_samples)):
                hierarchical = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='average',
                    metric='precomputed'
                )
                
                labels = hierarchical.fit_predict(distance_matrix)
                
                if len(np.unique(labels)) > 1:
                    silhouette = silhouette_score(embeddings, labels)
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_labels = labels
            
            return best_labels if best_labels is not None else all_labels[0]
            
        except Exception as e:
            print(f"   Consensus clustering failed, using best single method: {e}")
            return all_labels[0]
    
    def temporal_smoothing(self, labels, window_size=5, min_duration=3):
        """Apply temporal smoothing to remove rapid speaker changes"""
        print(f"🔄 Applying temporal smoothing (window={window_size}, min_duration={min_duration})...")
        
        if len(labels) < window_size:
            return labels
        
        # Apply median filtering for smoothing
        smoothed_labels = medfilt(labels.astype(float), kernel_size=window_size).astype(int)
        
        # Enforce minimum segment duration
        current_label = smoothed_labels[0]
        current_count = 1
        final_labels = [current_label]
        
        for i in range(1, len(smoothed_labels)):
            if smoothed_labels[i] == current_label:
                current_count += 1
            else:
                # If current segment is too short, extend previous segment
                if current_count < min_duration and len(final_labels) >= min_duration:
                    # Replace short segment with most frequent neighbor
                    for j in range(current_count):
                        if len(final_labels) > j:
                            final_labels[-j-1] = final_labels[-current_count-1] if len(final_labels) > current_count else current_label
                
                current_label = smoothed_labels[i]
                current_count = 1
            
            final_labels.append(current_label)
        
        # Handle last segment
        if current_count < min_duration and len(final_labels) >= min_duration:
            for j in range(current_count):
                if len(final_labels) > j:
                    final_labels[-j-1] = final_labels[-current_count-1]
        
        return np.array(final_labels[:len(labels)])  # Ensure same length
    
    def evaluate_clustering_quality(self, embeddings, labels, true_labels=None):
        """Comprehensive clustering quality evaluation"""
        metrics = {}
        
        if len(np.unique(labels)) <= 1:
            return {'error': 'Only one cluster found'}
        
        # Unsupervised metrics
        try:
            metrics['silhouette_score'] = silhouette_score(embeddings, labels)
        except:
            metrics['silhouette_score'] = -1
        
        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, labels)
        except:
            metrics['calinski_harabasz_score'] = 0
        
        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, labels)
        except:
            metrics['davies_bouldin_score'] = float('inf')
        
        # Cluster statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['n_clusters'] = len(unique_labels)
        metrics['cluster_sizes'] = counts
        metrics['min_cluster_size'] = counts.min()
        metrics['max_cluster_size'] = counts.max()
        metrics['cluster_balance'] = counts.std() / counts.mean() if counts.mean() > 0 else float('inf')
        
        # If true labels available (for validation)
        if true_labels is not None:
            try:
                metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, labels)
                metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, labels)
            except:
                pass
        
        return metrics

def perform_advanced_diarization(embeddings, max_speakers=8, fast_mode=False):
    """
    Main function to perform advanced diarization clustering
    
    Args:
        embeddings: numpy array of audio embeddings
        max_speakers: maximum number of speakers to consider
        fast_mode: if True, use only fastest clustering methods
    
    Returns:
        dict with clustering results and evaluation metrics
    """
    clustering_strategy = AdvancedClusteringStrategy(max_speakers=max_speakers, fast_mode=fast_mode)
    
    print(f"🎯 Starting advanced diarization on {len(embeddings)} segments...")
    
    # Find optimal clustering
    if len(embeddings) > 20:  # Use ensemble for larger datasets
        best_result = clustering_strategy.ensemble_clustering(embeddings)
    else:
        best_result, _ = clustering_strategy.find_optimal_clusters(embeddings)
    
    if best_result is None:
        print("❌ All clustering methods failed!")
        return None
    
    # Apply temporal smoothing
    smoothed_labels = clustering_strategy.temporal_smoothing(best_result['labels'])
    
    # Evaluate final result
    final_metrics = clustering_strategy.evaluate_clustering_quality(embeddings, smoothed_labels)
    
    final_result = {
        'labels': smoothed_labels,
        'n_speakers': len(np.unique(smoothed_labels)),
        'method': best_result['method'],
        'metrics': final_metrics,
        'raw_labels': best_result['labels']
    }
    
    print(f"✅ Advanced diarization complete!")
    print(f"   Method: {final_result['method']}")
    print(f"   Speakers: {final_result['n_speakers']}")
    silhouette_score_val = final_metrics.get('silhouette_score', 'N/A')
    if isinstance(silhouette_score_val, (int, float)):
        print(f"   Silhouette: {silhouette_score_val:.3f}")
    else:
        print(f"   Silhouette: {silhouette_score_val}")
    
    return final_result

if __name__ == "__main__":
    print("Testing Advanced Clustering...")
    
    # Create synthetic embeddings for testing
    np.random.seed(42)
    n_samples = 100
    n_features = 128
    
    # Create embeddings with some structure (3 clusters)
    cluster1 = np.random.randn(30, n_features) + [2, 2, 0, 0] + [0] * (n_features - 4)
    cluster2 = np.random.randn(35, n_features) + [-2, 2, 0, 0] + [0] * (n_features - 4)
    cluster3 = np.random.randn(35, n_features) + [0, -2, 0, 0] + [0] * (n_features - 4)
    
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([0] * 30 + [1] * 35 + [2] * 35)
    
    print(f"Test data: {embeddings.shape} embeddings, 3 true clusters")
    
    # Test the clustering
    result = perform_advanced_diarization(embeddings, max_speakers=6)
    
    if result:
        print(f"\nResults:")
        print(f"Detected clusters: {result['n_speakers']}")
        print(f"Silhouette score: {result['metrics']['silhouette_score']:.3f}")
        
        # Compare with true labels
        ari = adjusted_rand_score(true_labels, result['labels'])
        print(f"Adjusted Rand Index vs true labels: {ari:.3f}")
    
    print("✅ Advanced clustering test complete!")