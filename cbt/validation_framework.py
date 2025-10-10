#!/usr/bin/env python3
"""
validation_framework.py - Comprehensive Validation Framework for Bird Diarization

Features:
- Proper train/validation/test splits
- Comprehensive evaluation metrics
- Cross-validation for robust evaluation
- Model comparison utilities
- Statistical significance testing
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    calinski_harabasz_score, davies_bouldin_score
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class ValidationFramework:
    """Comprehensive validation framework for bird diarization"""
    
    def __init__(self, val_ratio=0.2, test_ratio=0.1, random_state=42):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.results_history = []
        
    def create_splits(self, dataset, stratify_by_file=True):
        """
        Create train/validation/test splits
        
        Args:
            dataset: Dataset object
            stratify_by_file: Whether to stratify splits by file origin
        
        Returns:
            dict with train/val/test datasets
        """
        print("📊 Creating train/validation/test splits...")
        
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        
        # Create stratification labels if needed
        if stratify_by_file and hasattr(dataset, 'file_ids'):
            stratify_labels = [dataset.file_ids[i] for i in indices]
            # Convert file IDs to numeric labels
            unique_files = list(set(stratify_labels))
            label_map = {file_id: i for i, file_id in enumerate(unique_files)}
            stratify_labels = [label_map[file_id] for file_id in stratify_labels]
        else:
            stratify_labels = None
        
        # First split: separate test set
        if self.test_ratio > 0:
            train_val_indices, test_indices = train_test_split(
                indices, 
                test_size=self.test_ratio,
                random_state=self.random_state,
                stratify=stratify_labels if stratify_labels else None
            )
        else:
            train_val_indices = indices
            test_indices = []
        
        # Second split: separate validation from training
        if self.val_ratio > 0 and len(train_val_indices) > 1:
            if stratify_labels:
                stratify_train_val = [stratify_labels[i] for i in train_val_indices]
            else:
                stratify_train_val = None
                
            train_indices, val_indices = train_test_split(
                train_val_indices,
                test_size=self.val_ratio / (1 - self.test_ratio),
                random_state=self.random_state,
                stratify=stratify_train_val
            )
        else:
            train_indices = train_val_indices
            val_indices = []
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices) if val_indices else None
        test_dataset = torch.utils.data.Subset(dataset, test_indices) if test_indices else None
        
        split_info = {
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'test_size': len(test_indices),
            'total_size': dataset_size
        }
        
        print(f"   Train: {split_info['train_size']} samples")
        print(f"   Val: {split_info['val_size']} samples") 
        print(f"   Test: {split_info['test_size']} samples")
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'info': split_info
        }
    
    def compute_diarization_metrics(self, embeddings, predicted_labels, true_labels=None, method_name="Unknown"):
        """
        Compute comprehensive diarization metrics
        
        Args:
            embeddings: numpy array of embeddings
            predicted_labels: predicted cluster labels
            true_labels: ground truth labels (if available)
            method_name: name of the method being evaluated
        
        Returns:
            dict with all computed metrics
        """
        metrics = {
            'method': method_name,
            'n_samples': len(embeddings),
            'n_predicted_clusters': len(np.unique(predicted_labels))
        }
        
        # Unsupervised clustering metrics
        if len(np.unique(predicted_labels)) > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(embeddings, predicted_labels)
            except:
                metrics['silhouette_score'] = -1
            
            try:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, predicted_labels)
            except:
                metrics['calinski_harabasz_score'] = 0
                
            try:
                metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, predicted_labels)
            except:
                metrics['davies_bouldin_score'] = float('inf')
        else:
            metrics['silhouette_score'] = -1
            metrics['calinski_harabasz_score'] = 0
            metrics['davies_bouldin_score'] = float('inf')
        
        # Cluster distribution metrics
        unique_labels, counts = np.unique(predicted_labels, return_counts=True)
        metrics['cluster_sizes'] = counts.tolist()
        metrics['min_cluster_size'] = int(counts.min())
        metrics['max_cluster_size'] = int(counts.max())
        metrics['cluster_size_std'] = float(counts.std())
        metrics['cluster_balance'] = float(counts.std() / counts.mean()) if counts.mean() > 0 else float('inf')
        
        # If ground truth available (for validation/testing)
        if true_labels is not None:
            metrics['n_true_clusters'] = len(np.unique(true_labels))
            
            try:
                metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_labels)
            except:
                metrics['adjusted_rand_score'] = 0
                
            try:
                metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, predicted_labels)
            except:
                metrics['normalized_mutual_info'] = 0
                
            try:
                metrics['homogeneity_score'] = homogeneity_score(true_labels, predicted_labels)
                metrics['completeness_score'] = completeness_score(true_labels, predicted_labels)
                metrics['v_measure_score'] = v_measure_score(true_labels, predicted_labels)
            except:
                metrics['homogeneity_score'] = 0
                metrics['completeness_score'] = 0
                metrics['v_measure_score'] = 0
        
        return metrics
    
    def cross_validate_model(self, model, dataset, k_folds=5, max_speakers=8):
        """
        Perform k-fold cross-validation
        
        Args:
            model: trained model
            dataset: full dataset
            k_folds: number of folds
            max_speakers: maximum number of speakers
        
        Returns:
            dict with cross-validation results
        """
        print(f"🔄 Performing {k_folds}-fold cross-validation...")
        
        # Create stratification labels based on file IDs
        if hasattr(dataset, 'file_ids'):
            stratify_labels = dataset.file_ids
            unique_files = list(set(stratify_labels))
            label_map = {file_id: i for i, file_id in enumerate(unique_files)}
            stratify_labels = [label_map[file_id] for file_id in stratify_labels]
        else:
            stratify_labels = None
        
        # Initialize cross-validation
        if stratify_labels and len(set(stratify_labels)) >= k_folds:
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
            cv_splits = list(cv.split(range(len(dataset)), stratify_labels))
        else:
            # Fall back to regular k-fold if stratification not possible
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
            cv_splits = list(cv.split(range(len(dataset))))
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"   Fold {fold + 1}/{k_folds}...")
            
            # Create fold datasets
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)
            
            # Extract embeddings for validation set
            embeddings = []
            model.eval()
            with torch.no_grad():
                for batch_data in val_loader:
                    x1, x2, _ = batch_data
                    # Use first view for evaluation
                    emb = model(x1.cuda() if torch.cuda.is_available() else x1)
                    embeddings.append(emb.cpu().numpy())
            
            embeddings = np.vstack(embeddings)
            
            # Perform clustering
            from advanced_clustering import perform_advanced_diarization
            clustering_result = perform_advanced_diarization(embeddings, max_speakers=max_speakers)
            
            if clustering_result:
                # Compute metrics
                fold_metrics = self.compute_diarization_metrics(
                    embeddings, 
                    clustering_result['labels'],
                    method_name=f"CV_Fold_{fold+1}"
                )
                fold_results.append(fold_metrics)
            else:
                print(f"   ⚠️  Fold {fold + 1} clustering failed")
        
        # Aggregate results
        if fold_results:
            cv_summary = self._aggregate_cv_results(fold_results)
            print(f"✅ Cross-validation complete. Mean silhouette: {cv_summary['silhouette_score_mean']:.3f} ± {cv_summary['silhouette_score_std']:.3f}")
            return cv_summary
        else:
            print("❌ All folds failed")
            return None
    
    def _aggregate_cv_results(self, fold_results):
        """Aggregate cross-validation results across folds"""
        metrics_to_aggregate = [
            'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score',
            'cluster_balance', 'n_predicted_clusters'
        ]
        
        # Add supervised metrics if available
        if 'adjusted_rand_score' in fold_results[0]:
            metrics_to_aggregate.extend([
                'adjusted_rand_score', 'normalized_mutual_info',
                'homogeneity_score', 'completeness_score', 'v_measure_score'
            ])
        
        aggregated = {}
        
        for metric in metrics_to_aggregate:
            values = [result[metric] for result in fold_results if metric in result and np.isfinite(result[metric])]
            if values:
                aggregated[f"{metric}_mean"] = np.mean(values)
                aggregated[f"{metric}_std"] = np.std(values)
                aggregated[f"{metric}_values"] = values
        
        aggregated['n_folds'] = len(fold_results)
        aggregated['fold_results'] = fold_results
        
        return aggregated
    
    def compare_methods(self, embeddings, method_results, save_path=None):
        """
        Compare multiple clustering methods
        
        Args:
            embeddings: audio embeddings
            method_results: dict of {method_name: labels}
            save_path: path to save comparison results
        
        Returns:
            comparison results and plots
        """
        print("📈 Comparing clustering methods...")
        
        comparison_results = []
        
        for method_name, labels in method_results.items():
            metrics = self.compute_diarization_metrics(embeddings, labels, method_name=method_name)
            comparison_results.append(metrics)
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_results)
        
        # Generate comparison plots
        if len(comparison_results) > 1:
            fig = self._create_comparison_plots(df)
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                df.to_csv(save_path.replace('.png', '.csv'), index=False)
                print(f"   Saved comparison to {save_path}")
        
        # Find best method
        if 'silhouette_score' in df.columns:
            best_idx = df['silhouette_score'].idxmax()
            best_method = df.iloc[best_idx]
            print(f"✅ Best method: {best_method['method']} (Silhouette: {best_method['silhouette_score']:.3f})")
            
            return {
                'comparison_df': df,
                'best_method': best_method.to_dict(),
                'all_results': comparison_results
            }
        
        return {
            'comparison_df': df,
            'all_results': comparison_results
        }
    
    def _create_comparison_plots(self, df):
        """Create visualization plots for method comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🔍 Clustering Method Comparison', fontsize=16, fontweight='bold')
        
        # 1. Silhouette Score Comparison
        if 'silhouette_score' in df.columns:
            ax1 = axes[0, 0]
            bars = ax1.bar(df['method'], df['silhouette_score'], alpha=0.8, color='skyblue')
            ax1.set_title('Silhouette Score Comparison')
            ax1.set_ylabel('Silhouette Score')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, df['silhouette_score']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Number of Clusters
        if 'n_predicted_clusters' in df.columns:
            ax2 = axes[0, 1]
            bars = ax2.bar(df['method'], df['n_predicted_clusters'], alpha=0.8, color='lightcoral')
            ax2.set_title('Number of Detected Clusters')
            ax2.set_ylabel('Number of Clusters')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, df['n_predicted_clusters']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{int(value)}', ha='center', va='bottom')
        
        # 3. Cluster Balance (lower is better)
        if 'cluster_balance' in df.columns:
            ax3 = axes[1, 0]
            bars = ax3.bar(df['method'], df['cluster_balance'], alpha=0.8, color='lightgreen')
            ax3.set_title('Cluster Balance (Lower = Better)')
            ax3.set_ylabel('Cluster Balance')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, df['cluster_balance']):
                if np.isfinite(value):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.2f}', ha='center', va='bottom')
        
        # 4. Multiple metrics radar chart or Davies-Bouldin score
        ax4 = axes[1, 1]
        if 'davies_bouldin_score' in df.columns:
            # Filter out infinite values
            finite_mask = np.isfinite(df['davies_bouldin_score'])
            if finite_mask.any():
                bars = ax4.bar(df.loc[finite_mask, 'method'], 
                             df.loc[finite_mask, 'davies_bouldin_score'], 
                             alpha=0.8, color='gold')
                ax4.set_title('Davies-Bouldin Score (Lower = Better)')
                ax4.set_ylabel('Davies-Bouldin Score')
                ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def statistical_significance_test(self, results1, results2, metric='silhouette_score'):
        """
        Test statistical significance between two methods
        
        Args:
            results1, results2: Cross-validation results from two methods
            metric: metric to compare
        
        Returns:
            statistical test results
        """
        if metric not in results1 or metric not in results2:
            return None
        
        values1 = results1[f"{metric}_values"]
        values2 = results2[f"{metric}_values"]
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(values1, values2)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1) + 
                             (len(values2) - 1) * np.var(values2)) / (len(values1) + len(values2) - 2))
        cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0
        
        return {
            'metric': metric,
            'method1_mean': np.mean(values1),
            'method2_mean': np.mean(values2),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant': p_value < 0.05,
            'interpretation': self._interpret_significance(p_value, cohens_d)
        }
    
    def _interpret_significance(self, p_value, cohens_d):
        """Interpret statistical significance results"""
        if p_value >= 0.05:
            return "No significant difference"
        
        effect_size = "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
        direction = "better" if cohens_d > 0 else "worse"
        
        return f"Significant difference (p < 0.05), {effect_size} effect, method 1 is {direction}"
    
    def save_evaluation_report(self, results, save_dir="evaluation_reports"):
        """Save comprehensive evaluation report"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results as JSON
        json_path = save_dir / f"evaluation_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_serializable = self._make_json_serializable(results)
            json.dump(json_serializable, f, indent=2)
        
        print(f"📄 Evaluation report saved to {json_path}")
        
        return json_path
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

def create_train_val_split(dataset, val_ratio=0.2, random_state=42):
    """
    Convenience function to create train/validation split
    
    Args:
        dataset: Dataset object
        val_ratio: ratio of validation data
        random_state: random seed
    
    Returns:
        train_dataset, val_dataset
    """
    framework = ValidationFramework(val_ratio=val_ratio, test_ratio=0, random_state=random_state)
    splits = framework.create_splits(dataset, stratify_by_file=True)
    return splits['train'], splits['val']

def evaluate_clustering_performance(embeddings, labels, method_name="Unknown"):
    """
    Convenience function to evaluate clustering performance
    
    Args:
        embeddings: audio embeddings
        labels: cluster labels
        method_name: name of the method
    
    Returns:
        evaluation metrics
    """
    framework = ValidationFramework()
    return framework.compute_diarization_metrics(embeddings, labels, method_name=method_name)

if __name__ == "__main__":
    print("Testing Validation Framework...")
    
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 200
    n_features = 128
    
    # Create structured embeddings (4 clusters)
    cluster1 = np.random.randn(50, n_features) + [2, 2] + [0] * (n_features - 2)
    cluster2 = np.random.randn(50, n_features) + [-2, 2] + [0] * (n_features - 2)
    cluster3 = np.random.randn(50, n_features) + [2, -2] + [0] * (n_features - 2)
    cluster4 = np.random.randn(50, n_features) + [-2, -2] + [0] * (n_features - 2)
    
    embeddings = np.vstack([cluster1, cluster2, cluster3, cluster4])
    true_labels = np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 50)
    
    # Test different clustering methods
    from sklearn.cluster import KMeans, DBSCAN
    
    method_results = {
        'KMeans': KMeans(n_clusters=4, random_state=42).fit_predict(embeddings),
        'KMeans_5': KMeans(n_clusters=5, random_state=42).fit_predict(embeddings),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5).fit_predict(embeddings)
    }
    
    # Test the framework
    framework = ValidationFramework()
    
    # Compare methods
    comparison = framework.compare_methods(embeddings, method_results)
    
    if comparison:
        print(f"\nBest method: {comparison['best_method']['method']}")
        print(f"Silhouette score: {comparison['best_method']['silhouette_score']:.3f}")
    
    print("✅ Validation framework test complete!")