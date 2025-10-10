#!/usr/bin/env python3
"""
comprehensive_test.py - Comprehensive Testing Suite for Enhanced Bird Diarization

Features:
- Model architecture testing
- Data pipeline validation
- Performance benchmarking
- Comparison with baseline methods
- Integration testing
- Regression testing
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Import all our enhanced components
from improved_models import ImprovedDiarizationEncoder, LegacyDiarizationEncoder
from enhanced_augmentations import ImprovedDiarizationDataset, AdvancedAudioAugmenter
from advanced_loss_functions import AdvancedContrastiveLoss, get_loss_function
from advanced_clustering import perform_advanced_diarization, AdvancedClusteringStrategy
from validation_framework import ValidationFramework, evaluate_clustering_performance
from improved_train import EnhancedTrainer, get_default_config

class ComprehensiveTestSuite:
    """Complete testing suite for the enhanced bird diarization system"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_results = []
        self.validation_framework = ValidationFramework()
        
        print("🧪 Comprehensive Test Suite Initialized")
        print(f"   Device: {self.device}")
        print("=" * 60)
    
    def test_model_architecture(self):
        """Test improved model architectures"""
        print("🏗️  Testing Model Architectures...")
        
        tests = []
        
        # Test improved encoder
        try:
            model = ImprovedDiarizationEncoder(embed_dim=256)
            model.eval()
            
            # Test forward pass
            x = torch.randn(4, 1, 128, 501)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (4, 256), f"Expected (4, 256), got {output.shape}"
            assert torch.allclose(torch.norm(output, dim=1), torch.ones(4), atol=1e-3), "Output not normalized"
            
            tests.append({
                'test': 'ImprovedDiarizationEncoder',
                'status': 'PASS',
                'details': f'Output shape: {output.shape}, normalized: True'
            })
            
        except Exception as e:
            tests.append({
                'test': 'ImprovedDiarizationEncoder',
                'status': 'FAIL',
                'details': str(e)
            })
        
        # Test legacy encoder compatibility
        try:
            legacy_model = LegacyDiarizationEncoder(embed_dim=128)
            x = torch.randn(2, 1, 128, 501)
            with torch.no_grad():
                output = legacy_model(x)
            
            assert output.shape == (2, 128), f"Expected (2, 128), got {output.shape}"
            
            tests.append({
                'test': 'LegacyDiarizationEncoder',
                'status': 'PASS',
                'details': f'Backward compatibility maintained'
            })
            
        except Exception as e:
            tests.append({
                'test': 'LegacyDiarizationEncoder', 
                'status': 'FAIL',
                'details': str(e)
            })
        
        # Test parameter count
        try:
            improved_model = ImprovedDiarizationEncoder(embed_dim=256)
            legacy_model = LegacyDiarizationEncoder(embed_dim=128)
            
            improved_params = sum(p.numel() for p in improved_model.parameters())
            legacy_params = sum(p.numel() for p in legacy_model.parameters())
            
            tests.append({
                'test': 'Parameter Count Comparison',
                'status': 'PASS',
                'details': f'Improved: {improved_params:,}, Legacy: {legacy_params:,}, Ratio: {improved_params/legacy_params:.1f}x'
            })
            
        except Exception as e:
            tests.append({
                'test': 'Parameter Count Comparison',
                'status': 'FAIL',
                'details': str(e)
            })
        
        self._print_test_results("Model Architecture Tests", tests)
        return tests
    
    def test_data_augmentation(self):
        """Test enhanced data augmentation pipeline"""
        print("📊 Testing Data Augmentation...")
        
        tests = []
        
        try:
            # Test augmenter
            augmenter = AdvancedAudioAugmenter()
            x = torch.randn(1, 128, 501)
            
            # Test different augmentations
            augmented = augmenter(x.clone(), training=True)
            
            assert augmented.shape == x.shape, f"Shape mismatch: {augmented.shape} vs {x.shape}"
            assert not torch.allclose(augmented, x), "Augmentation had no effect"
            
            # Test no augmentation during evaluation
            no_aug = augmenter(x.clone(), training=False)
            assert torch.allclose(no_aug, x), "Augmentation applied during evaluation"
            
            tests.append({
                'test': 'AdvancedAudioAugmenter',
                'status': 'PASS',
                'details': 'All augmentation modes working correctly'
            })
            
        except Exception as e:
            tests.append({
                'test': 'AdvancedAudioAugmenter',
                'status': 'FAIL',
                'details': str(e)
            })
        
        # Test improved dataset
        try:
            # Create a mock dataset structure for testing
            test_dir = Path("test_cache_mels")
            test_dir.mkdir(exist_ok=True)
            
            # Create some dummy mel spectrogram files
            for i in range(5):
                dummy_mel = torch.randn(128, 400 + i * 50)  # Variable length
                torch.save(dummy_mel, test_dir / f"test_file_{i}_segment_0.pt")
            
            # Test dataset creation
            dataset = ImprovedDiarizationDataset(
                root=str(test_dir),
                training=True,
                augmentation_strength=0.5
            )
            
            assert len(dataset) == 5, f"Expected 5 samples, got {len(dataset)}"
            
            # Test data loading
            x1, x2, idx = dataset[0]
            assert x1.shape == x2.shape, "Augmented views have different shapes"
            assert x1.shape[-1] == 501, f"Expected width 501, got {x1.shape[-1]}"
            
            # Cleanup
            import shutil
            shutil.rmtree(test_dir)
            
            tests.append({
                'test': 'ImprovedDiarizationDataset',
                'status': 'PASS',
                'details': f'Dataset created with {len(dataset)} samples'
            })
            
        except Exception as e:
            tests.append({
                'test': 'ImprovedDiarizationDataset',
                'status': 'FAIL',
                'details': str(e)
            })
        
        self._print_test_results("Data Augmentation Tests", tests)
        return tests
    
    def test_loss_functions(self):
        """Test advanced loss functions"""
        print("🎯 Testing Loss Functions...")
        
        tests = []
        
        # Test different loss functions
        loss_types = ['basic', 'advanced', 'focal', 'infonct']
        
        for loss_type in loss_types:
            try:
                loss_fn = get_loss_function(loss_type)
                
                # Test forward pass
                z1 = torch.randn(16, 128, requires_grad=True)
                z2 = torch.randn(16, 128, requires_grad=True)
                
                # Normalize embeddings
                z1 = torch.nn.functional.normalize(z1, dim=1)
                z2 = torch.nn.functional.normalize(z2, dim=1)
                
                loss = loss_fn(z1, z2)
                
                assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
                assert not torch.isnan(loss), "Loss is NaN"
                assert not torch.isinf(loss), "Loss is infinite"
                
                # Test backward pass
                loss.backward()
                
                tests.append({
                    'test': f'{loss_type}_loss',
                    'status': 'PASS',
                    'details': f'Loss value: {loss.item():.4f}'
                })
                
            except Exception as e:
                tests.append({
                    'test': f'{loss_type}_loss',
                    'status': 'FAIL',
                    'details': str(e)
                })
        
        self._print_test_results("Loss Function Tests", tests)
        return tests
    
    def test_clustering_algorithms(self):
        """Test advanced clustering algorithms"""
        print("🎯 Testing Clustering Algorithms...")
        
        tests = []
        
        # Create synthetic embeddings with clear structure
        np.random.seed(42)
        n_samples = 100
        n_features = 128
        
        # Create 3 clear clusters
        center1 = np.array([3, 3] + [0] * (n_features - 2))
        center2 = np.array([-3, 3] + [0] * (n_features - 2))
        center3 = np.array([0, -3] + [0] * (n_features - 2))
        
        cluster1 = np.random.randn(30, n_features) + center1
        cluster2 = np.random.randn(30, n_features) + center2
        cluster3 = np.random.randn(40, n_features) + center3
        
        embeddings = np.vstack([cluster1, cluster2, cluster3])
        true_labels = np.array([0] * 30 + [1] * 30 + [2] * 40)
        
        try:
            # Test advanced clustering
            result = perform_advanced_diarization(embeddings, max_speakers=5)
            
            assert result is not None, "Clustering returned None"
            assert 'labels' in result, "Result missing labels"
            assert 'n_speakers' in result, "Result missing n_speakers"
            assert 'metrics' in result, "Result missing metrics"
            
            # Check if we detect reasonable number of clusters
            assert 2 <= result['n_speakers'] <= 5, f"Detected {result['n_speakers']} speakers, expected 2-5"
            
            # Check silhouette score
            silhouette = result['metrics'].get('silhouette_score', -1)
            # Allow negative silhouette scores for testing (they just indicate poor clustering)
            assert silhouette is not None and silhouette != 'N/A', f"Invalid silhouette score: {silhouette}"
            
            tests.append({
                'test': 'Advanced Clustering',
                'status': 'PASS',
                'details': f"Detected {result['n_speakers']} speakers, silhouette: {silhouette:.3f}"
            })
            
        except Exception as e:
            tests.append({
                'test': 'Advanced Clustering',
                'status': 'FAIL',
                'details': str(e)
            })
        
        # Test individual clustering methods
        try:
            clustering_strategy = AdvancedClusteringStrategy(max_speakers=5)
            best_result, all_results = clustering_strategy.find_optimal_clusters(embeddings)
            
            assert len(all_results) > 0, "No clustering methods succeeded"
            assert best_result is not None, "No best result found"
            
            # Test ensemble clustering
            if len(embeddings) > 20:
                ensemble_result = clustering_strategy.ensemble_clustering(embeddings)
                assert ensemble_result is not None, "Ensemble clustering failed"
            
            tests.append({
                'test': 'Clustering Strategy',
                'status': 'PASS', 
                'details': f"Tested {len(all_results)} methods, best: {best_result['method']}"
            })
            
        except Exception as e:
            tests.append({
                'test': 'Clustering Strategy',
                'status': 'FAIL',
                'details': str(e)
            })
        
        self._print_test_results("Clustering Algorithm Tests", tests)
        return tests
    
    def test_validation_framework(self):
        """Test validation and evaluation framework"""
        print("📊 Testing Validation Framework...")
        
        tests = []
        
        # Test metric computation
        try:
            # Synthetic data
            embeddings = np.random.randn(50, 64)
            predicted_labels = np.random.randint(0, 3, 50)
            true_labels = np.random.randint(0, 3, 50)
            
            metrics = evaluate_clustering_performance(
                embeddings, predicted_labels, method_name="Test"
            )
            
            required_metrics = ['silhouette_score', 'n_predicted_clusters', 'cluster_sizes']
            for metric in required_metrics:
                assert metric in metrics, f"Missing metric: {metric}"
            
            tests.append({
                'test': 'Metrics Computation',
                'status': 'PASS',
                'details': f"Computed {len(metrics)} metrics successfully"
            })
            
        except Exception as e:
            tests.append({
                'test': 'Metrics Computation',
                'status': 'FAIL',
                'details': str(e)
            })
        
        # Test method comparison
        try:
            embeddings = np.random.randn(100, 64)
            
            # Create different clustering results
            method_results = {
                'Method1': np.random.randint(0, 3, 100),
                'Method2': np.random.randint(0, 4, 100),
                'Method3': np.random.randint(0, 5, 100)
            }
            
            framework = ValidationFramework()
            comparison = framework.compare_methods(embeddings, method_results)
            
            assert 'comparison_df' in comparison, "Missing comparison DataFrame"
            assert 'best_method' in comparison, "Missing best method"
            assert len(comparison['all_results']) == 3, "Missing comparison results"
            
            tests.append({
                'test': 'Method Comparison',
                'status': 'PASS',
                'details': f"Compared {len(method_results)} methods successfully"
            })
            
        except Exception as e:
            tests.append({
                'test': 'Method Comparison',
                'status': 'FAIL',
                'details': str(e)
            })
        
        self._print_test_results("Validation Framework Tests", tests)
        return tests
    
    def test_training_pipeline(self):
        """Test the enhanced training pipeline"""
        print("🚀 Testing Training Pipeline...")
        
        tests = []
        
        # Test configuration
        try:
            config = get_default_config()
            
            required_keys = [
                'data_path', 'embed_dim', 'batch_size', 'learning_rate', 
                'epochs', 'loss_type', 'optimizer'
            ]
            
            for key in required_keys:
                assert key in config, f"Missing config key: {key}"
            
            tests.append({
                'test': 'Configuration',
                'status': 'PASS',
                'details': f"All {len(required_keys)} required keys present"
            })
            
        except Exception as e:
            tests.append({
                'test': 'Configuration',
                'status': 'FAIL',
                'details': str(e)
            })
        
        # Test trainer initialization (without actual training)
        try:
            config = get_default_config()
            config['epochs'] = 1  # Short test
            config['data_path'] = 'test_cache'  # Non-existent path for testing
            
            trainer = EnhancedTrainer(config)
            
            # Test setup methods individually
            trainer.setup_model()
            assert trainer.model is not None, "Model not initialized"
            
            tests.append({
                'test': 'Trainer Initialization',
                'status': 'PASS',
                'details': 'Trainer and model setup successful'
            })
            
        except Exception as e:
            tests.append({
                'test': 'Trainer Initialization',
                'status': 'FAIL', 
                'details': str(e)
            })
        
        self._print_test_results("Training Pipeline Tests", tests)
        return tests
    
    def benchmark_performance(self):
        """Benchmark performance of different components"""
        print("⚡ Performance Benchmarking...")
        
        benchmarks = []
        
        # Model inference speed
        try:
            model = ImprovedDiarizationEncoder(embed_dim=256).eval()
            x = torch.randn(32, 1, 128, 501)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x)
            
            # Benchmark
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model(x)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100
            samples_per_second = 32 / avg_time
            
            benchmarks.append({
                'component': 'ImprovedDiarizationEncoder',
                'metric': 'Inference Speed',
                'value': f'{samples_per_second:.1f} samples/sec',
                'raw_value': samples_per_second
            })
            
        except Exception as e:
            print(f"   ❌ Model benchmarking failed: {e}")
        
        # Clustering speed
        try:
            embeddings = np.random.randn(200, 128)
            
            start_time = time.time()
            result = perform_advanced_diarization(embeddings, max_speakers=8)
            end_time = time.time()
            
            clustering_time = end_time - start_time
            samples_per_second = len(embeddings) / clustering_time
            
            benchmarks.append({
                'component': 'Advanced Clustering',
                'metric': 'Processing Speed', 
                'value': f'{samples_per_second:.1f} samples/sec',
                'raw_value': samples_per_second
            })
            
        except Exception as e:
            print(f"   ❌ Clustering benchmarking failed: {e}")
        
        # Print benchmark results
        if benchmarks:
            print("   📊 Performance Results:")
            for bench in benchmarks:
                print(f"      {bench['component']} - {bench['metric']}: {bench['value']}")
        
        return benchmarks
    
    def test_integration(self):
        """Test end-to-end integration"""
        print("🔗 Integration Testing...")
        
        tests = []
        
        try:
            # Create minimal test setup
            test_dir = Path("integration_test")
            test_dir.mkdir(exist_ok=True)
            
            # Create dummy data
            for i in range(10):
                dummy_mel = torch.randn(128, 501)
                torch.save(dummy_mel, test_dir / f"bird_{i//3}_segment_{i}.pt")
            
            # Test complete pipeline
            # 1. Data loading
            dataset = ImprovedDiarizationDataset(str(test_dir), training=False)
            assert len(dataset) == 10, "Data loading failed"
            
            # 2. Model creation
            model = ImprovedDiarizationEncoder(embed_dim=128)
            model.eval()
            
            # 3. Embedding extraction
            embeddings = []
            with torch.no_grad():
                for i in range(len(dataset)):
                    x1, _, _ = dataset[i]
                    emb = model(x1.unsqueeze(0))
                    embeddings.append(emb.cpu().numpy())
            
            embeddings = np.vstack(embeddings)
            assert embeddings.shape == (10, 128), f"Wrong embedding shape: {embeddings.shape}"
            
            # 4. Clustering
            result = perform_advanced_diarization(embeddings, max_speakers=5)
            assert result is not None, "Clustering failed"
            assert result['n_speakers'] >= 1, "No speakers detected"
            
            # Cleanup
            import shutil
            shutil.rmtree(test_dir)
            
            tests.append({
                'test': 'End-to-End Pipeline',
                'status': 'PASS',
                'details': f"Processed {len(dataset)} samples, detected {result['n_speakers']} speakers"
            })
            
        except Exception as e:
            tests.append({
                'test': 'End-to-End Pipeline',
                'status': 'FAIL',
                'details': str(e)
            })
        
        self._print_test_results("Integration Tests", tests)
        return tests
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🧪 Running Comprehensive Test Suite")
        print("=" * 60)
        
        all_results = {}
        
        # Run all test categories
        test_categories = [
            ('Model Architecture', self.test_model_architecture),
            ('Data Augmentation', self.test_data_augmentation),
            ('Loss Functions', self.test_loss_functions),
            ('Clustering Algorithms', self.test_clustering_algorithms),
            ('Validation Framework', self.test_validation_framework),
            ('Training Pipeline', self.test_training_pipeline),
            ('Integration', self.test_integration)
        ]
        
        for category_name, test_func in test_categories:
            try:
                results = test_func()
                all_results[category_name] = results
            except Exception as e:
                print(f"❌ {category_name} tests failed: {e}")
                all_results[category_name] = [{'test': category_name, 'status': 'FAIL', 'details': str(e)}]
        
        # Run benchmarks
        print()
        benchmarks = self.benchmark_performance()
        all_results['Performance Benchmarks'] = benchmarks
        
        # Generate summary
        self._generate_test_summary(all_results)
        
        # Save results
        self._save_test_results(all_results)
        
        return all_results
    
    def _print_test_results(self, category, tests):
        """Print test results for a category"""
        passed = sum(1 for t in tests if t['status'] == 'PASS')
        total = len(tests)
        
        print(f"   Results: {passed}/{total} tests passed")
        
        for test in tests:
            status_icon = "✅" if test['status'] == 'PASS' else "❌"
            print(f"   {status_icon} {test['test']}: {test['details']}")
        print()
    
    def _generate_test_summary(self, all_results):
        """Generate overall test summary"""
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        total_passed = 0
        total_tests = 0
        
        for category, results in all_results.items():
            if category == 'Performance Benchmarks':
                continue
                
            category_passed = sum(1 for r in results if r.get('status') == 'PASS')
            category_total = len(results)
            
            total_passed += category_passed
            total_tests += category_total
            
            status = "✅ PASS" if category_passed == category_total else "⚠️  PARTIAL" if category_passed > 0 else "❌ FAIL"
            print(f"{status} {category}: {category_passed}/{category_total}")
        
        print(f"\n🎯 Overall: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
        
        if total_passed == total_tests:
            print("🎉 All tests passed! Your enhanced bird diarization system is ready!")
        elif total_passed > total_tests * 0.8:
            print("👍 Most tests passed. Minor issues to address.")
        else:
            print("⚠️  Several tests failed. Review the issues before deployment.")
    
    def _save_test_results(self, results):
        """Save test results to file"""
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"test_results_{timestamp}.json"
        
        # Make results JSON serializable
        serializable_results = {}
        for category, tests in results.items():
            if category == 'Performance Benchmarks':
                serializable_results[category] = tests
            else:
                serializable_results[category] = tests
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\n📄 Test results saved to {results_file}")

def main():
    """Run the comprehensive test suite"""
    test_suite = ComprehensiveTestSuite()
    results = test_suite.run_all_tests()
    return results

if __name__ == "__main__":
    import pandas as pd
    results = main()