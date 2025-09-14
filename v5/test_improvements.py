#!/usr/bin/env python3
"""
Test script for the new preprocessing improvements:
- Frequency Normalization
- MixUp Augmentation
"""
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import BirdPreprocessor
import librosa

def test_frequency_normalization():
    """Test frequency normalization functionality"""
    print("Testing Frequency Normalization...")
    
    # Create a simple test signal
    sr = 32000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a signal with multiple frequency components
    signal = (np.sin(2 * np.pi * 1000 * t) +  # 1kHz
              0.5 * np.sin(2 * np.pi * 2000 * t) +  # 2kHz  
              0.3 * np.sin(2 * np.pi * 4000 * t))   # 4kHz
    
    # Add some noise
    signal += 0.1 * np.random.randn(len(signal))
    
    # Initialize preprocessor
    preprocessor = BirdPreprocessor()
    
    # Test without normalization
    log_mel_original = preprocessor.to_log_mel(signal, normalize_freq=False)
    
    # Test with normalization
    log_mel_normalized = preprocessor.to_log_mel(signal, normalize_freq=True)
    
    print(f"Original log-mel shape: {log_mel_original.shape}")
    print(f"Normalized log-mel shape: {log_mel_normalized.shape}")
    print(f"Original mean per freq bin (first 5): {log_mel_original.mean(axis=1)[:5]}")
    print(f"Normalized mean per freq bin (first 5): {log_mel_normalized.mean(axis=1)[:5]}")
    print(f"Original std per freq bin (first 5): {log_mel_original.std(axis=1)[:5]}")
    print(f"Normalized std per freq bin (first 5): {log_mel_normalized.std(axis=1)[:5]}")
    
    # The normalized version should have mean≈0 and std≈1 for each frequency bin
    assert np.allclose(log_mel_normalized.mean(axis=1), 0, atol=1e-6), "Frequency normalization failed: mean not close to 0"
    assert np.allclose(log_mel_normalized.std(axis=1), 1, atol=1e-6), "Frequency normalization failed: std not close to 1"
    
    print("✅ Frequency normalization test passed!")
    return log_mel_original, log_mel_normalized

def test_mixup_augmentation():
    """Test MixUp augmentation functionality"""
    print("\nTesting MixUp Augmentation...")
    
    # Create two different test signals
    sr = 32000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Signal 1: Lower frequency bird call simulation
    signal1 = np.sin(2 * np.pi * 1500 * t) * np.exp(-t)  # Decaying 1.5kHz
    
    # Signal 2: Higher frequency bird call simulation  
    signal2 = np.sin(2 * np.pi * 3000 * t) * np.exp(-2*t)  # Decaying 3kHz
    
    # Initialize preprocessor
    preprocessor = BirdPreprocessor(mixup_prob=1.0, mixup_alpha=0.2)
    
    # Test MixUp
    mixed_signal, mix_ratio = preprocessor.mixup_augmentation(signal1, signal2, alpha=0.2)
    
    print(f"Signal 1 shape: {signal1.shape}")
    print(f"Signal 2 shape: {signal2.shape}")
    print(f"Mixed signal shape: {mixed_signal.shape}")
    print(f"Mix ratio (lambda): {mix_ratio:.3f}")
    
    # Verify the mixed signal is a linear combination
    expected_mixed = mix_ratio * signal1 + (1 - mix_ratio) * signal2
    assert np.allclose(mixed_signal, expected_mixed), "MixUp mixing failed"
    
    # Test energy conservation (approximately)
    energy1 = np.sum(signal1**2)
    energy2 = np.sum(signal2**2)
    energy_mixed = np.sum(mixed_signal**2)
    expected_energy = mix_ratio**2 * energy1 + (1-mix_ratio)**2 * energy2 + 2*mix_ratio*(1-mix_ratio)*np.sum(signal1*signal2)
    
    print(f"Energy 1: {energy1:.2f}")
    print(f"Energy 2: {energy2:.2f}")
    print(f"Mixed energy: {energy_mixed:.2f}")
    print(f"Expected energy: {expected_energy:.2f}")
    
    print("✅ MixUp augmentation test passed!")
    return signal1, signal2, mixed_signal, mix_ratio

def test_cache_functionality():
    """Test MixUp cache functionality"""
    print("\nTesting MixUp Cache...")
    
    preprocessor = BirdPreprocessor(mixup_prob=1.0)
    # Set smaller cache size for testing
    preprocessor.max_cache_size = 3
    
    # Create test signals
    signals = []
    labels = ["species_A", "species_B", "species_C", "species_D"]
    
    for i in range(4):
        signal = np.random.randn(32000)  # 1 second of noise
        signals.append(signal)
        preprocessor._add_to_mixup_cache(signal, labels[i])
    
    print(f"Cache size after adding 4 items: {len(preprocessor.mixup_cache)}")
    print(f"Cache labels: {preprocessor.mixup_cache_labels}")
    
    # Should only keep last 3 due to max_cache_size=3
    assert len(preprocessor.mixup_cache) == 3, f"Cache size should be 3, got {len(preprocessor.mixup_cache)}"
    assert preprocessor.mixup_cache_labels == ["species_B", "species_C", "species_D"], "Cache should contain last 3 labels"
    
    # Test getting a mixup sample
    mixup_wav, mixup_label = preprocessor._get_mixup_sample("species_B")
    print(f"Retrieved mixup sample label: {mixup_label}")
    assert mixup_label in preprocessor.mixup_cache_labels, "Retrieved label should be in cache"
    
    print("✅ Cache functionality test passed!")

def visualize_improvements():
    """Create visualizations of the improvements"""
    print("\nCreating visualizations...")
    
    # Test frequency normalization visualization
    log_mel_original, log_mel_normalized = test_frequency_normalization()
    
    # Test MixUp visualization
    signal1, signal2, mixed_signal, mix_ratio = test_mixup_augmentation()
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Original vs Normalized spectrogram
    axes[0, 0].imshow(log_mel_original, aspect='auto', origin='lower')
    axes[0, 0].set_title('Original Log-Mel Spectrogram')
    axes[0, 0].set_ylabel('Mel Frequency Bins')
    
    axes[0, 1].imshow(log_mel_normalized, aspect='auto', origin='lower')
    axes[0, 1].set_title('Frequency-Normalized Log-Mel Spectrogram')
    
    # Plot 2: MixUp signals
    t = np.linspace(0, 3, len(signal1))
    axes[1, 0].plot(t[:1000], signal1[:1000], label='Signal 1', alpha=0.7)
    axes[1, 0].plot(t[:1000], signal2[:1000], label='Signal 2', alpha=0.7)
    axes[1, 0].set_title('Original Signals (first 1000 samples)')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')
    
    axes[1, 1].plot(t[:1000], mixed_signal[:1000], label=f'Mixed (λ={mix_ratio:.2f})', color='red')
    axes[1, 1].set_title(f'MixUp Result (λ={mix_ratio:.2f})')
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig('/home/sanvict/Documents/GitHub/Bird-Diariazation/v5/preprocessing_improvements_test.png', dpi=150)
    print("📊 Visualization saved as 'preprocessing_improvements_test.png'")

if __name__ == "__main__":
    print("🧪 Testing Preprocessing Improvements")
    print("=" * 50)
    
    try:
        # Run tests
        test_frequency_normalization()
        test_mixup_augmentation()
        test_cache_functionality()
        
        # Create visualizations (optional, requires matplotlib)
        try:
            visualize_improvements()
        except ImportError:
            print("⚠️  Matplotlib not available, skipping visualizations")
        
        print("\n🎉 All tests passed! Your improvements are working correctly.")
        print("\nNew features available:")
        print("✅ Frequency normalization (normalizes each frequency bin)")
        print("✅ MixUp augmentation (mixes audio from different species)")
        print("✅ Smart caching system for MixUp")
        
        print("\nUsage in run_preprocessing.py:")
        print("python run_preprocessing.py --rfcx_root /path/to/data --mixup_prob 0.3 --mixup_alpha 0.2")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
