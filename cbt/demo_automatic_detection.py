#!/usr/bin/env python3
"""
demo_automatic_detection.py - Demonstrates Automatic Bird Detection

Shows how the enhanced system automatically detects the number of birds
without you specifying how many there are!
"""

import numpy as np
import torch
from advanced_clustering import perform_advanced_diarization
from improved_models import ImprovedDiarizationEncoder

def demo_automatic_detection():
    """Demonstrate automatic bird detection"""
    print("🎯 AUTOMATIC BIRD DETECTION DEMO")
    print("=" * 50)
    
    # Create synthetic audio embeddings representing different scenarios
    scenarios = {
        "Few Birds (2-3)": create_few_birds_scenario(),
        "Many Birds (6-8)": create_many_birds_scenario(), 
        "Variable Birds": create_variable_scenario(),
        "Real-world Mix": create_realistic_scenario()
    }
    
    for scenario_name, embeddings in scenarios.items():
        print(f"\n🐦 {scenario_name}:")
        print(f"   Input: {len(embeddings)} audio segments")
        print("   Specification: ❌ NO number specified (automatic detection)")
        
        # ✨ AUTOMATIC DETECTION - No manual specification!
        result = perform_advanced_diarization(
            embeddings,
            max_speakers=10  # Only upper limit for safety
        )
        
        if result:
            print(f"   🎯 AUTOMATICALLY DETECTED: {result['n_speakers']} birds!")
            print(f"   📊 Quality Score: {result['metrics']['silhouette_score']:.3f}")
            print(f"   🔧 Method Used: {result['method']}")
            print(f"   ✅ Confidence: {'High' if result['metrics']['silhouette_score'] > 0.3 else 'Medium' if result['metrics']['silhouette_score'] > 0.1 else 'Low'}")
        else:
            print("   ❌ Detection failed")

def create_few_birds_scenario():
    """Create embeddings representing 2-3 birds"""
    np.random.seed(42)
    
    # 2 distinct bird types
    bird1 = np.random.randn(15, 256) + [3, 2, 0, 1] + [0] * 252  # Bird type 1
    bird2 = np.random.randn(20, 256) + [-2, 3, 1, 0] + [0] * 252  # Bird type 2
    
    return np.vstack([bird1, bird2])

def create_many_birds_scenario():
    """Create embeddings representing 6-8 birds"""
    np.random.seed(123)
    
    birds = []
    centers = [
        [4, 4, 0, 0], [-4, 4, 0, 0], [4, -4, 0, 0], [-4, -4, 0, 0],
        [0, 4, 3, 0], [0, -4, -3, 0], [3, 0, 0, 4]
    ]
    
    for i, center in enumerate(centers):
        segments = np.random.randint(8, 15)  # Variable segments per bird
        bird = np.random.randn(segments, 256) + center + [0] * 252
        birds.append(bird)
    
    return np.vstack(birds)

def create_variable_scenario():
    """Create scenario with variable bird activity"""
    np.random.seed(456)
    
    # Some birds very active, others less so
    very_active_bird = np.random.randn(25, 256) + [5, 0, 0, 0] + [0] * 252
    active_bird = np.random.randn(15, 256) + [0, 5, 0, 0] + [0] * 252
    quiet_bird = np.random.randn(8, 256) + [0, 0, 5, 0] + [0] * 252
    occasional_bird = np.random.randn(5, 256) + [0, 0, 0, 5] + [0] * 252
    
    return np.vstack([very_active_bird, active_bird, quiet_bird, occasional_bird])

def create_realistic_scenario():
    """Create realistic mixed scenario"""
    np.random.seed(789)
    
    # Mix of clear and similar birds
    clear_bird1 = np.random.randn(18, 256) + [4, 4, 0, 0] + [0] * 252
    clear_bird2 = np.random.randn(16, 256) + [-4, -4, 0, 0] + [0] * 252
    similar_bird1 = np.random.randn(12, 256) + [2, 2, 1, 1] + [0] * 252
    similar_bird2 = np.random.randn(10, 256) + [2.5, 2.5, 1.2, 1.2] + [0] * 252  # Similar to above
    distant_bird = np.random.randn(14, 256) + [0, 0, 6, 0] + [0] * 252
    
    return np.vstack([clear_bird1, clear_bird2, similar_bird1, similar_bird2, distant_bird])

def demo_with_real_model():
    """Demo with actual model inference"""
    print(f"\n" + "=" * 60)
    print("🏗️ DEMO WITH ENHANCED MODEL")
    print("=" * 60)
    
    try:
        # Create enhanced model
        model = ImprovedDiarizationEncoder(embed_dim=256)
        model.eval()
        
        print("✅ Enhanced model created successfully!")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create synthetic audio spectrograms
        n_segments = 30
        audio_spectrograms = torch.randn(n_segments, 1, 128, 501)  # [batch, channels, mel_bins, time]
        
        print(f"\n🎵 Processing {n_segments} audio segments...")
        
        # Extract embeddings
        embeddings = []
        with torch.no_grad():
            for i in range(n_segments):
                emb = model(audio_spectrograms[i:i+1])
                embeddings.append(emb.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        print(f"   Embeddings shape: {embeddings.shape}")
        
        # ✨ AUTOMATIC DETECTION
        print("\n🔍 Performing AUTOMATIC bird detection...")
        result = perform_advanced_diarization(embeddings)
        
        if result:
            print(f"\n🎯 FINAL RESULTS:")
            print(f"   🐦 Automatically detected: {result['n_speakers']} different birds")
            print(f"   📊 Quality score: {result['metrics']['silhouette_score']:.3f}")
            print(f"   🔧 Best method: {result['method']}")
            print(f"   📈 Cluster balance: {result['metrics']['cluster_balance']:.3f}")
            
            # Show speaker distribution
            unique, counts = np.unique(result['labels'], return_counts=True)
            print(f"\n📋 Speaker Activity Distribution:")
            for speaker, count in zip(unique, counts):
                percentage = (count / len(result['labels'])) * 100
                print(f"   🐦 Bird {speaker}: {count} segments ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Make sure you've run the setup correctly!")

def show_automatic_detection_features():
    """Show all the automatic detection capabilities"""
    print(f"\n" + "=" * 60)
    print("🎯 AUTOMATIC DETECTION CAPABILITIES")
    print("=" * 60)
    
    features = [
        "✅ NO manual specification required",
        "✅ Tests 2-12 speakers automatically", 
        "✅ Uses 5 different clustering algorithms",
        "✅ Ensemble voting for robustness",
        "✅ Quality-based selection (silhouette score)",
        "✅ Temporal smoothing removes noise",
        "✅ Handles variable bird activity",
        "✅ Works with any audio length",
        "✅ Automatic parameter optimization",
        "✅ Statistical validation of results"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print(f"\n💡 KEY POINT: You only set max_speakers as a safety limit!")
    print(f"   The system finds the OPTIMAL number within that range.")

if __name__ == "__main__":
    # Run all demos
    demo_automatic_detection()
    demo_with_real_model() 
    show_automatic_detection_features()
    
    print(f"\n🎉 CONCLUSION:")
    print(f"   Your enhanced system AUTOMATICALLY detects bird counts!")
    print(f"   No manual specification needed - it's fully automatic! 🚀")