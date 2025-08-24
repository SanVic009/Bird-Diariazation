import os

# Configuration for quick testing (fast, few species)
QUICK_TEST_CONFIG = {
    'audio': {
        'sample_rate': 22050,
        'n_mfcc': 13,
        'n_fft': 2048,
        'hop_length': 512,
        'max_len': 5,  # Shorter audio for speed
        'noise_reduction': False  # Disabled for speed
    },
    'model': {
        'hidden_size': 128,  # Smaller model
        'num_layers': 1,
        'dropout': 0.3
    },
    'training': {
        'batch_size': 16,
        'learning_rate': 0.001,
        'num_epochs': 10,  # Fewer epochs
        'patience': 5,
        'test_size': 0.2,
        'val_size': 0.1,
        'num_workers': 2
    },
    'dataset': {
        'max_files': 1000,  # Only 1000 files for testing
        'min_samples_per_species': 5,
        'balanced_sampling': False
    }
}

# Configuration for development (moderate dataset)
DEVELOPMENT_CONFIG = {
    'audio': {
        'sample_rate': 22050,
        'n_mfcc': 13,
        'n_fft': 2048,
        'hop_length': 512,
        'max_len': 10,
        'noise_reduction': True
    },
    'model': {
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.3
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 25,
        'patience': 8,
        'test_size': 0.2,
        'val_size': 0.1,
        'num_workers': 4
    },
    'dataset': {
        'max_files': 8000,  # 8000 files - good balance
        'min_samples_per_species': 10,
        'balanced_sampling': False
    }
}

# Configuration for full training (complete dataset)
FULL_TRAINING_CONFIG = {
    'audio': {
        'sample_rate': 22050,
        'n_mfcc': 13,
        'n_fft': 2048,
        'hop_length': 512,
        'max_len': 10,
        'noise_reduction': True,
        'num_threads': -1  # Use all CPU cores for audio processing
    },
    'model': {
        'hidden_size': 512,  # Increased for better GPU utilization
        'num_layers': 3,     # Deeper model for GPU utilization
        'dropout': 0.3
    },
    'training': {
        'batch_size': 128,   # Increased for better GPU utilization
        'learning_rate': 0.001,
    'num_epochs': 50,
        'patience': 10,
        'test_size': 0.2,
        'val_size': 0.1,
        'num_workers': -1,   # Use all CPU cores for data loading
        'num_workers': os.cpu_count() // 2
    },
    'dataset': {
        'max_files': None,  # Use ALL 24,459 files!
        'min_samples_per_species': 5,
        'balanced_sampling': False
    }
}

# Configuration for balanced training (equal representation)
BALANCED_CONFIG = {
    'audio': {
        'sample_rate': 22050,
        'n_mfcc': 13,
        'n_fft': 2048,
        'hop_length': 512,
        'max_len': 10,
        'noise_reduction': True
    },
    'model': {
        'hidden_size': 256,
        'num_layers': 2,
        'dropout': 0.3
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 40,
        'patience': 10,
        'test_size': 0.2,
        'val_size': 0.1,
        'num_workers': os.cpu_count() // 2
    },
    'dataset': {
        'max_files': None,
        'min_samples_per_species': 20,  # Higher minimum
        'balanced_sampling': True,  # Balance species representation
        'max_samples_per_species': 100  # Cap at 100 per species
    }
}

# High-performance configuration (for GPU training)
GPU_CONFIG = {
    'audio': {
        'sample_rate': 22050,
        'n_mfcc': 13,
        'n_fft': 2048,
        'hop_length': 512,
        'max_len': 15,  # Longer audio for better features
        'noise_reduction': True
    },
    'model': {
        'hidden_size': 512,  # Larger model
        'num_layers': 3,
        'dropout': 0.4
    },
    'training': {
        'batch_size': 32,  # Larger batches for GPU
        'learning_rate': 0.001,
        'num_epochs': 100,
        'patience': 15,
        'test_size': 0.15,
        'val_size': 0.15,
        'num_workers': os.cpu_count()
    },
    'dataset': {
        'max_files': None,  # Use all files
        'min_samples_per_species': 5,
        'balanced_sampling': False
    }
}

def get_config(config_name):
    """Get configuration by name"""
    configs = {
        'quick': QUICK_TEST_CONFIG,
        'dev': DEVELOPMENT_CONFIG,
        'full': FULL_TRAINING_CONFIG,
        'balanced': BALANCED_CONFIG,
        'gpu': GPU_CONFIG
    }
    
    if config_name not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return configs[config_name]

def print_config_summary():
    """Print summary of all available configurations"""
    print("🔧 Available Training Configurations:")
    print("=" * 50)
    
    configs = {
        'quick': ('Quick Test', QUICK_TEST_CONFIG),
        'dev': ('Development', DEVELOPMENT_CONFIG),
        'full': ('Full Training', FULL_TRAINING_CONFIG),
        'balanced': ('Balanced Training', BALANCED_CONFIG),
        'gpu': ('GPU Optimized', GPU_CONFIG)
    }
    
    for name, (description, config) in configs.items():
        max_files = config['dataset']['max_files']
        epochs = config['training']['num_epochs']
        batch_size = config['training']['batch_size']
        
        print(f"\n{name.upper()}: {description}")
        print(f"  Files: {'All (~24K)' if max_files is None else f'{max_files:,}'}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Estimated Training Time: {estimate_training_time(config)}")

def estimate_training_time(config):
    """Estimate training time based on configuration"""
    max_files = config['dataset']['max_files'] or 24459
    epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    
    # Rough estimates based on CPU training
    batches_per_epoch = max_files / batch_size
    seconds_per_batch = 15  # Rough estimate
    
    total_seconds = batches_per_epoch * epochs * seconds_per_batch
    
    if total_seconds < 3600:
        return f"{total_seconds/60:.0f} minutes"
    else:
        return f"{total_seconds/3600:.1f} hours"

if __name__ == "__main__":
    print_config_summary()
