#!/usr/bin/env python3
"""
improved_train.py - Enhanced Bird Diarization Training with All Improvements

Features:
- Improved model architectures (ResNet + Attention)
- Advanced data augmentations
- Enhanced contrastive loss with hard negative mining
- Proper validation splits
- Advanced clustering evaluation
- Comprehensive logging and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from pathlib import Path
import wandb  # For experiment tracking (optional)
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our enhanced components
from improved_models import ImprovedDiarizationEncoder
from enhanced_augmentations import ImprovedDiarizationDataset
from advanced_loss_functions import AdvancedContrastiveLoss, get_loss_function
from advanced_clustering import perform_advanced_diarization
from validation_framework import ValidationFramework, create_train_val_split

class ImprovedEarlyStopping:
    """Enhanced early stopping with better monitoring"""
    
    def __init__(self, patience=15, min_delta=1e-4, path="models/best_model.pt", 
                 monitor='val_loss', mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
    def __call__(self, current_score, model, epoch):
        if self.mode == 'min':
            is_better = current_score < (self.best_score - self.min_delta)
        else:
            is_better = current_score > (self.best_score + self.min_delta)
        
        if is_better:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'score': current_score,
                'monitor': self.monitor
            }, self.path)
            print(f"✅ New best {self.monitor}: {current_score:.6f} at epoch {epoch}")
            
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🛑 Early stopping at epoch {epoch}")
                print(f"   Best {self.monitor}: {self.best_score:.6f} at epoch {self.best_epoch}")

class EnhancedTrainer:
    """Enhanced trainer with all improvements"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.validation_framework = ValidationFramework()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.early_stopping = None
        
        # Training state
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        print(f"🚀 Enhanced Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Config: {config}")
        
    def setup_model(self):
        """Setup model architecture"""
        print("🏗️  Setting up model architecture...")
        
        self.model = ImprovedDiarizationEncoder(
            embed_dim=self.config['embed_dim'],
            num_heads=self.config.get('num_heads', 8),
            dropout=self.config.get('dropout', 0.1)
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   Model: ImprovedDiarizationEncoder")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Embedding dimension: {self.config['embed_dim']}")
        
    def setup_data(self):
        """Setup data loaders with validation split"""
        print("📊 Setting up data loaders...")
        
        # Create dataset
        full_dataset = ImprovedDiarizationDataset(
            root=self.config['data_path'],
            segment_length=self.config.get('segment_length', 2.0),
            training=True,
            augmentation_strength=self.config.get('augmentation_strength', 1.0)
        )
        
        if len(full_dataset) == 0:
            raise ValueError(f"No data found in {self.config['data_path']}")
        
        # Create train/validation split
        train_dataset, val_dataset = create_train_val_split(
            full_dataset, 
            val_ratio=self.config.get('val_ratio', 0.2),
            random_state=self.config.get('random_state', 42)
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        ) if val_dataset else None
        
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset) if val_dataset else 0}")
        print(f"   Batch size: {self.config['batch_size']}")
        
    def setup_training_components(self):
        """Setup optimizer, scheduler, loss function, and early stopping"""
        print("⚙️  Setting up training components...")
        
        # Loss function
        loss_type = self.config.get('loss_type', 'advanced')
        self.criterion = get_loss_function(
            stage=loss_type,
            temperature=self.config.get('temperature', 0.1),
            use_hard_negatives=self.config.get('use_hard_negatives', True),
            hard_negative_weight=self.config.get('hard_negative_weight', 2.0)
        )
        
        # Optimizer
        optimizer_type = self.config.get('optimizer', 'adamw')
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0.01),
                betas=self.config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Scheduler
        scheduler_type = self.config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs']
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 20),
                gamma=self.config.get('gamma', 0.5)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.get('lr_patience', 5),
                factor=self.config.get('lr_factor', 0.5),
                verbose=True
            )
        
        # Early stopping
        self.early_stopping = ImprovedEarlyStopping(
            patience=self.config.get('patience', 15),
            min_delta=self.config.get('min_delta', 1e-4),
            path=self.config.get('model_save_path', 'models/best_enhanced_model.pt'),
            monitor='val_silhouette' if self.val_loader else 'train_loss',
            mode='max' if self.val_loader else 'min'
        )
        
        print(f"   Optimizer: {optimizer_type}")
        print(f"   Scheduler: {scheduler_type}")
        print(f"   Loss function: {loss_type}")
        print(f"   Learning rate: {self.config['learning_rate']}")
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        
        for batch_idx, (x1, x2, indices) in enumerate(progress_bar):
            x1, x2 = x1.to(self.device), x2.to(self.device)
            
            # Forward pass
            z1 = self.model(x1)
            z2 = self.model(x2)
            
            # Compute loss
            loss = self.criterion(z1, z2)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"⚠️  NaN loss detected at batch {batch_idx}")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg': f"{epoch_loss/num_batches:.4f}"
            })
        
        return epoch_loss / num_batches if num_batches > 0 else float('inf')
    
    def validate_epoch(self):
        """Validate for one epoch"""
        if not self.val_loader:
            return None, {}
        
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        all_embeddings = []
        
        with torch.no_grad():
            for x1, x2, indices in tqdm(self.val_loader, desc="Validation"):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                
                # Forward pass
                z1 = self.model(x1)
                z2 = self.model(x2)
                
                # Compute loss
                loss = self.criterion(z1, z2)
                
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    num_batches += 1
                
                # Collect embeddings for clustering evaluation
                all_embeddings.append(z1.cpu().numpy())
        
        avg_val_loss = val_loss / num_batches if num_batches > 0 else float('inf')
        
        # Perform clustering evaluation (simplified for faster training)
        val_metrics = {}
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
            
            # Quick clustering evaluation using only K-means
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            best_silhouette = -1
            best_n_clusters = 2
            
            # Test only a few cluster numbers for speed
            for n_clusters in range(2, min(6, len(embeddings))):
                if n_clusters < len(embeddings):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
                    labels = kmeans.fit_predict(embeddings)
                    if len(np.unique(labels)) > 1:
                        silhouette = silhouette_score(embeddings, labels)
                        if silhouette > best_silhouette:
                            best_silhouette = silhouette
                            best_n_clusters = n_clusters
            
            val_metrics = {
                'silhouette_score': best_silhouette,
                'n_speakers': best_n_clusters,
                'clustering_method': 'kmeans_fast'
            }
        
        return avg_val_loss, val_metrics
    
    def train(self):
        """Main training loop"""
        print(f"🎯 Starting training for {self.config['epochs']} epochs...")
        
        # Setup training components
        self.setup_model()
        self.setup_data()
        self.setup_training_components()
        
        # Training loop
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            
            print(f"\n📅 Epoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate (only every val_frequency epochs for speed)
            val_frequency = self.config.get('val_frequency', 1)
            if epoch % val_frequency == 0 or epoch == self.config['epochs'] - 1:
                val_loss, val_metrics = self.validate_epoch()
                if val_loss is not None:
                    self.val_losses.append(val_loss)
                    self.val_metrics.append(val_metrics)
            else:
                val_loss, val_metrics = None, {}
            
            # Update scheduler
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if val_loss is not None:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step(train_loss)
            else:
                self.scheduler.step()
            
            # Print epoch summary
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"   Train Loss: {train_loss:.6f}")
            
            if val_loss is not None:
                print(f"   Val Loss: {val_loss:.6f}")
                if 'silhouette_score' in val_metrics:
                    print(f"   Val Silhouette: {val_metrics['silhouette_score']:.4f}")
                if 'n_speakers' in val_metrics:
                    print(f"   Detected Speakers: {val_metrics['n_speakers']}")
            
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Early stopping
            monitor_value = (val_metrics.get('silhouette_score', -1) if val_loss is not None 
                           else -train_loss)
            self.early_stopping(monitor_value, self.model, epoch)
            
            if self.early_stopping.early_stop:
                break
            
            # Save periodic checkpoints
            if (epoch + 1) % self.config.get('checkpoint_freq', 20) == 0:
                self.save_checkpoint(epoch)
        
        print(f"\n✅ Training completed!")
        
        # Load best model
        self.load_best_model()
        
        # Final evaluation
        self.final_evaluation()
        
        # Save training plots
        self.save_training_plots()
        
        return self.model
    
    def save_checkpoint(self, epoch):
        """Save training checkpoint"""
        checkpoint_path = f"models/checkpoint_epoch_{epoch+1}.pt"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'config': self.config
        }, checkpoint_path)
        
        print(f"   💾 Checkpoint saved: {checkpoint_path}")
    
    def load_best_model(self):
        """Load the best model from early stopping"""
        if os.path.exists(self.early_stopping.path):
            checkpoint = torch.load(self.early_stopping.path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📥 Loaded best model from epoch {checkpoint['epoch']}")
        
    def final_evaluation(self):
        """Perform final evaluation on validation set"""
        if not self.val_loader:
            return
        
        print("🔬 Performing final evaluation...")
        
        self.model.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for x1, x2, _ in self.val_loader:
                x1 = x1.to(self.device)
                z1 = self.model(x1)
                all_embeddings.append(z1.cpu().numpy())
        
        embeddings = np.vstack(all_embeddings)
        
        # Comprehensive clustering evaluation
        final_result = perform_advanced_diarization(embeddings, max_speakers=8)
        
        if final_result:
            print(f"   🎯 Final Results:")
            print(f"      Method: {final_result['method']}")
            print(f"      Speakers: {final_result['n_speakers']}")
            
            metrics = final_result['metrics']
            print(f"      Silhouette: {metrics.get('silhouette_score', 'N/A'):.4f}")
            print(f"      Calinski-Harabasz: {metrics.get('calinski_harabasz_score', 'N/A'):.2f}")
            print(f"      Davies-Bouldin: {metrics.get('davies_bouldin_score', 'N/A'):.4f}")
            
            # Save final results
            results_path = "results/final_evaluation.json"
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            
            with open(results_path, 'w') as f:
                # Make results JSON serializable
                serializable_results = {
                    'final_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                    for k, v in metrics.items()},
                    'n_speakers': int(final_result['n_speakers']),
                    'method': final_result['method'],
                    'config': self.config
                }
                json.dump(serializable_results, f, indent=2)
            
            print(f"   📄 Final results saved to {results_path}")
    
    def save_training_plots(self):
        """Save training progress plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🎯 Enhanced Training Progress', fontsize=16, fontweight='bold')
        
        # Training loss
        axes[0, 0].plot(self.train_losses, label='Training Loss', color='blue')
        if self.val_losses:
            axes[0, 0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training/Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation silhouette score
        if self.val_metrics and any('silhouette_score' in m for m in self.val_metrics):
            silhouette_scores = [m.get('silhouette_score', 0) for m in self.val_metrics]
            axes[0, 1].plot(silhouette_scores, label='Validation Silhouette', color='green')
            axes[0, 1].set_title('Validation Silhouette Score')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Silhouette Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Number of detected speakers over time
        if self.val_metrics and any('n_speakers' in m for m in self.val_metrics):
            n_speakers = [m.get('n_speakers', 0) for m in self.val_metrics]
            axes[1, 0].plot(n_speakers, label='Detected Speakers', color='orange', marker='o')
            axes[1, 0].set_title('Number of Detected Speakers')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Number of Speakers')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate schedule
        if hasattr(self.scheduler, 'get_last_lr'):
            learning_rates = []
            # This is approximate - in practice you'd want to log this during training
            for epoch in range(len(self.train_losses)):
                learning_rates.append(self.config['learning_rate'] * (0.95 ** epoch))
        else:
            learning_rates = [self.config['learning_rate']] * len(self.train_losses)
            
        axes[1, 1].plot(learning_rates, label='Learning Rate', color='purple')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        plots_path = "plots/training_progress.png"
        os.makedirs(os.path.dirname(plots_path), exist_ok=True)
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        print(f"   📈 Training plots saved to {plots_path}")
        plt.close()

def get_default_config():
    """Get default training configuration"""
    return {
        # Data
        'data_path': '../cache_mels/',  # Fixed path
        'segment_length': 2.0,
        'augmentation_strength': 1.0,
        
        # Model
        'embed_dim': 256,
        'num_heads': 8,
        'dropout': 0.1,
        
        # Training
        'batch_size': 16,
        'learning_rate': 0.0003,
        'epochs': 100,
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        
        # Loss
        'loss_type': 'advanced',
        'temperature': 0.1,
        'use_hard_negatives': True,
        'hard_negative_weight': 2.0,
        
        # Validation
        'val_ratio': 0.2,
        'max_speakers': 8,
        'val_frequency': 5,  # Run validation every 5 epochs for speed
        
        # Early stopping
        'patience': 15,
        'min_delta': 1e-4,
        
        # Other
        'num_workers': 4,
        'max_grad_norm': 1.0,
        'checkpoint_freq': 20,
        'random_state': 42,
        'model_save_path': 'models/best_enhanced_model.pt'
    }

def main():
    """Main training function"""
    print("🚀 Starting Enhanced Bird Diarization Training")
    print("=" * 60)
    
    # Get configuration
    config = get_default_config()
    
    # Override with any custom configurations here
    # config['batch_size'] = 32  # Example override
    
    # Create trainer
    trainer = EnhancedTrainer(config)
    
    try:
        # Train the model
        final_model = trainer.train()
        
        print("\n🎉 Training completed successfully!")
        print("📁 Check the following directories for results:")
        print("   - models/ - Saved model checkpoints")
        print("   - results/ - Evaluation results")
        print("   - plots/ - Training progress plots")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    
    return trainer

if __name__ == "__main__":
    trainer = main()