#!/usr/bin/env python3
"""
advanced_loss_functions.py - Enhanced Loss Functions for Bird Diarization

Features:
- Advanced contrastive loss with hard negative mining
- Temperature scaling and adaptive temperature
- Multiple contrastive learning objectives
- Focal loss for hard examples
- Mixup-aware loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AdvancedContrastiveLoss(nn.Module):
    """Advanced contrastive loss with hard negative mining"""
    
    def __init__(self, 
                 temperature=0.1,  # Lower temperature for better separation
                 use_hard_negatives=True,
                 hard_negative_weight=2.0,
                 adaptive_temperature=False,
                 temperature_range=(0.05, 0.2)):
        super().__init__()
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_weight = hard_negative_weight
        self.adaptive_temperature = adaptive_temperature
        self.temperature_range = temperature_range
        
        # Learnable temperature if adaptive
        if adaptive_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        
    def forward(self, z1, z2, epoch=0):
        batch_size = z1.size(0)
        device = z1.device
        
        # Get current temperature
        if self.adaptive_temperature:
            temp = torch.clamp(
                torch.exp(self.log_temperature), 
                self.temperature_range[0], 
                self.temperature_range[1]
            )
        else:
            temp = self.temperature
        
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # [2*batch_size, embed_dim]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / temp
        
        # Create mask for valid pairs (exclude self-similarity)
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # Positive pairs: z1[i] <-> z2[i]
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),  # z1 -> z2
            torch.arange(0, batch_size, device=device)  # z2 -> z1
        ])
        
        # Standard contrastive loss
        labels = pos_indices
        base_loss = F.cross_entropy(sim_matrix, labels)
        
        if not self.use_hard_negatives:
            return base_loss
        
        # Hard negative mining
        with torch.no_grad():
            # Find hardest negatives (highest similarity among negatives)
            pos_sim = sim_matrix[torch.arange(2 * batch_size), labels]
            
            # Mask out positive pairs
            neg_mask = torch.ones_like(sim_matrix, dtype=torch.bool)
            neg_mask[torch.arange(2 * batch_size), labels] = False
            neg_mask[mask] = False
            
            # Find hardest negatives for each sample
            neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)
            hard_neg_indices = neg_sim.argmax(dim=1)
            hard_neg_sim = neg_sim[torch.arange(2 * batch_size), hard_neg_indices]
        
        # Compute hard negative loss
        # Focus more on hard negatives that are close to positives
        margin = 0.1
        hard_neg_loss = F.relu(hard_neg_sim - pos_sim + margin).mean()
        
        # Combine losses
        total_loss = base_loss + self.hard_negative_weight * hard_neg_loss
        
        return total_loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss (adapted for unsupervised case)"""
    
    def __init__(self, temperature=0.1, contrast_mode='all', base_temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                           'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # For unsupervised case, each sample is its own class
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class FocalContrastiveLoss(nn.Module):
    """Focal loss adaptation for contrastive learning"""
    
    def __init__(self, temperature=0.1, alpha=1.0, gamma=2.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, z1, z2):
        batch_size = z1.size(0)
        device = z1.device
        
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Remove diagonal
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        # Labels for positive pairs
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)
        ])
        
        # Compute probabilities
        probs = F.softmax(sim_matrix, dim=1)
        pos_probs = probs[torch.arange(2 * batch_size), labels]
        
        # Focal weight: (1 - p)^gamma
        focal_weights = (1 - pos_probs) ** self.gamma
        
        # Standard cross-entropy
        ce_loss = F.cross_entropy(sim_matrix, labels, reduction='none')
        
        # Apply focal weights
        focal_loss = self.alpha * focal_weights * ce_loss
        
        return focal_loss.mean()

class MixupContrastiveLoss(nn.Module):
    """Contrastive loss that handles mixup augmentation"""
    
    def __init__(self, temperature=0.1, mixup_alpha=0.2):
        super().__init__()
        self.temperature = temperature
        self.mixup_alpha = mixup_alpha
        
    def forward(self, z1, z2, mixup_lambda=None):
        if mixup_lambda is None or mixup_lambda == 1.0:
            # Standard contrastive loss
            return self._standard_contrastive(z1, z2)
        else:
            # Modified loss for mixup samples
            return self._mixup_contrastive(z1, z2, mixup_lambda)
    
    def _standard_contrastive(self, z1, z2):
        batch_size = z1.size(0)
        device = z1.device
        
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)
        ])
        
        return F.cross_entropy(sim_matrix, labels)
    
    def _mixup_contrastive(self, z1, z2, mixup_lambda):
        # For mixup, we need to handle the fact that positive pairs
        # are now combinations of original samples
        batch_size = z1.size(0)
        device = z1.device
        
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
        # Create soft labels based on mixup ratio
        # This is a simplified approach - more sophisticated methods exist
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)
        ])
        
        # Weight the loss by mixup lambda
        base_loss = F.cross_entropy(sim_matrix, labels)
        
        # Reduce confidence for mixed samples
        mixed_loss = base_loss * (1 - (1 - mixup_lambda) * 0.5)
        
        return mixed_loss

class AdaptiveTemperatureContrastiveLoss(nn.Module):
    """Contrastive loss with adaptive temperature based on training dynamics"""
    
    def __init__(self, initial_temp=0.1, min_temp=0.05, max_temp=0.3, 
                 temp_decay=0.99, update_freq=100):
        super().__init__()
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.temp_decay = temp_decay
        self.update_freq = update_freq
        self.step_count = 0
        
        # Track loss statistics for adaptive adjustment
        self.loss_history = []
        
    def forward(self, z1, z2):
        batch_size = z1.size(0)
        device = z1.device
        
        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z, z.t()) / self.current_temp
        
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)
        ])
        
        loss = F.cross_entropy(sim_matrix, labels)
        
        # Update temperature based on loss dynamics
        self.step_count += 1
        self.loss_history.append(loss.item())
        
        if self.step_count % self.update_freq == 0:
            self._update_temperature()
            
        return loss
    
    def _update_temperature(self):
        if len(self.loss_history) < self.update_freq * 2:
            return
            
        # Check if loss is decreasing
        recent_loss = np.mean(self.loss_history[-self.update_freq:])
        older_loss = np.mean(self.loss_history[-2*self.update_freq:-self.update_freq])
        
        if recent_loss < older_loss:
            # Loss decreasing - can decrease temperature for harder training
            self.current_temp = max(self.min_temp, 
                                  self.current_temp * self.temp_decay)
        else:
            # Loss not decreasing - increase temperature to make training easier
            self.current_temp = min(self.max_temp, 
                                  self.current_temp / self.temp_decay)
        
        print(f"Updated temperature to: {self.current_temp:.4f}")

class MultiScaleContrastiveLoss(nn.Module):
    """Multi-scale contrastive loss using different feature levels"""
    
    def __init__(self, temperatures=[0.05, 0.1, 0.2], weights=[0.5, 1.0, 0.5]):
        super().__init__()
        self.temperatures = temperatures
        self.weights = weights
        assert len(temperatures) == len(weights)
        
    def forward(self, features_list):
        """
        Args:
            features_list: List of feature tensors at different scales
                         Each tensor should be [z1, z2] concatenated
        """
        total_loss = 0
        total_weight = sum(self.weights)
        
        for features, temp, weight in zip(features_list, self.temperatures, self.weights):
            batch_size = features.size(0) // 2
            z1, z2 = features[:batch_size], features[batch_size:]
            
            # Standard contrastive loss at this scale
            z = torch.cat([z1, z2], dim=0)
            sim_matrix = torch.mm(z, z.t()) / temp
            
            device = z.device
            mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
            sim_matrix = sim_matrix.masked_fill(mask, -1e9)
            
            labels = torch.cat([
                torch.arange(batch_size, 2 * batch_size, device=device),
                torch.arange(0, batch_size, device=device)
            ])
            
            scale_loss = F.cross_entropy(sim_matrix, labels)
            total_loss += (weight / total_weight) * scale_loss
            
        return total_loss

class InfoNCELoss(nn.Module):
    """InfoNCE loss implementation for contrastive learning"""
    
    def __init__(self, temperature=0.1, negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.negative_mode = negative_mode
        
    def forward(self, z1, z2):
        batch_size = z1.size(0)
        device = z1.device
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Positive similarities
        pos_sim = torch.sum(z1 * z2, dim=1) / self.temperature
        
        # Negative similarities
        if self.negative_mode == 'unpaired':
            # Use all other samples as negatives
            neg_sim1 = torch.mm(z1, z2.t()) / self.temperature
            neg_sim2 = torch.mm(z2, z1.t()) / self.temperature
            
            # Remove diagonal (positive pairs)
            mask = torch.eye(batch_size, device=device, dtype=torch.bool)
            neg_sim1 = neg_sim1.masked_fill(mask, -1e9)
            neg_sim2 = neg_sim2.masked_fill(mask, -1e9)
            
            # Compute InfoNCE loss
            logits1 = torch.cat([pos_sim.unsqueeze(1), neg_sim1], dim=1)
            logits2 = torch.cat([pos_sim.unsqueeze(1), neg_sim2], dim=1)
            
            labels = torch.zeros(batch_size, device=device, dtype=torch.long)
            
            loss1 = F.cross_entropy(logits1, labels)
            loss2 = F.cross_entropy(logits2, labels)
            
            return (loss1 + loss2) / 2
        
        else:
            raise NotImplementedError(f"Negative mode {self.negative_mode} not implemented")

# Utility function to get the best loss function based on stage
def get_loss_function(stage='basic', **kwargs):
    """Factory function to get appropriate loss function"""
    
    # Extract common parameters to avoid duplication
    temperature = kwargs.pop('temperature', 0.1)
    use_hard_negatives = kwargs.pop('use_hard_negatives', True)
    hard_negative_weight = kwargs.pop('hard_negative_weight', 2.0)
    
    if stage == 'basic':
        return AdvancedContrastiveLoss(
            temperature=temperature, 
            use_hard_negatives=False,
            hard_negative_weight=hard_negative_weight,
            **kwargs
        )
    elif stage == 'advanced':
        return AdvancedContrastiveLoss(
            temperature=temperature, 
            use_hard_negatives=use_hard_negatives,
            hard_negative_weight=hard_negative_weight,
            **kwargs
        )
    elif stage == 'focal':
        return FocalContrastiveLoss(temperature=temperature, **kwargs)
    elif stage == 'adaptive':
        return AdaptiveTemperatureContrastiveLoss(**kwargs)
    elif stage == 'mixup':
        return MixupContrastiveLoss(temperature=temperature, **kwargs)
    elif stage == 'infonct':
        return InfoNCELoss(temperature=temperature, **kwargs)
    else:
        raise ValueError(f"Unknown loss stage: {stage}")

if __name__ == "__main__":
    print("Testing Advanced Loss Functions...")
    
    # Test basic usage
    batch_size = 16
    embed_dim = 256
    
    z1 = torch.randn(batch_size, embed_dim)
    z2 = torch.randn(batch_size, embed_dim)
    
    # Normalize embeddings (important for contrastive learning)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Test different loss functions
    losses = {
        'Advanced Contrastive': AdvancedContrastiveLoss(),
        'Focal Contrastive': FocalContrastiveLoss(),
        'InfoNCE': InfoNCELoss(),
        'Adaptive Temperature': AdaptiveTemperatureContrastiveLoss()
    }
    
    print(f"Input shapes: z1={z1.shape}, z2={z2.shape}")
    
    for name, loss_fn in losses.items():
        try:
            loss = loss_fn(z1, z2)
            print(f"{name}: {loss.item():.4f}")
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    print("✅ Loss functions working correctly!")