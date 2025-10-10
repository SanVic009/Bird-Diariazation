import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import math

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class MixUpLoss(nn.Module):
    """Loss function for MixUp/CutMix"""
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
    
    def forward(self, pred, target):
        if isinstance(target, tuple):
            y_a, y_b, lam = target
            return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
        return self.criterion(pred, target)

class AdvancedTrainer:
    def __init__(self, model, train_loader, val_loader, device, num_classes):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Advanced optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=0.003, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # One cycle learning rate
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=0.003,
            epochs=50,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Advanced loss
        self.criterion = FocalLoss(alpha=1, gamma=2)
        self.mixup_criterion = MixUpLoss(self.criterion)
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Augmentations
        from augmentations import AdvancedAugmentation
        self.aug = AdvancedAugmentation(prob=0.7)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Advanced augmentations
            if epoch < 30:  # Use mixup/cutmix for first 30 epochs
                if torch.rand(1) < 0.5:
                    data, target = self.aug.mixup(data, target)
                else:
                    data, target = self.aug.cutmix(data, target)
            
            data = self.aug.spec_augment_advanced(data)
            data = self.aug.gaussian_noise(data)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output = self.model(data)
                loss = self.mixup_criterion(output, target)
            
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy (handle mixup case)
            if not isinstance(target, tuple):
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        if total > 0:
            accuracy = 100. * correct / total
        else:
            accuracy = 0
            
        return total_loss / len(self.train_loader), accuracy
    
    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        val_loss /= len(self.val_loader)
        accuracy = 100. * correct / total
        
        return val_loss, accuracy