import torch
import torch.nn.functional as F
import numpy as np

class AdvancedAugmentation:
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def mixup(self, x, y, alpha=0.4):
        """MixUp augmentation"""
        if torch.rand(1) > self.prob:
            return x, y
            
        batch_size = x.size(0)
        lam = np.random.beta(alpha, alpha)
        
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index]
        
        y_a, y_b = y, y[index]
        return mixed_x, (y_a, y_b, lam)
    
    def cutmix(self, x, y, alpha=1.0):
        """CutMix augmentation"""
        if torch.rand(1) > self.prob:
            return x, y
            
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        _, _, h, w = x.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        y_a, y_b = y, y[index]
        return x, (y_a, y_b, lam)
    
    def spec_augment_advanced(self, x):
        """Advanced SpecAugment"""
        if torch.rand(1) > self.prob:
            return x
            
        _, _, freq_dim, time_dim = x.shape
        
        # Multiple time masks
        for _ in range(2):
            if torch.rand(1) < 0.8:
                t = torch.randint(0, min(20, time_dim//8), (1,)).item()
                t0 = torch.randint(0, time_dim - t, (1,)).item()
                x[:, :, :, t0:t0+t] = 0
        
        # Multiple frequency masks  
        for _ in range(2):
            if torch.rand(1) < 0.8:
                f = torch.randint(0, min(15, freq_dim//8), (1,)).item()
                f0 = torch.randint(0, freq_dim - f, (1,)).item()
                x[:, :, f0:f0+f, :] = 0
                
        return x
    
    def gaussian_noise(self, x, std=0.1):
        """Add Gaussian noise"""
        if torch.rand(1) > self.prob:
            return x
        noise = torch.randn_like(x) * std
        return x + noise