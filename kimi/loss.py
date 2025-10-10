# loss.py
import torch
import torch.nn.functional as F

def nt_xent(z1, z2, temperature=0.1):
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    logits = torch.mm(z, z.t()) / temperature
    n = z1.shape[0]
    labels = torch.arange(2*n, device=z.device)
    mask = torch.eye(2*n, device=z.device).bool()
    logits[mask] = -65500  # Use fp16-safe value instead of -1e9
    return F.cross_entropy(logits, torch.cat([labels[n:], labels[:n]]))