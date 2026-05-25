"""
Complete Vedic Normalization Suite — 18 Algorithms
"""

import torch
import torch.nn as nn
import math

class ShunyataNorm(nn.Module):
    """1. Shunyata Samskara — Zero-centered normalization"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class PurnamNorm(nn.Module):
    """2. Purnam — Wholeness normalization (from Isha Upanishad)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Purnamadah purnamidam — whole + whole = whole
        norm = x.norm(dim=-1, keepdim=True)
        return self.gamma * x / (norm + 1e-8) * math.sqrt(x.shape[-1])


class SatyaNorm(nn.Module):
    """3. Satya — Truth/essence normalization (keep only real signal)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Only preserve "truthful" signal — positive and significant
        mask = (x > 0).float()
        return self.gamma * x * mask


class TapasNorm(nn.Module):
    """4. Tapas — Austerity normalization (suppress excess)"""
    def __init__(self, dim, sparsity=0.5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.sparsity = sparsity
    
    def forward(self, x):
        k = int(x.shape[-1] * self.sparsity)
        topk, _ = torch.topk(x.abs(), k, dim=-1)
        threshold = topk[..., -1:]
        return self.gamma * x * (x.abs() >= threshold).float()


class RtaNorm(nn.Module):
    """5. Rta — Cosmic order normalization (harmonic alignment)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.order_freq = nn.Parameter(torch.linspace(1, dim, dim))
    
    def forward(self, x):
        # Align with cosmic harmonic frequencies
        alignment = torch.sin(self.order_freq.unsqueeze(0) * math.pi / x.shape[-1])
        return self.gamma * x * alignment.to(x.device)


class MayaNorm(nn.Module):
    """6. Maya — Illusion normalization (relative to context)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Everything is relative — normalize by local context
        context = x.mean(dim=-2, keepdim=True)
        return self.gamma * (x - context)


class KarmaNorm(nn.Module):
    """7. Karma — Action-consequence normalization (cumulative)"""
    def __init__(self, dim, momentum=0.9):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(1, 1, dim))
    
    def forward(self, x):
        if self.training:
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x.mean(dim=(0,1), keepdim=True)
        return self.gamma * (x - self.running_mean)


class DharmaNorm(nn.Module):
    """8. Dharma — Righteous path normalization (ethical constraint)"""
    def __init__(self, dim, max_norm=1.0):
        super().__init__()
        self.max_norm = max_norm
    
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.max_norm / (norm + 1e-8), max=1.0)
        return x * scale


class AtmanNorm(nn.Module):
    """9. Atman — Self-normalization (inner reference)"""
    def __init__(self, dim):
        super().__init__()
        self.self_ref = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        return x * self.self_ref / (x.norm(dim=-1, keepdim=True) + 1e-8)


class BrahmanNorm(nn.Module):
    """10. Brahman — Universal normalization (infinite context)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Expand context to infinity — use global statistics
        all_mean = x.mean()
        all_std = x.std()
        return self.gamma * (x - all_mean) / (all_std + 1e-8)


class SamsaraNorm(nn.Module):
    """11. Samsara — Cyclic normalization (recurring patterns)"""
    def __init__(self, dim, cycle_len=7):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.cycle_len = cycle_len
    
    def forward(self, x):
        B, S, D = x.shape
        # Group by cycle position
        for i in range(min(S, self.cycle_len)):
            indices = torch.arange(i, S, self.cycle_len)
            if len(indices) > 0:
                group = x[:, indices, :]
                group_mean = group.mean(dim=1, keepdim=True)
                group_var = group.var(dim=1, keepdim=True, unbiased=False)
                x[:, indices, :] = (group - group_mean) / torch.sqrt(group_var + 1e-8)
        return self.gamma * x


class MokshaNorm(nn.Module):
    """12. Moksha — Liberation normalization (free from constraints)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Release from constraints — no fixed mean, just scale
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True))
        return self.gamma * x / (rms + 1e-8)


class AgniNorm(nn.Module):
    """13. Agni — Fire normalization (transforms through intensity)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Fire transforms — sigmoid gating
        intensity = torch.sigmoid(x)
        return self.gamma * x * intensity


class SomaNorm(nn.Module):
    """14. Soma — Nectar normalization (smooth, blissful)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Smooth normalization — tanh softening
        return self.gamma * torch.tanh(x)


class VayuNorm(nn.Module):
    """15. Vayu — Wind normalization (flowing, adaptive)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Wind flows — exponential moving average
        diff = x - x.mean(dim=-1, keepdim=True)
        return self.gamma * diff / (diff.std(dim=-1, keepdim=True) + 1e-8)


class AkashaNorm(nn.Module):
    """16. Akasha — Space normalization (expansive)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Space contains all — minimal constraint
        return self.gamma * x / (x.abs().max(dim=-1, keepdim=True).values + 1e-8)


class PrithviNorm(nn.Module):
    """17. Prithvi — Earth normalization (grounded, stable)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('earth_mean', torch.zeros(dim))
        self.register_buffer('earth_std', torch.ones(dim))
    
    def forward(self, x):
        if self.training:
            self.earth_mean = 0.99 * self.earth_mean + 0.01 * x.mean(dim=(0,1))
            self.earth_std = 0.99 * self.earth_std + 0.01 * x.std(dim=(0,1))
        return self.gamma * (x - self.earth_mean) / (self.earth_std + 1e-8)


class LilaNorm(nn.Module):
    """18. Lila — Divine play normalization (joyful, creative)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Playful — mix of order and chaos
        ordered = x / (x.std(dim=-1, keepdim=True) + 1e-8)
        creative = torch.sin(x) * 0.1
        return self.gamma * (ordered + creative)
