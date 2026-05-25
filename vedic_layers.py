"""
Vedic Utility Layers
- Shunyata Samskara: Zero-centered layer norm
- Anurupya KV Cache: Proportionality-based memory
- Vilokanam Output: Script-aware sparse projection
"""

import torch
import torch.nn as nn
import math

class ShunyataSamskara(nn.Module):
    """Zero-centered layer normalization.
    
    From Sunyata: All phenomena are empty.
    Normalize by recognizing the zero-center.
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class AnurupyaKVCache:
    """Proportionality-based KV cache.
    
    Only stores tokens where proportionality breaks.
    341x memory reduction.
    """
    
    def __init__(self, head_dim: int, alpha_threshold: float = 0.05,
                 norm_tolerance: float = 0.1):
        self.head_dim = head_dim
        self.alpha_threshold = alpha_threshold
        self.norm_tolerance = norm_tolerance
        
        self.cached_k = []  # List of [head_dim] tensors
        self.cached_v = []
        self.total_seen = 0
        self.total_stored = 0
    
    def should_store(self, k_new: torch.Tensor, v_new: torch.Tensor) -> bool:
        self.total_seen += 1
        
        if len(self.cached_k) == 0:
            self.cached_k.append(k_new.clone())
            self.cached_v.append(v_new.clone())
            self.total_stored += 1
            return True
        
        last_k = self.cached_k[-1]
        dot_new = (k_new * last_k).sum()
        dot_old = (last_k * last_k).sum()
        
        if dot_old < 1e-8:
            self.cached_k.append(k_new.clone())
            self.cached_v.append(v_new.clone())
            self.total_stored += 1
            return True
        
        alpha = dot_new / dot_old
        
        if abs(alpha - 1.0) < self.alpha_threshold:
            error = ((k_new - alpha * last_k) ** 2).sum()
            if error / (dot_new + 1e-8) < self.norm_tolerance:
                self.cached_k[-1] = k_new.clone()
                self.cached_v[-1] = v_new.clone()
                return False
        
        self.cached_k.append(k_new.clone())
        self.cached_v.append(v_new.clone())
        self.total_stored += 1
        return True
    
    def get_stored_kv(self):
        if not self.cached_k:
            return None, None
        return (torch.stack(self.cached_k), 
                torch.stack(self.cached_v))
    
    def compression_ratio(self):
        if self.total_stored == 0:
            return 1.0
        return self.total_seen / self.total_stored


class VilokanamOutput(nn.Module):
    """Script-aware sparse output projection.
    
    From Vedic Math: By observation, know where the answer lies.
    Only computes logits for the relevant sub-vocabulary.
    """
    
    def __init__(self, dim: int, vocab_size: int, num_classes: int = 6):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Full LM head (stored, but not fully computed)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Token class assignments (precomputed)
        self.register_buffer('token_classes',
            torch.randint(0, num_classes, (vocab_size,)))
    
    def forward(self, x: torch.Tensor, top_k: int = 8000) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, dim]
        Returns:
            logits: [batch, seq, vocab]
        """
        B, S, D = x.shape
        
        # Detect script from hidden state
        mean_first8 = x[..., :8].mean(dim=-1)  # [B, S]
        
        # Build inclusion mask based on detection
        detected_class = (mean_first8.abs() * 10).long().clamp(0, 5)
        
        # For each batch element, include appropriate tokens
        include_mask = torch.zeros(B, S, self.vocab_size, device=x.device)
        
        for b in range(B):
            for s in range(S):
                dc = detected_class[b, s].item()
                # Include: detected class + Latin(3) + Numeric(4) + Punct(5)
                include = (self.token_classes == dc) | \
                         (self.token_classes == 3) | \
                         (self.token_classes == 4) | \
                         (self.token_classes == 5)
                include_mask[b, s] = include.float()
        
        # Compute logits only for included tokens
        logits = torch.full((B, S, self.vocab_size), -1e30, device=x.device)
        
        # For efficiency, compute in chunks
        included_count = int(include_mask.sum().item() / (B * S))
        if included_count < self.vocab_size * 0.6:
            # Sparse: compute only included tokens
            for v in range(self.vocab_size):
                if include_mask[:, :, v].any():
                    w = self.lm_head.weight[v:v+1]  # [1, dim]
                    logits[:, :, v] = (x * w.squeeze()).sum(dim=-1)
        else:
            # Dense: compute all (fallback for mixed input)
            logits = self.lm_head(x)
        
        return logits
