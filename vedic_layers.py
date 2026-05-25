import torch
import torch.nn as nn
import math

class ShunyataSamskara(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

class VilokanamOutput(nn.Module):
    def __init__(self, dim, vocab_size, num_classes=6):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02 / math.sqrt(dim))
        self.register_buffer('token_classes', torch.randint(0, num_classes, (vocab_size,)))
    def forward(self, x):
        return self.lm_head(x)

class AnurupyaKVCache:
    def __init__(self, head_dim, alpha_threshold=0.05, norm_tolerance=0.1):
        self.head_dim = head_dim
        self.alpha_threshold = alpha_threshold
        self.norm_tolerance = norm_tolerance
        self.cached_k = []
        self.cached_v = []
        self.total_seen = 0
        self.total_stored = 0
    def should_store(self, k_new, v_new):
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
    def compression_ratio(self):
        if self.total_stored == 0: return 1.0
        return self.total_seen / self.total_stored
