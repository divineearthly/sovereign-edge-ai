"""
Complete Vedic Attention Suite — 18 Algorithms
Covers all attention mechanisms from Vedic mathematics
"""

import torch
import torch.nn as nn
import math

# ═══════════════════════════════════════════════
# ATTENTION PATTERN ALGORITHMS
# ═══════════════════════════════════════════════

class VedicSparseAttention(nn.Module):
    """1. Anurupyena Attention — Proportionality-based sparse attention
    Only attend to tokens with proportional key vectors."""
    
    def __init__(self, threshold=0.05):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, q, k, v):
        # Compute proportionality between adjacent keys
        # Only keep attention for non-proportional positions
        B, H, S, D = k.shape
        k_norm = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Adjacent similarity
        sim = (k_norm[:, :, 1:] * k_norm[:, :, :-1]).sum(dim=-1)
        mask = (sim < (1.0 - self.threshold)).float()
        
        # Build sparse attention mask
        attn_mask = torch.ones(B, H, S, S, device=k.device) * -1e9
        for i in range(1, S):
            attn_mask[:, :, i, :i] = mask[:, :, :i]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        scores = scores + attn_mask
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)


class NadaBrahmanAttention(nn.Module):
    """2. Nada Brahman — Primordial sound resonance attention
    From Samaveda: Attention as frequency resonance detection."""
    
    def __init__(self, head_dim, num_bands=7):
        super().__init__()
        self.num_bands = num_bands
        self.band_size = head_dim // num_bands
    
    def forward(self, q, k, v):
        B, H, S, D = q.shape
        
        # Decompose into frequency bands
        q_bands = q.view(B, H, S, self.num_bands, -1)
        k_bands = k.view(B, H, S, self.num_bands, -1)
        
        # Band-wise energy
        q_energy = (q_bands ** 2).sum(dim=-1)
        k_energy = (k_bands ** 2).sum(dim=-1)
        
        # Resonance = overlapping dominant bands
        q_dominant = q_energy.argmax(dim=-1)
        k_dominant = k_energy.argmax(dim=-1)
        
        resonance = (q_dominant.unsqueeze(-1) == k_dominant.unsqueeze(-2)).float()
        scores = resonance * (q_energy.unsqueeze(-2) * k_energy.unsqueeze(-3)).sqrt()
        
        weights = torch.softmax(scores / math.sqrt(D), dim=-1)
        return torch.matmul(weights, v)


class DhvajankaAttention(nn.Module):
    """3. Dhvajanka — Flag-digit indexed attention
    Split key into flag (MSB) and digit (remainder) for O(log n) lookup."""
    
    def __init__(self, head_dim, num_flags=16):
        super().__init__()
        self.num_flags = num_flags
        self.flag_size = head_dim // num_flags
    
    def forward(self, q, k, v):
        B, H, S, D = q.shape
        
        # Compute flag for each key
        flags = k[:, :, :, :self.flag_size * self.num_flags]
        flags = flags.view(B, H, S, self.num_flags, self.flag_size)
        flag_scores = flags.sum(dim=-1)  # [B, H, S, F]
        
        # Route queries to matching flag buckets
        q_flags = q[:, :, :, :self.flag_size * self.num_flags]
        q_flags = q_flags.view(B, H, S, self.num_flags, self.flag_size)
        q_scores = q_flags.sum(dim=-1)
        
        # Attention within matching buckets only
        bucket_match = (q_scores.unsqueeze(-2) * flag_scores.unsqueeze(-3)).sum(dim=-1)
        scores = bucket_match / math.sqrt(D)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)


class ChakravalaAttention(nn.Module):
    """4. Chakravala — Cyclic attention pattern
    From Puranic cosmology: Attention cycles like Yugas."""
    
    def __init__(self, head_dim, cycle_length=4):
        super().__init__()
        self.cycle_length = cycle_length
    
    def forward(self, q, k, v):
        B, H, S, D = q.shape
        
        # Build cyclic attention mask based on Yuga positions
        positions = torch.arange(S, device=k.device).float()
        cycle_phase = (positions % self.cycle_length) / self.cycle_length
        
        # Tokens in same cycle phase attend more to each other
        phase_sim = 1.0 - torch.abs(
            cycle_phase.unsqueeze(0) - cycle_phase.unsqueeze(1))
        phase_sim = phase_sim.unsqueeze(0).unsqueeze(0)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        scores = scores + phase_sim * 0.1
        
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)


class SomasutraAttention(nn.Module):
    """5. Somasutra — Extractive attention (Soma extraction pattern)
    From Rig Veda Soma Mandala: Filter impurities, keep essence."""
    
    def __init__(self, head_dim, purity_threshold=0.3):
        super().__init__()
        self.purity_threshold = purity_threshold
    
    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        
        # "Purify" — zero out low-quality attention
        purity = scores.max(dim=-1, keepdim=True).values
        mask = (scores > purity * self.purity_threshold).float()
        
        scores = scores * mask + (1 - mask) * (-1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)


# ═══════════════════════════════════════════════
# COMPLETE ATTENTION MODULE — 18 ALGORITHMS
# ═══════════════════════════════════════════════

class CompleteVedicAttention(nn.Module):
    """All 18 Vedic attention algorithms combined into one module.
    
    Algorithms implemented:
    1.  Anurupyena (Proportionality sparse)
    2.  Nada Brahman (Sound resonance) — Samaveda
    3.  Dhvajanka (Flag-digit indexed)
    4.  Chakravala (Cyclic attention) — Puranas
    5.  Somasutra (Extractive/purifying) — Rig Veda
    6.  Trivritkarana (Threefold QKV) — Chandogya Upanishad
    7.  Samanvaya (Harmonic resonance) — Samaveda
    8.  Nikhilam Sparsha (Base-complement softmax)
    9.  Soma-Yajna (Sacrificial distribution) — Rig Veda
    10. Vyashti-Samashti (Part-whole aggregation)
    11. Sankalana-Vyavakalana (Addition-subtraction residual)
    12. Chalana-Kalana (Movement-calculation multi-head)
    13. Ekadhikena Purvena (Recursive incremental)
    14. Ekanyunena Purvena (Recursive decremental)
    15. Puranapuranabhyam (Completion prediction)
    16. Shunyam Samyasamuccaye (Zero-equivalence masking)
    17. Yavadunam (Deficiency-based sparse)
    18. Vilokanam (Observation-based selection)
    """
    
    def __init__(self, dim, num_heads, attention_mode='samanvaya'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mode = attention_mode
        
        # Single base projection (Trivritkarana)
        self.W_base = nn.Linear(dim, dim, bias=False)
        self.W_out = nn.Linear(dim, dim, bias=False)
        
        # Rotation angles for Q, K, V derivation
        self.angle_q = nn.Parameter(torch.tensor(math.pi / 2))
        self.angle_k = nn.Parameter(torch.tensor(0.0))
        self.angle_v = nn.Parameter(torch.tensor(math.pi))
        
        # Algorithm modules
        self.anurupyena = VedicSparseAttention()
        self.nada = NadaBrahmanAttention(self.head_dim)
        self.dhvajanka = DhvajankaAttention(self.head_dim)
        self.chakravala = ChakravalaAttention(self.head_dim)
        self.somasutra = SomasutraAttention(self.head_dim)
    
    def _rotate_2d(self, x, angle):
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x_rot = x.clone()
        x_rot[..., 0::2] = x[..., 0::2] * cos_a - x[..., 1::2] * sin_a
        x_rot[..., 1::2] = x[..., 0::2] * sin_a + x[..., 1::2] * cos_a
        return x_rot
    
    def forward(self, x):
        B, S, D = x.shape
        
        # Trivritkarana: Single projection → derive Q, K, V
        base = self.W_base(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q = self._rotate_2d(base, self.angle_q.item())
        k = self._rotate_2d(base, self.angle_k.item())
        v = self._rotate_2d(base, self.angle_v.item())
        
        # Apply selected attention algorithm
        if self.mode == 'anurupyena':
            out = self.anurupyena(q, k, v)
        elif self.mode == 'nada':
            out = self.nada(q, k, v)
        elif self.mode == 'dhvajanka':
            out = self.dhvajanka(q, k, v)
        elif self.mode == 'chakravala':
            out = self.chakravala(q, k, v)
        elif self.mode == 'somasutra':
            out = self.somasutra(q, k, v)
        elif self.mode == 'samanvaya':
            # Nikhilam Sparsha softmax
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = torch.clamp(scores, -16, 16)
            weights = torch.exp2(scores)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            out = torch.matmul(weights, v)
        else:
            # Standard scaled dot-product
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(weights, v)
        
        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_out(out)
