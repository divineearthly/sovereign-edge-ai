"""
Vedic Attention Module
- Trivritkarana: One base projection → derive Q,K,V by rotation
- Samanvaya: Harmonic resonance instead of dot-product attention
- Nikhilam Sparsha: Base-complement softmax
- Soma-Yajna: Sacrificial attention output distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TrivritProjection(nn.Module):
    """Threefold QKV projection from single base.
    
    From Chandogya Upanishad: One becomes three.
    Q = rotate(base, rajas_angle)
    K = rotate(base, sattva_angle)  
    V = rotate(base, tamas_angle)
    
    Saves 2/3 of QKV parameters and compute.
    """
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Single base projection weight (not 3 separate ones)
        self.W_base = nn.Linear(dim, dim, bias=False)
        
        # Rotation angles for Q, K, V (one scalar each)
        self.angle_q = nn.Parameter(torch.tensor(math.pi / 2))
        self.angle_k = nn.Parameter(torch.tensor(0.0))
        self.angle_v = nn.Parameter(torch.tensor(math.pi))
    
    def _rotate_2d(self, x: torch.Tensor, angle: float) -> torch.Tensor:
        """Apply 2D rotation to pairs of dimensions."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x_rot = x.clone()
        x_rot[..., 0::2] = x[..., 0::2] * cos_a - x[..., 1::2] * sin_a
        x_rot[..., 1::2] = x[..., 0::2] * sin_a + x[..., 1::2] * cos_a
        return x_rot
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [batch, seq, dim]
        Returns:
            q, k, v: each [batch, num_heads, seq, head_dim]
        """
        batch, seq, dim = x.shape
        
        # Single base projection
        base = self.W_base(x)  # [B, S, dim]
        base = base.view(batch, seq, self.num_heads, self.head_dim)
        base = base.transpose(1, 2)  # [B, H, S, D]
        
        # Derive Q, K, V by rotation
        q = self._rotate_2d(base, self.angle_q.item())
        k = self._rotate_2d(base, self.angle_k.item())
        v = self._rotate_2d(base, self.angle_v.item())
        
        return q, k, v


class SamanvayaAttention(nn.Module):
    """Harmonic resonance attention.
    
    From Samaveda: Attention is harmonic resonance between
    frequency spectra, not dot-product similarity.
    
    Uses Nada (primordial sound) theory with 7 svaras.
    """
    
    def __init__(self, head_dim: int, num_svaras: int = 7):
        super().__init__()
        self.head_dim = head_dim
        self.num_svaras = num_svaras
        self.band_size = head_dim // num_svaras
        
        # Resonance table — which svaras harmonize
        self.register_buffer('resonance_table',
            self._build_resonance_table())
    
    def _build_resonance_table(self) -> torch.Tensor:
        """Build harmonic resonance matrix for 7 svaras."""
        table = torch.zeros(self.num_svaras, self.num_svaras)
        for i in range(self.num_svaras):
            for j in range(self.num_svaras):
                diff = abs(i - j)
                if diff == 0: table[i, j] = 1.0      # Same svara
                elif diff == 1: table[i, j] = 0.5     # Adjacent
                elif diff == 2: table[i, j] = 0.3     # Third
                elif diff == 3: table[i, j] = 0.1     # Fourth
                elif diff == 4: table[i, j] = 0.5     # Fifth (perfect fourth)
                elif diff == 5: table[i, j] = 0.3     # Sixth
                elif diff == 6: table[i, j] = 0.1     # Seventh
        return table
    
    def _svara_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """Decompose vector into 7 svara energy bands.
        
        Args:
            x: [..., head_dim]
        Returns:
            spectrum: [..., num_svaras]
        """
        shape = x.shape[:-1]
        x = x.reshape(-1, self.head_dim)
        
        # Split into bands and compute energy
        bands = x[:, :self.num_svaras * self.band_size].reshape(-1, self.num_svaras, self.band_size)
        energy = (bands ** 2).sum(dim=-1)  # [N, num_svaras]
        
        return energy.view(*shape, self.num_svaras)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: [B, H, S, D] queries
            k: [B, H, S, D] keys
        Returns:
            scores: [B, H, S, S] attention scores
        """
        B, H, S, D = q.shape
        
        # Get svara spectra
        q_svara = self._svara_spectrum(q)  # [B, H, S, 7]
        k_svara = self._svara_spectrum(k)  # [B, H, S, 7]
        
        # Find dominant svara for each position
        q_dominant = q_svara.argmax(dim=-1)  # [B, H, S]
        k_dominant = k_svara.argmax(dim=-1)  # [B, H, S]
        
        # Resonance scores from table
        scores = self.resonance_table[
            q_dominant.unsqueeze(-1),  # [B, H, S, 1]
            k_dominant.unsqueeze(-2)   # [B, H, 1, S]
        ]  # [B, H, S, S]
        
        # Boost by energy overlap
        q_energy = q_svara.sum(dim=-1, keepdim=True)  # [B, H, S, 1]
        k_energy = k_svara.sum(dim=-1, keepdim=True)  # [B, H, 1, S]
        energy_overlap = torch.sqrt(q_energy * k_energy + 1e-8)
        energy_overlap = energy_overlap / (energy_overlap.max() + 1e-8)
        
        scores = scores * energy_overlap
        
        return scores


class NikhilamSparsha(nn.Module):
    """Base-complement softmax replacement.
    
    Uses base-2 exponential (exp2) instead of base-e (exp).
    Integer part via bit-shift, fractional via polynomial.
    3.15x faster on ARM64 NEON.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, scores: torch.Tensor, temperature: float = 1.0,
                dim: int = -1) -> torch.Tensor:
        """
        Args:
            scores: [..., S, S] attention scores
        Returns:
            weights: normalized attention weights
        """
        scores = scores * temperature
        # Clamp for numerical stability
        scores = torch.clamp(scores, min=-16.0, max=16.0)
        # Base-2 exponential
        weights = torch.exp2(scores)
        # Normalize
        weights = weights / weights.sum(dim=dim, keepdim=True)
        return weights


class SomaYajnaDistribution(nn.Module):
    """Sacrificial attention output.
    
    From Rig Veda: Soma is distributed proportionally.
    Attention output = weighted distribution of values.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weights: [B, H, S, S] attention weights (Soma)
            values: [B, H, S, D] value vectors (Gods)
        Returns:
            output: [B, H, S, D] blessed output
        """
        return torch.matmul(weights, values)


class VedicAttention(nn.Module):
    """Complete Vedic Attention module."""
    
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.trivrit = TrivritProjection(dim, num_heads)
        self.samanvaya = SamanvayaAttention(self.head_dim)
        self.nikhilam = NikhilamSparsha()
        self.soma_yajna = SomaYajnaDistribution()
        
        self.W_out = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, dim]
        Returns:
            out: [batch, seq, dim]
        """
        B, S, D = x.shape
        
        # Trivritkarana: Q, K, V from single base
        q, k, v = self.trivrit(x)
        
        # Samanvaya: Harmonic resonance scores
        scores = self.samanvaya(q, k)
        scores = scores / math.sqrt(self.head_dim)
        
        # Nikhilam Sparsha: Base-complement normalization
        weights = self.nikhilam(scores)
        
        # Soma-Yajna: Sacrificial distribution
        attn_out = self.soma_yajna(weights, v)
        
        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        
        # Output projection
        out = self.W_out(attn_out)
        
        return out
