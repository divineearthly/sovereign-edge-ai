"""
Vedic Embedding Layer
- Matrika Nyasa: Algorithmic phoneme→vector encoding
- Kalachakra: Cyclic recursive position encoding
"""

import torch
import torch.nn as nn
import math

class MatrikaNyasaEmbedding(nn.Module):
    """Algorithmic embedding — no lookup table needed.
    
    Maps token IDs to vectors using phonetic principles
    from the Sanskrit Matrika (alphabet of seed syllables).
    Memory: O(dim) instead of O(vocab * dim).
    """
    
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Base frequencies for each phonetic category (10 categories)
        self.register_buffer('category_phase', 
            torch.linspace(0, 2*math.pi, 10))
        
        # Learnable modulation for each category
        self.category_modulation = nn.Parameter(
            torch.randn(10, dim) * 0.02)
    
    def _get_phonetic_category(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to phonetic categories based on Unicode ranges.
        
        Simplified: use token_id % 10 as pseudo-category.
        In production, use actual Unicode analysis of token text.
        """
        return token_ids % 10
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Generate embeddings algorithmically.
        
        Args:
            token_ids: [batch, seq_len] integer token IDs
        
        Returns:
            embeddings: [batch, seq_len, dim]
        """
        batch, seq = token_ids.shape
        categories = self._get_phonetic_category(token_ids)  # [B, S]
        
        # Each dimension is a harmonic of the phonetic category
        positions = torch.arange(self.dim, device=token_ids.device).float()
        harmonics = positions / self.dim  # [dim]
        
        # Generate sinusoidal base
        phase = self.category_phase[categories]  # [B, S, 1]
        embedding = torch.sin(
            phase.unsqueeze(-1) * harmonics.unsqueeze(0).unsqueeze(0) 
            + positions.unsqueeze(0).unsqueeze(0) * 0.01
        )  # [B, S, dim]
        
        # Apply category-specific modulation
        modulation = self.category_modulation[categories]  # [B, S, dim]
        embedding = embedding * modulation * 0.1
        
        return embedding


class KalachakraPositionEncoding(nn.Module):
    """Cyclic recursive position encoding.
    
    From Puranic cosmology: Time cycles through 4 Yugas
    in 4:3:2:1 ratio. Position[t] = rotate(Position[t-1]).
    Perfect for autoregressive generation.
    """
    
    def __init__(self, dim: int, max_seq: int = 32768):
        super().__init__()
        self.dim = dim
        self.yuga_ratios = nn.Parameter(
            torch.tensor([4.0, 3.0, 2.0, 1.0]) / 10.0,
            requires_grad=False)
        
        # Learnable rotation angles per head
        self.base_angle = nn.Parameter(torch.tensor(0.0003))
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate cyclic position encodings.
        
        Args:
            seq_len: Number of positions
            device: torch device
        
        Returns:
            pos_enc: [seq_len, dim]
        """
        # Determine which Yuga each position falls in
        positions = torch.arange(seq_len, device=device).float()
        cycle_pos = positions % 10.0  # Yuga cycle = 10
        
        # Assign Yuga index and compute phase
        yuga_idx = torch.zeros(seq_len, device=device).long()
        yuga_idx[cycle_pos >= 4.0] = 1
        yuga_idx[cycle_pos >= 7.0] = 2
        yuga_idx[cycle_pos >= 9.0] = 3
        
        # Cumulative phase based on Yuga ratio
        ratios = self.yuga_ratios[yuga_idx]  # [seq_len]
        phase = torch.cumsum(ratios * self.base_angle, dim=0)  # [seq_len]
        
        # Generate sinusoidal encoding
        freqs = torch.arange(self.dim, device=device).float() / self.dim
        pos_enc = torch.sin(phase.unsqueeze(-1) * freqs.unsqueeze(0))
        
        return pos_enc * 0.1
