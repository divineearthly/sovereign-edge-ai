print("DEBUG: vedic_embedding.py is being loaded!")
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
    def __init__(self, vocab_size: int, dim: int, max_val: int = 128, base_freq: float = 0.0001):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_val = max_val
        self.base_freq = base_freq

        # Create frequencies for sine/cosine based on dim
        inv_freq = 1.0 / (self.base_freq ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq] input token IDs
        Returns:
            embeddings: [batch, seq, dim] embeddings
        """
        # Normalize token_ids to a range (e.g., 0 to max_val)
        # A simple linear scaling, more complex Vedic mappings can be here
        normalized_ids = (token_ids.float() / self.vocab_size) * self.max_val

        # Expand normalized_ids for sine/cosine calculation
        sinusoid_inp = torch.einsum('bs,d->bsd', normalized_ids, self.inv_freq)

        embeddings = torch.cat([
            sinusoid_inp.sin(),
            sinusoid_inp.cos()
        ], dim=-1)

        return embeddings


class KalachakraPositionEncoding(nn.Module):
    """Cyclic recursive position encoding.

    Positions are not absolute but relative to cycles, like Kalachakra.
    Memory: O(max_seq * dim) for fixed max_seq, or can be dynamic.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base_cycle: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base_cycle = base_cycle

        # Similar to MatrikaNyasa, but for positions
        inv_freq = 1.0 / (self.base_cycle ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute position encodings up to max_seq_len
        # This can be made dynamic for very long sequences
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum('s,d->sd', position, self.inv_freq)

        pe = torch.cat([
            sinusoid_inp.sin(),
            sinusoid_inp.cos()
        ], dim=-1)
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            seq_len: Current sequence length
            device: Device to place the tensor on
        Returns:
            position_encodings: [seq_len, dim] position embeddings
        """
        assert seq_len <= self.max_seq_len, \
            f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
        return self.pe[:seq_len].to(device)
