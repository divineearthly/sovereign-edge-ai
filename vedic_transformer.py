"""
Complete Vedic Transformer Model
Uses only Vedic-Upanishadic-Puranic operations throughout.
"""

import torch
import torch.nn as nn

from vedic_embedding import MatrikaNyasaEmbedding, KalachakraPositionEncoding
from vedic_attention import VedicAttention
from vedic_ffn import PanchikaranaFFN, TrigunaQutrit
from vedic_layers import ShunyataSamskara, VilokanamOutput


class VedicTransformerLayer(nn.Module):
    """One complete Vedic Transformer layer."""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Attention
        self.attention = VedicAttention(dim, num_heads)
        self.norm1 = ShunyataSamskara(dim)
        
        # FFN
        self.ffn = PanchikaranaFFN(dim, dropout)
        self.activation = TrigunaQutrit()
        self.norm2 = ShunyataSamskara(dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(self.norm2(x))
        ffn_out = self.activation(ffn_out)
        x = x + self.dropout(ffn_out)
        
        return x


class VedicTransformer(nn.Module):
    """Complete Vedic Transformer for language modeling.
    
    Replaces every standard operation with its Vedic equivalent:
    - Matrika Nyasa embedding (not lookup table)
    - Kalachakra position encoding (cyclic recursive)
    - Trivritkarana QKV (single projection, derived by rotation)
    - Samanvaya attention (harmonic resonance, not dot product)
    - Nikhilam Sparsha softmax (base-complement normalization)
    - Soma-Yajna output (sacrificial distribution)
    - Panchikarana FFN (five-element expansion)
    - Triguna Qutrit activation (three-guna gating)
    - Shunyata Samskara norm (zero-centered)
    - Vilokanam output (observation-based sparse projection)
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        dropout: float = 0.1,
        max_seq: int = 32768
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        
        # Embedding
        self.embedding = MatrikaNyasaEmbedding(vocab_size, dim)
        self.position = KalachakraPositionEncoding(dim, max_seq)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            VedicTransformerLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output
        self.norm_final = ShunyataSamskara(dim)
        self.output = VilokanamOutput(dim, vocab_size)
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq] input token IDs
        Returns:
            logits: [batch, seq, vocab] output logits
        """
        B, S = token_ids.shape
        device = token_ids.device
        
        # Matrika Nyasa embedding
        x = self.embedding(token_ids)  # [B, S, D]
        
        # Kalachakra position encoding
        pos_enc = self.position(S, device)  # [S, D]
        x = x + pos_enc.unsqueeze(0)
        
        x = self.dropout(x)
        
        # Vedic Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final norm
        x = self.norm_final(x)
        
        # Vilokanam output projection
        logits = self.output(x)
        
        return logits
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
