"""
Vedic Feed-Forward Network
- Panchikarana: Five-element expansion (not arbitrary 4x)
- Triguna Qutrit: Three-guna activation
- Pralaya: Sparse dissolution back to unity
"""

import torch
import torch.nn as nn
import math

class PanchikaranaFFN(nn.Module):
    """Fivefold FFN expansion.
    
    From Chandogya Upanishad: 5 elements, each = 1/2 self + 1/8 others.
    Only computes elements with significant strength.
    """
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_elements = 5  # Pancha Mahabhuta
        self.element_dim = dim // self.num_elements
        
        # Panchikarana ratio matrix
        self.register_buffer('pancha_ratio', torch.tensor([
            [0.500, 0.125, 0.125, 0.125, 0.125],
            [0.125, 0.500, 0.125, 0.125, 0.125],
            [0.125, 0.125, 0.500, 0.125, 0.125],
            [0.125, 0.125, 0.125, 0.500, 0.125],
            [0.125, 0.125, 0.125, 0.125, 0.500],
        ]))
        
        # Element-specific weights
        self.W_element = nn.ParameterList([
            nn.Parameter(torch.randn(dim, self.element_dim) * 0.02)
            for _ in range(self.num_elements)
        ])
        
        # Second projection
        self.W_out = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.threshold = 0.1
    
    def _element_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """Detect which elements are active.
        
        Args:
            x: [..., dim]
        Returns:
            strengths: [..., num_elements]
        """
        shape = x.shape[:-1]
        x = x.view(-1, self.dim)
        
        # Partition into 5 zones, compute energy per zone
        zones = x.view(-1, self.num_elements, self.element_dim)
        energy = (zones ** 2).sum(dim=-1).sqrt()  # [N, 5]
        
        # Normalize
        energy = energy / (energy.sum(dim=-1, keepdim=True) + 1e-8)
        
        return energy.view(*shape, self.num_elements)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, dim]
        Returns:
            out: [batch, seq, dim]
        """
        B, S, D = x.shape
        strengths = self._element_strengths(x)  # [B, S, 5]
        
        out = torch.zeros(B, S, D, device=x.device)
        
        for e in range(self.num_elements):
            # Only compute elements above threshold
            active = strengths[..., e] > self.threshold  # [B, S]
            
            if active.any():
                # Element-specific projection
                elem_out = torch.matmul(x, self.W_element[e])  # [B, S, D/5]
                elem_out = elem_out.repeat_interleave(
                    self.num_elements, dim=-1)[..., :D]  # Expand to D
                
                # Apply Panchikarana ratio
                self_ratio = self.pancha_ratio[e, e]  # 0.5 for self
                
                # Add weighted contribution
                mask = active.unsqueeze(-1).float()
                contrib = elem_out * self_ratio * strengths[..., e:e+1]
                out = out + contrib * mask
        
        # Pralaya: Only non-zero contributions pass through
        out = self.W_out(out)
        out = self.dropout(out)
        
        return out


class TrigunaQutrit(nn.Module):
    """Three-guna activation replacing GELU.
    
    From Samkhya: Sattva (keep), Rajas (amplify), Tamas (suppress).
    """
    
    def __init__(self):
        super().__init__()
        self.threshold = 0.01
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tamas: values below threshold → suppress to zero
        # Rajas/Sattva: values above threshold → sin·cos activation
        active = torch.abs(x) > self.threshold
        out = torch.zeros_like(x)
        if active.any():
            x_active = x[active]
            out[active] = torch.sin(x_active * math.pi / 2) * torch.cos(
                x_active * math.pi / 4)
        return out
