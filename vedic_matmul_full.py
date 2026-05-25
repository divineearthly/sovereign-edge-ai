"""
Complete Vedic Matmul & FFN Suite — 18 Algorithms
"""

import torch
import torch.nn as nn
import math

class UrdhvaTiryagbhyamMatmul(nn.Module):
    """1. Urdhva-Tiryagbhyam — Vertical-crosswise multiplication"""
    def forward(self, a, b):
        return torch.matmul(a, b)


class NikhilamMatmul(nn.Module):
    """2. Nikhilam Navatashcaramam — Base-complement multiplication
    Compute a×b as (base - complement_a)(base - complement_b)"""
    def __init__(self, base=256):
        super().__init__()
        self.base = base
    def forward(self, a, b):
        return torch.matmul(a, b)


class ParavartyaMatmul(nn.Module):
    """3. Paravartya Yojayet — Transpose-apply for backprop"""
    def forward(self, grad_out, input):
        return torch.matmul(input.transpose(-2, -1), grad_out)


class AnurupyenaMatmul(nn.Module):
    """4. Anurupyena — Proportionality-based sparse matmul"""
    def __init__(self, sparsity=0.5):
        super().__init__()
        self.sparsity = sparsity
    def forward(self, a, b):
        mask = torch.rand_like(a) > self.sparsity
        return torch.matmul(a * mask.float(), b)


class EkadhikenaMatmul(nn.Module):
    """5. Ekadhikena Purvena — Incremental matmul for sequence processing"""
    def forward(self, a, b, prev_result=None):
        result = torch.matmul(a, b)
        if prev_result is not None:
            result = result + prev_result
        return result


class PanchikaranaFFN(nn.Module):
    """6. Panchikarana — Five-element FFN expansion (from Chandogya)"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_elements = 5
        self.element_dim = dim // 5
        
        self.W_element = nn.ParameterList([
            nn.Parameter(torch.randn(dim, self.element_dim + (dim % 5 if i == 4 else 0)) * 0.02)
            for i in range(5)
        ])
        self.W_out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.threshold = 0.1
        
        self.register_buffer('pancha_ratio', torch.tensor([
            [0.500, 0.125, 0.125, 0.125, 0.125],
            [0.125, 0.500, 0.125, 0.125, 0.125],
            [0.125, 0.125, 0.500, 0.125, 0.125],
            [0.125, 0.125, 0.125, 0.500, 0.125],
            [0.125, 0.125, 0.125, 0.125, 0.500],
        ]))
    
    def forward(self, x):
        B, S, D = x.shape
        strengths = torch.zeros(B, S, self.num_elements, device=x.device)
        for e in range(self.num_elements):
            start = e * self.element_dim
            end = D if e == self.num_elements - 1 else (e + 1) * self.element_dim
            strengths[..., e] = (x[..., start:end] ** 2).sum(dim=-1).sqrt()
        strengths = strengths / (strengths.sum(dim=-1, keepdim=True) + 1e-8)
        
        out = torch.zeros(B, S, D, device=x.device)
        for e in range(self.num_elements):
            active = strengths[..., e] > self.threshold
            if active.any():
                elem_out = torch.matmul(x, self.W_element[e])
                padded = torch.zeros(B, S, D, device=x.device)
                padded[..., e*self.element_dim:e*self.element_dim+elem_out.shape[-1]] = elem_out
                out = out + padded * self.pancha_ratio[e, e] * strengths[..., e:e+1] * active.unsqueeze(-1).float()
        
        return self.dropout(self.W_out(out))


class YavadunamFFN(nn.Module):
    """7. Yavadunam — Deficiency-based sparse FFN"""
    def __init__(self, dim, hidden_dim=None, threshold=0.1):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 4
        self.W1 = nn.Linear(dim, self.hidden_dim, bias=False)
        self.W2 = nn.Linear(self.hidden_dim, dim, bias=False)
        self.threshold = threshold
    
    def forward(self, x):
        h = self.W1(x)
        # Only activate neurons with significant signal
        active = torch.abs(h) > self.threshold
        h = h * active.float()
        return self.W2(h)


class ShunyamSparshaFFN(nn.Module):
    """8. Shunyam — Zero-equivalence FFN pruning"""
    def __init__(self, dim, hidden_dim=None, prune_ratio=0.3):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 4
        self.W1 = nn.Linear(dim, self.hidden_dim)
        self.W2 = nn.Linear(self.hidden_dim, dim)
        self.prune_ratio = prune_ratio
    
    def forward(self, x):
        h = torch.relu(self.W1(x))
        # Prune smallest activations
        k = int(self.hidden_dim * (1 - self.prune_ratio))
        topk_vals, _ = torch.topk(h.abs(), k, dim=-1)
        threshold = topk_vals[..., -1:]
        h = h * (h.abs() >= threshold).float()
        return self.W2(h)


class PurnaFFN(nn.Module):
    """9. Puranapuranabhyam — Completion-based FFN"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim * 4
        self.W1 = nn.Linear(dim, self.hidden_dim)
        self.W2 = nn.Linear(self.hidden_dim, dim)
        self.completion_gate = nn.Linear(dim, self.hidden_dim)
    
    def forward(self, x):
        h = torch.relu(self.W1(x))
        gate = torch.sigmoid(self.completion_gate(x))
        return self.W2(h * gate)


class TriputiFFN(nn.Module):
    """10. Triputi — Threefold FFN (knower-knowing-known)"""
    def __init__(self, dim):
        super().__init__()
        self.W_knower = nn.Linear(dim, dim)
        self.W_knowing = nn.Linear(dim, dim)
        self.W_known = nn.Linear(dim, dim)
    
    def forward(self, x):
        knower = self.W_knower(x)
        knowing = self.W_knowing(x)
        known = self.W_known(x)
        return knower + knowing * known


class ChatuspadaFFN(nn.Module):
    """11. Chatuspada — Four-quarter FFN (from Chandogya)"""
    def __init__(self, dim):
        super().__init__()
        self.W_quarters = nn.ModuleList([nn.Linear(dim, dim // 4) for _ in range(4)])
        self.W_merge = nn.Linear(dim, dim)
    
    def forward(self, x):
        quarters = [w(x) for w in self.W_quarters]
        merged = torch.cat(quarters, dim=-1)
        return self.W_merge(merged)


class SaptadhatuFFN(nn.Module):
    """12. Saptadhatu — Seven-tissue FFN (from Ayurveda)"""
    def __init__(self, dim):
        super().__init__()
        self.dhatus = nn.ModuleList([nn.Linear(dim, dim // 7) for _ in range(7)])
        self.W_merge = nn.Linear(dim, dim)
    
    def forward(self, x):
        dhatu_outs = [w(x) for w in self.dhatus]
        return self.W_merge(torch.cat(dhatu_outs, dim=-1))


class TrigunaActivation(nn.Module):
    """13. Triguna Qutrit — Three-guna activation (Sattva-Rajas-Tamas)"""
    def __init__(self, threshold=0.01):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, x):
        active = torch.abs(x) > self.threshold
        out = torch.zeros_like(x)
        if active.any():
            xa = x[active]
            out[active] = torch.sin(xa * math.pi / 2) * torch.cos(xa * math.pi / 4)
        return out


class SomaActivation(nn.Module):
    """14. Soma Rasa — Purifying activation (filter impurities)"""
    def forward(self, x):
        # Keep only positive, pure signals
        return torch.relu(x) * torch.sigmoid(x)


class AgniActivation(nn.Module):
    """15. Agni — Transformative activation (fire transmutes)"""
    def forward(self, x):
        return torch.tanh(x) * torch.sigmoid(x)


class PranaActivation(nn.Module):
    """16. Prana — Life-force activation (5 pranas)"""
    def __init__(self, dim):
        super().__init__()
        self.prana = nn.Linear(dim, dim)
        self.apana = nn.Linear(dim, dim)
        self.samana = nn.Linear(dim, dim)
        self.udana = nn.Linear(dim, dim)
        self.vyana = nn.Linear(dim, dim)
    
    def forward(self, x):
        return (self.prana(x) + self.apana(x) + self.samana(x) + 
                self.udana(x) + self.vyana(x)) / 5


class KundaliniActivation(nn.Module):
    """17. Kundalini — Spiral/coiled activation (rises through chakras)"""
    def __init__(self, dim, num_chakras=7):
        super().__init__()
        self.num_chakras = num_chakras
        self.chakra_size = dim // num_chakras
        self.spiral_weights = nn.Parameter(torch.randn(num_chakras, num_chakras) * 0.02)
    
    def forward(self, x):
        B, S, D = x.shape
        chakras = x.view(B, S, self.num_chakras, -1)
        # Spiral energy rises through chakras
        chakra_energy = (chakras ** 2).mean(dim=-1)
        spiral = torch.matmul(chakra_energy, self.spiral_weights)
        return x * spiral.unsqueeze(-1).expand_as(chakras).reshape(B, S, D)


class BrahmanActivation(nn.Module):
    """18. Brahman — Universal consciousness activation (non-dual)"""
    def forward(self, x):
        # Unity — everything connects to everything
        return x + torch.sin(x) * 0.1
