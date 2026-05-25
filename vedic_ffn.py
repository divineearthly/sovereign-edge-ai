import torch
import torch.nn as nn

class PanchikaranaFFN(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_elements = 5
        self.element_dim = dim // self.num_elements
        self.register_buffer('pancha_ratio', torch.tensor([
            [0.500, 0.125, 0.125, 0.125, 0.125],
            [0.125, 0.500, 0.125, 0.125, 0.125],
            [0.125, 0.125, 0.500, 0.125, 0.125],
            [0.125, 0.125, 0.125, 0.500, 0.125],
            [0.125, 0.125, 0.125, 0.125, 0.500],
        ]))
        self.W_element = nn.ParameterList([
            nn.Parameter(torch.randn(dim, self.element_dim + (dim % self.num_elements if e == self.num_elements - 1 else 0)) * 0.02)
            for e in range(self.num_elements)
        ])
        self.W_out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.threshold = 0.1

    def _element_strengths(self, x):
        shape = x.shape[:-1]
        x = x.reshape(-1, self.dim)
        N = x.shape[0]
        energy = torch.zeros(N, self.num_elements, device=x.device)
        for e in range(self.num_elements):
            start = e * self.element_dim
            end = self.dim if e == self.num_elements - 1 else (e + 1) * self.element_dim
            energy[:, e] = (x[:, start:end] ** 2).sum(dim=-1).sqrt()
        energy = energy / (energy.sum(dim=-1, keepdim=True) + 1e-8)
        return energy.view(*shape, self.num_elements)

    def forward(self, x):
        B, S, D = x.shape
        strengths = self._element_strengths(x)
        out = torch.zeros(B, S, D, device=x.device)
        for e in range(self.num_elements):
            active = strengths[..., e] > self.threshold
            if active.any():
                elem_out = torch.matmul(x, self.W_element[e])
                padded = torch.zeros(B, S, D, device=x.device)
                start = e * self.element_dim
                padded[..., start:start + elem_out.shape[-1]] = elem_out
                contrib = padded * self.pancha_ratio[e, e] * strengths[..., e:e+1]
                out = out + contrib * active.unsqueeze(-1).float()
        out = self.W_out(out)
        return self.dropout(out)

class TrigunaQutrit(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 0.01
    def forward(self, x):
        import math
        active = torch.abs(x) > self.threshold
        out = torch.zeros_like(x)
        if active.any():
            xa = x[active]
            out[active] = torch.sin(xa * math.pi / 2) * torch.cos(xa * math.pi / 4)
        return out
