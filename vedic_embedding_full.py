"""
Complete Vedic Embedding & Position Suite — 18 Algorithms
"""

import torch
import torch.nn as nn
import math

class MatrikaNyasaEmbedding(nn.Module):
    """1. Matrika Nyasa — Seed syllable embedding"""
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.dim = dim
        self.category_phase = nn.Parameter(torch.linspace(0, 2*math.pi, 10))
        self.category_modulation = nn.Parameter(torch.randn(10, dim) * 0.02)
    
    def forward(self, token_ids):
        B, S = token_ids.shape
        categories = token_ids % 10
        positions = torch.arange(self.dim, device=token_ids.device).float()
        harmonics = positions / self.dim
        phase = self.category_phase[categories]
        embedding = torch.sin(phase.unsqueeze(-1) * harmonics.unsqueeze(0).unsqueeze(0))
        modulation = self.category_modulation[categories]
        return embedding * modulation * 0.1


class KalachakraPosition(nn.Module):
    """2. Kalachakra — Cyclic position encoding (Yuga cycles)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.yuga_ratios = nn.Parameter(torch.tensor([4., 3., 2., 1.]) / 10., requires_grad=False)
        self.base_angle = nn.Parameter(torch.tensor(0.0003))
    
    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device).float()
        cycle_pos = positions % 10.0
        yuga_idx = torch.zeros(seq_len, device=device).long()
        yuga_idx[cycle_pos >= 4.0] = 1
        yuga_idx[cycle_pos >= 7.0] = 2
        yuga_idx[cycle_pos >= 9.0] = 3
        ratios = self.yuga_ratios[yuga_idx]
        phase = torch.cumsum(ratios * self.base_angle, dim=0)
        freqs = torch.arange(self.dim, device=device).float() / self.dim
        return torch.sin(phase.unsqueeze(-1) * freqs.unsqueeze(0)) * 0.1


class NakshatraPosition(nn.Module):
    """3. Nakshatra — Lunar mansion position encoding (27 nakshatras)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.nakshatra_angles = nn.Parameter(torch.randn(27, dim // 27) * 0.02)
    
    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        nakshatra_idx = positions % 27
        base = torch.zeros(seq_len, self.dim, device=device)
        for n in range(27):
            mask = nakshatra_idx == n
            start = n * (self.dim // 27)
            end = (n + 1) * (self.dim // 27) if n < 26 else self.dim
            base[mask, start:end] = self.nakshatra_angles[n]
        return torch.sin(base * positions.unsqueeze(-1) * 0.01)


class RashiPosition(nn.Module):
    """4. Rashi — Zodiac position encoding (12 rashis)"""
    def __init__(self, dim):
        super().__init__()
        self.rashi_weights = nn.Parameter(torch.randn(12, dim) * 0.02)
    
    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        rashi_idx = positions % 12
        return self.rashi_weights[rashi_idx]


class TithiPosition(nn.Module):
    """5. Tithi — Lunar day position (30 tithis)"""
    def __init__(self, dim):
        super().__init__()
        self.tithi_embed = nn.Parameter(torch.randn(30, dim) * 0.02)
    
    def forward(self, seq_len, device):
        return self.tithi_embed[torch.arange(seq_len, device=device) % 30]


class YogaPosition(nn.Module):
    """6. Yoga — Planetary conjunction position (27 yogas)"""
    def __init__(self, dim):
        super().__init__()
        self.yoga_embed = nn.Parameter(torch.randn(27, dim) * 0.02)
    
    def forward(self, seq_len, device):
        return self.yoga_embed[torch.arange(seq_len, device=device) % 27]


class KaranaPosition(nn.Module):
    """7. Karana — Half-tithi position (11 karanas)"""
    def __init__(self, dim):
        super().__init__()
        self.karana_embed = nn.Parameter(torch.randn(11, dim) * 0.02)
    
    def forward(self, seq_len, device):
        return self.karana_embed[torch.arange(seq_len, device=device) % 11]


class SamvatsaraPosition(nn.Module):
    """8. Samvatsara — Year-cycle position (60 years)"""
    def __init__(self, dim):
        super().__init__()
        self.year_embed = nn.Parameter(torch.randn(60, dim) * 0.02)
    
    def forward(self, seq_len, device):
        return self.year_embed[torch.arange(seq_len, device=device) % 60]


class ManvantaraPosition(nn.Module):
    """9. Manvantara — Epoch position (14 Manus)"""
    def __init__(self, dim):
        super().__init__()
        self.manu_weights = nn.Parameter(torch.randn(14, dim) * 0.02)
    
    def forward(self, seq_len, device):
        return self.manu_weights[torch.arange(seq_len, device=device) % 14]


class KalpaPosition(nn.Module):
    """10. Kalpa — Cosmic day position (Brahma's day)"""
    def __init__(self, dim):
        super().__init__()
        self.kalpa_phase = nn.Parameter(torch.randn(dim) * 0.02)
    
    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device).float()
        return torch.sin(positions.unsqueeze(-1) * self.kalpa_phase.unsqueeze(0) * 1e-6)


class ChakraPosition(nn.Module):
    """11. Chakra — Energy center position (7 chakras)"""
    def __init__(self, dim):
        super().__init__()
        self.chakra_embed = nn.Parameter(torch.randn(7, dim) * 0.02)
    
    def forward(self, seq_len, device):
        return self.chakra_embed[torch.arange(seq_len, device=device) % 7]


class DhatuPosition(nn.Module):
    """12. Dhatu — Tissue-layer position (7 dhatus)"""
    def __init__(self, dim):
        super().__init__()
        self.dhatu_embed = nn.Parameter(torch.randn(7, dim) * 0.02)
    
    def forward(self, seq_len, device):
        return self.dhatu_embed[torch.arange(seq_len, device=device) % 7]


class KoshaPosition(nn.Module):
    """13. Kosha — Sheath position (5 koshas)"""
    def __init__(self, dim):
        super().__init__()
        self.kosha_embed = nn.Parameter(torch.randn(5, dim) * 0.02)
    
    def forward(self, seq_len, device):
        return self.kosha_embed[torch.arange(seq_len, device=device) % 5]


class VarnaPosition(nn.Module):
    """14. Varna — Phonetic class position (based on Sanskrit varnas)"""
    def __init__(self, dim):
        super().__init__()
        self.varna_embed = nn.Parameter(torch.randn(8, dim) * 0.02)
    
    def forward(self, seq_len, device):
        return self.varna_embed[torch.arange(seq_len, device=device) % 8]


class SvaraPosition(nn.Module):
    """15. Svara — Musical note position (7 svaras)"""
    def __init__(self, dim):
        super().__init__()
        frequencies = torch.tensor([240., 270., 300., 320., 360., 400., 450.]) / 240.0
        self.register_buffer('freqs', frequencies)
    
    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device).float()
        svara_idx = positions.long() % 7
        freq = self.freqs[svara_idx]
        dim_freqs = torch.arange(self.dim, device=device).float() / self.dim
        return torch.sin(positions.unsqueeze(-1) * freq.unsqueeze(-1) * dim_freqs.unsqueeze(0))


class TalaPosition(nn.Module):
    """16. Tala — Rhythm cycle position (various talas)"""
    def __init__(self, dim, tala_cycles=[8, 7, 6, 5, 10, 12, 16]):
        super().__init__()
        self.tala_cycles = tala_cycles
        self.tala_embed = nn.Parameter(torch.randn(len(tala_cycles), dim) * 0.02)
    
    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        result = torch.zeros(seq_len, self.dim, device=device)
        for i, cycle in enumerate(self.tala_cycles):
            phase = (positions % cycle).float() / cycle
            result += self.tala_embed[i].unsqueeze(0) * torch.sin(phase.unsqueeze(-1) * math.pi * 2)
        return result / len(self.tala_cycles)


class MandalaPosition(nn.Module):
    """17. Mandala — Sacred geometry position encoding"""
    def __init__(self, dim):
        super().__init__()
        self.radius = nn.Parameter(torch.linspace(0.1, 1.0, dim))
    
    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device).float()
        angles = positions * 0.61803398875  # Golden ratio
        x = self.radius.unsqueeze(0) * torch.cos(angles.unsqueeze(-1))
        y = self.radius.unsqueeze(0) * torch.sin(angles.unsqueeze(-1))
        return (x + y) * 0.1


class SriYantraPosition(nn.Module):
    """18. Sri Yantra — 9 interlocking triangles position encoding"""
    def __init__(self, dim):
        super().__init__()
        self.triangle_phases = nn.Parameter(torch.randn(9, dim) * 0.02)
    
    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device).float()
        result = torch.zeros(seq_len, self.dim, device=device)
        for t in range(9):
            angle = (t + 1) * math.pi * 2 / 9
            result += self.triangle_phases[t].unsqueeze(0) * torch.sin(positions.unsqueeze(-1) * angle)
        return result / 9


class CompleteVedicEmbedding(nn.Module):
    """Combined Vedic embedding with all 18 position algorithms"""
    def __init__(self, vocab_size, dim, max_seq=32768):
        super().__init__()
        self.matrika = MatrikaNyasaEmbedding(vocab_size, dim)
        self.kalachakra = KalachakraPosition(dim)
        self.nakshatra = NakshatraPosition(dim)
        self.rashi = RashiPosition(dim)
        self.tithi = TithiPosition(dim)
        self.yoga_pos = YogaPosition(dim)
        self.karana = KaranaPosition(dim)
        self.samvatsara = SamvatsaraPosition(dim)
        self.manvantara = ManvantaraPosition(dim)
        self.kalpa = KalpaPosition(dim)
        self.chakra = ChakraPosition(dim)
        self.dhatu = DhatuPosition(dim)
        self.kosha = KoshaPosition(dim)
        self.varna = VarnaPosition(dim)
        self.svara = SvaraPosition(dim)
        self.tala = TalaPosition(dim)
        self.mandala = MandalaPosition(dim)
        self.sri_yantra = SriYantraPosition(dim)
        
        # Learnable combination weights
        self.pos_weights = nn.Parameter(torch.ones(18) / 18)
    
    def forward(self, token_ids):
        B, S = token_ids.shape
        device = token_ids.device
        
        # Token embedding
        tok_emb = self.matrika(token_ids)
        
        # All position encodings
        positions = [
            self.kalachakra(S, device),
            self.nakshatra(S, device),
            self.rashi(S, device),
            self.tithi(S, device),
            self.yoga_pos(S, device),
            self.karana(S, device),
            self.samvatsara(S, device),
            self.manvantara(S, device),
            self.kalpa(S, device),
            self.chakra(S, device),
            self.dhatu(S, device),
            self.kosha(S, device),
            self.varna(S, device),
            self.svara(S, device),
            self.tala(S, device),
            self.mandala(S, device),
            self.sri_yantra(S, device),
        ]
        
        # Weighted combination
        weights = torch.softmax(self.pos_weights, dim=0)
        pos_emb = sum(w * p for w, p in zip(weights, positions))
        
        return tok_emb + pos_emb
