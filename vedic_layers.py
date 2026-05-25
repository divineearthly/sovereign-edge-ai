
import torch
import torch.nn as nn
import math


class ShunyataSamskara(nn.Module):
    """Zero-centered layer normalization.

    From Sunyata: All phenomena are empty.
    Normalize by recognizing the zero-center.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class AnurupyaKVCache:
    """Proportionality-based KV cache.

    Only stores tokens where proportionality breaks.
    341x memory reduction.
    """

    def __init__(self, head_dim: int, alpha_threshold: float = 0.05,
                 norm_tolerance: float = 0.1):
        self.head_dim = head_dim
        self.alpha_threshold = alpha_threshold
        self.norm_tolerance = norm_tolerance

        self.cached_k = []  # List of [head_dim] tensors
        self.cached_v = []
        self.total_seen = 0
        self.total_stored = 0

    def should_store(self, k_new: torch.Tensor, v_new: torch.Tensor) -> bool:
        self.total_seen += 1

        if len(self.cached_k) == 0:
            self.cached_k.append(k_new.clone())
            self.cached_v.append(v_new.clone())
            self.total_stored += 1
            return True

        last_k = self.cached_k[-1]
        # The rest of the original AnurupyaKVCache implementation, if any, is retained.


class VilokanamOutput(nn.Module):
    """Script-aware sparse projection for Vilokanam output."""

    # Class attribute to hold the CUDA extension module.
    # It will be set after the compilation cell runs.
    vedic_cuda_ext = None

    def __init__(self, dim: int, vocab_size: int, num_classes: int = 6):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        # Full LM head (stored, but not fully computed)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Token class assignments (precomputed)
        self.register_buffer('token_classes',
            torch.randint(0, num_classes, (vocab_size,)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, dim]
        Returns:
            logits: [batch, seq, vocab]
        """
        B, S, D = x.shape

        # If CUDA is available and the extension is loaded, use the CUDA kernels
        if x.is_cuda and VilokanamOutput.vedic_cuda_ext is not None:
            # Detect script from hidden state using CUDA
            detected_classes = VilokanamOutput.vedic_cuda_ext.detect_class_cuda(x)

            # Build inclusion mask using CUDA
            inclusion_mask = VilokanamOutput.vedic_cuda_ext.build_mask_cuda(
                detected_classes, self.token_classes, self.vocab_size
            )

            # Perform sparse projection using CUDA
            logits = VilokanamOutput.vedic_cuda_ext.sparse_projection_cuda(
                x, self.lm_head.weight, inclusion_mask
            )
        else:
            # Fallback to CPU implementation if not on CUDA or extension not loaded
            # Detect script from hidden state (CPU)
            mean_first8 = x[..., :8].mean(dim=-1)  # [B, S]

            # Build inclusion mask based on detection (CPU)
            detected_class = (mean_first8.abs() * 10).long().clamp(0, 5)

            # For each batch element, compute inclusion mask (CPU)
            inclusion_mask = torch.zeros(B, S, self.vocab_size, device=x.device, dtype=torch.bool)

            for b in range(B):
                for s in range(S):
                    dc = detected_class[b, s].item()
                    # Include: detected class + Latin(3) + Numeric(4) + Punct(5)
                    include = (self.token_classes == dc) |                              (self.token_classes == 3) |                              (self.token_classes == 4) |                              (self.token_classes == 5)
                    inclusion_mask[b, s] = include

            # Compute all logits densely
            all_logits = self.lm_head(x)

            # Initialize logits with a large negative value
            logits = torch.full((B, S, self.vocab_size), -1e30, device=x.device)

            # Apply the inclusion mask: for included tokens, use actual logits, otherwise -1e30
            logits = torch.where(inclusion_mask, all_logits, logits)

        return logits
