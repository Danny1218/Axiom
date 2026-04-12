from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def sinkhorn_balance(
    K: torch.Tensor,
    *,
    row_target: torch.Tensor,
    col_target: torch.Tensor,
    num_iters: int,
    min_val: float = 1e-8,
) -> torch.Tensor:
    """Scale rows/cols of K so row sums = row_target, col sums = col_target (Sinkhorn–Knopp)."""
    B, E = K.shape
    assert row_target.shape == (B,) and col_target.shape == (E,)
    K = K.clamp_min(min_val)
    u = torch.ones(B, device=K.device, dtype=K.dtype)
    v = torch.ones(E, device=K.device, dtype=K.dtype)
    for _ in range(num_iters):
        u = row_target / (K @ v).clamp_min(min_val)
        v = col_target / (K.T @ u).clamp_min(min_val)
    return u.unsqueeze(1) * K * v.unsqueeze(0)


class SinkhornRouter(nn.Module):
    """Maps features to doubly-balanced routing weights over (unmasked) experts."""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        *,
        num_iters: int = 8,
        epsilon: float = 0.1,
        mutation_entropy_norm_threshold: float = 0.92,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.mutation_entropy_norm_threshold = mutation_entropy_norm_threshold
        self.proj = nn.Linear(dim, num_experts, bias=True)
        self.register_buffer(
            "_entropy_denominators",
            torch.tensor([math.log(max(a, 2)) for a in range(num_experts + 1)], dtype=torch.float64),
            persistent=False,
        )

    def _normalized_entropy_tensor(self, p_a: torch.Tensor) -> torch.Tensor:
        """Batch-mean routing entropy / log(max(A,2)), clamped to [0, 1]."""
        p = p_a.clamp_min(1e-12)
        ent = -(p * p.log()).sum(dim=-1).mean()
        den = self._entropy_denominators[p_a.shape[-1]].to(device=p_a.device, dtype=p_a.dtype)
        return (ent / den).clamp(0.0, 1.0)

    def forward(
        self, x: torch.Tensor, expert_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (..., dim) -> P: (..., E) with row-sums 1 on active support and balanced column totals
        over the batch. Inactive experts (mask 0) receive zero mass.
        Also returns normalized_entropy as a 0-dim tensor (requires_grad follows P_active).
        """
        *lead, d = x.shape
        flat = x.reshape(-1, d)
        B = flat.shape[0]
        logits = self.proj(flat) / self.epsilon
        if expert_mask is None:
            mask = torch.ones(self.num_experts, device=x.device, dtype=torch.bool)
        else:
            mask = expert_mask.to(device=x.device, dtype=torch.bool).reshape(-1)
            if mask.numel() != self.num_experts:
                raise ValueError("expert_mask length must match num_experts")
        # Dummy first expert when nothing is active so Sinkhorn always has A>=1 (no Python branch on mask.any()).
        idx0 = torch.zeros(1, device=mask.device, dtype=torch.long)
        dummy_active = torch.zeros(self.num_experts, device=mask.device, dtype=torch.bool).scatter(0, idx0, True)
        none_active = (~mask.any()).expand_as(mask)
        mask_eff = mask | (none_active & dummy_active)
        idx = mask_eff.nonzero(as_tuple=False).squeeze(1)
        A = idx.numel()
        logits_a = logits[:, idx]
        k = torch.exp(logits_a - logits_a.max(dim=-1, keepdim=True).values.detach())
        row_target = torch.ones(B, device=x.device, dtype=x.dtype)
        col_target = torch.full((A,), float(B) / float(A), device=x.device, dtype=x.dtype)
        p_a = sinkhorn_balance(k, row_target=row_target, col_target=col_target, num_iters=self.num_iters)
        p = torch.zeros(B, self.num_experts, device=x.device, dtype=x.dtype)
        p[:, idx] = p_a
        p = p * mask.to(dtype=p.dtype)
        norm_ent = self._normalized_entropy_tensor(p_a) * mask.any().to(dtype=p.dtype)
        return p.reshape(*lead, self.num_experts), norm_ent
