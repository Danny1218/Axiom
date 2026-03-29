from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.proj = nn.Linear(dim, num_experts, bias=True)

    def forward(self, x: torch.Tensor, expert_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (..., dim) -> P: (..., E) with row-sums 1 on active support and balanced column totals
        over the batch. Inactive experts (mask 0) receive zero mass.
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
        if not mask.any():
            return torch.zeros(B, self.num_experts, device=x.device, dtype=x.dtype)

        idx = mask.nonzero(as_tuple=False).squeeze(1)
        A = idx.numel()
        logits_a = logits[:, idx]
        K = torch.exp(logits_a - logits_a.max(dim=-1, keepdim=True).values.detach())
        row_target = torch.ones(B, device=x.device, dtype=x.dtype)
        col_target = torch.full((A,), float(B) / float(A), device=x.device, dtype=x.dtype)
        P_a = sinkhorn_balance(K, row_target=row_target, col_target=col_target, num_iters=self.num_iters)
        P = torch.zeros(B, self.num_experts, device=x.device, dtype=x.dtype)
        P[:, idx] = P_a
        return P.reshape(*lead, self.num_experts)
