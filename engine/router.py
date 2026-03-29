from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from engine.signals import MutationSignal


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
        self.last_mutation_signal: Optional[MutationSignal] = None

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
            self.last_mutation_signal = MutationSignal(False, 0.0, 0, 0.0)
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
        self.last_mutation_signal = self._mutation_from_routing(P_a, A)
        return P.reshape(*lead, self.num_experts)

    def _mutation_from_routing(self, P_active: torch.Tensor, num_active: int) -> MutationSignal:
        """High entropy ⇒ nearly uniform routing ⇒ trigger NAS mutation."""
        if num_active <= 1:
            return MutationSignal(False, 0.0, num_active, 0.0)
        p = P_active.clamp_min(1e-12)
        ent = -(p * p.log()).sum(dim=-1).mean()
        h_max = math.log(num_active)
        norm = (ent / h_max).clamp(0.0, 1.0).item()
        triggered = norm >= self.mutation_entropy_norm_threshold
        return MutationSignal(triggered, float(ent.item()), num_active, norm)
