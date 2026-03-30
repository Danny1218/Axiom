from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from axiom.primitives.liquid_tensor import LiquidStateTensor, stack_liquid_states


def _rbf_basis(fused_norm: torch.Tensor, num_basis: int) -> torch.Tensor:
    """Per-channel Gaussian RBFs on [0, 1]: fused_norm (B, D) → phi (B, D, K)."""
    scalar_x = torch.sigmoid(fused_norm)
    if num_basis < 2:
        return torch.ones(
            *scalar_x.shape,
            1,
            device=fused_norm.device,
            dtype=fused_norm.dtype,
        )
    centers = torch.linspace(0.0, 1.0, num_basis, device=fused_norm.device, dtype=fused_norm.dtype).view(
        1, 1, -1
    )
    width = 1.0 / max(num_basis - 1, 1)
    diff = (scalar_x.unsqueeze(-1) - centers) / width
    return torch.exp(-(diff**2))


class LiquidKANNode(nn.Module):
    """
    Liquid memory + KAN: per-dimension Gaussian RBF bases; sequence input fused into the basis path.
    `forward(h)` runs a fixed-depth recurrence (compile-time unroll). `forward_sequence` consumes
    a list of `LiquidStateTensor` (per-timestep τ and payload).
    """

    def __init__(
        self,
        dim: int,
        *,
        num_basis: int = 8,
        max_unroll: int = 8,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_basis = num_basis
        self.max_unroll = max_unroll
        self.fuse_proj = nn.Linear(dim * 2, dim, bias=False)
        self.w_gate = nn.Linear(dim * 3, 1, bias=True)
        self.coeffs = nn.Parameter(torch.randn(dim, num_basis) / math.sqrt(float(num_basis)))

    def _kan_update(
        self,
        h_cur: torch.Tensor,
        x_t: torch.Tensor,
        h0: torch.Tensor,
        t_norm: torch.Tensor,
    ) -> torch.Tensor:
        del t_norm  # reserved for future time-conditioning; kept for call-site stability
        h2 = h_cur.reshape(h_cur.shape[0], -1)
        x2 = x_t.reshape(h_cur.shape[0], -1)
        h02 = h0.reshape(h_cur.shape[0], -1)
        fused = self.fuse_proj(torch.cat([h2, x2], dim=-1))
        fused_norm = F.layer_norm(fused, (self.dim,))
        phi = _rbf_basis(fused_norm, self.num_basis)
        out = (phi * self.coeffs.unsqueeze(0)).sum(dim=-1)
        gate = torch.sigmoid(self.w_gate(torch.cat([h2, x2, h02], dim=-1)))
        return (out * gate).reshape(h2.shape[0], -1)

    def _liquid_mix(self, h: torch.Tensor, proposal: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        h2 = h.reshape(h.shape[0], -1)
        p2 = proposal.reshape(h2.shape[0], -1)
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor(tau, device=h2.device, dtype=h2.dtype)
        tau = tau.reshape(()).clamp(min=1e-5)
        alpha = torch.exp(-1.0 / tau)
        a = alpha.expand(h2.shape[0]).reshape(-1, 1)
        return a * h2 + (1.0 - a) * p2

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Recurrent liquid-KAN steps (τ = 1 per step) for graph wiring from trunk features."""
        h0 = h.reshape(h.shape[0], -1)
        h_cur = h0
        T = max(self.max_unroll, 1)
        tau = torch.tensor(1.0, device=h.device, dtype=h.dtype)
        x_dummy = torch.zeros_like(h_cur)
        for t in range(T):
            tn = torch.full((h_cur.size(0), 1), t / max(T - 1, 1), device=h.device, dtype=h.dtype)
            prop = self._kan_update(h_cur, x_dummy, h0, tn)
            h_cur = self._liquid_mix(h_cur, prop, tau)
        return h_cur

    def forward_sequence(self, states: Sequence[LiquidStateTensor], h_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sequence of liquid states: per-step τ from each `LiquidStateTensor`, KAN proposal + mix."""
        vals, taus = stack_liquid_states(list(states))
        T, D = vals.shape
        if D != self.dim:
            raise ValueError(f"state dim {D} != node dim {self.dim}")
        h_cur = vals[0].unsqueeze(0)
        if h_init is not None:
            h_cur = h_init.reshape(1, -1)
        h0 = h_cur.clone()
        for t in range(T):
            tn = torch.full((h_cur.size(0), 1), t / max(T - 1, 1), device=h_cur.device, dtype=h_cur.dtype)
            x_t = vals[t].unsqueeze(0)
            prop = self._kan_update(h_cur, x_t, h0, tn)
            h_cur = self._liquid_mix(h_cur, prop, taus[t].reshape(()))
        return h_cur.squeeze(0)

    def forward_sequence_tensors(
        self,
        seq: torch.Tensor,
        *,
        taus: Optional[torch.Tensor] = None,
        h_init: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dense sequence (B, T, D) from IR snapshots; returns (B, D). Same recurrence as `forward_sequence`."""
        if seq.dim() == 2:
            seq = seq.unsqueeze(0)
        B, T, D = seq.shape
        if T < 1:
            raise ValueError("forward_sequence_tensors needs T>=1")
        if D != self.dim:
            raise ValueError(f"state dim {D} != node dim {self.dim}")
        if mask is not None and mask.shape != (B, T):
            raise ValueError(f"mask shape {tuple(mask.shape)} != (B, T) = ({B}, {T})")
        if taus is None:
            tau_vec = torch.ones(T, device=seq.device, dtype=seq.dtype)
        else:
            tau_vec = taus.reshape(-1)
            if tau_vec.shape[0] != T:
                raise ValueError(f"taus length {tau_vec.shape[0]} != T {T}")
        h_cur = seq[:, 0, :]
        if h_init is not None:
            h_cur = h_init.reshape(B, D)
        h0 = h_cur.clone()
        for t in range(T):
            tn = torch.full((B, 1), t / max(T - 1, 1), device=seq.device, dtype=seq.dtype)
            x_t = seq[:, t, :]
            h_prev = h_cur
            prop = self._kan_update(h_prev, x_t, h0, tn)
            h_next = self._liquid_mix(h_prev, prop, tau_vec[t].reshape(()))
            if mask is not None:
                m_t = mask[:, t].unsqueeze(-1)
                h_cur = torch.where(m_t, h_next, h_prev)
            else:
                h_cur = h_next
        return h_cur
