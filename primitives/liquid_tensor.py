from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiquidStateTensor(nn.Module):
    """
    Stateful vector with learnable temporal decay τ (liquid memory).
    `data` is the wrapped state; τ is positive via softplus.
    """

    def __init__(self, dim: int, *, tau_init: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.data = nn.Parameter(torch.zeros(dim))
        t0 = max(float(tau_init), 1e-4)
        inv = torch.log(torch.expm1(torch.tensor(t0, dtype=torch.float32)))
        self._log_tau = nn.Parameter(inv.reshape(()))

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self._log_tau).squeeze() + 1e-5

    def assign_from(self, x: torch.Tensor) -> None:
        """Copy values from tensor matching `dim` (no grad through copy into param)."""
        with torch.no_grad():
            self.data.copy_(x.reshape(-1)[: self.dim])

    def forward(self) -> torch.Tensor:
        return self.data


def stack_liquid_states(states: list[LiquidStateTensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack `data` as (T, D) and per-step τ as (T,)."""
    if not states:
        raise ValueError("need at least one LiquidStateTensor")
    d = states[0].dim
    vals = torch.stack([s.data for s in states], dim=0)
    taus = torch.stack([s.tau.reshape(()) for s in states], dim=0)
    return vals, taus
