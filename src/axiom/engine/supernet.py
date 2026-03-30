from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn


class TTLoRAAdapter(nn.Module):
    """Three-factor tensor-train style path: x → (x @ U) @ V @ W (rank-r bottleneck)."""

    def __init__(self, dim: int, rank: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.U = nn.Parameter(torch.empty(dim, rank))
        self.V = nn.Parameter(torch.empty(rank, rank))
        self.W = nn.Parameter(torch.empty(rank, dim))
        nn.init.normal_(self.U, std=0.02)
        nn.init.normal_(self.V, std=0.02)
        nn.init.normal_(self.W, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.U
        h = h @ self.V
        return h @ self.W


class LatentSupernet(nn.Module):
    """Frozen shared trunk + dictionary of TT-LoRA adapters; binary masks gate adds."""

    def __init__(
        self,
        dim: int,
        adapter_names: Iterable[str],
        *,
        rank: int = 4,
        trunk: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        names = tuple(adapter_names)
        self._name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(names)}
        if trunk is None:
            trunk = nn.Linear(dim, dim, bias=True)
        self.trunk = trunk
        for p in self.trunk.parameters():
            p.requires_grad = False
        self.adapters = nn.ModuleDict({n: TTLoRAAdapter(dim, rank) for n in names})
        self.register_buffer("adapter_mask", torch.zeros(len(names)))
        self.register_buffer("is_shadow", torch.zeros(len(names), dtype=torch.bool))

    @property
    def adapter_names(self) -> Tuple[str, ...]:
        return tuple(self.adapters.keys())

    def set_adapter_mask(self, name: str, value: float) -> None:
        """1.0 = apply adapter; 0.0 = inactive (default)."""
        self.adapter_mask[self._name_to_idx[name]] = float(value)

    def set_masks(self, masks: Mapping[str, float]) -> None:
        for k, v in masks.items():
            self.set_adapter_mask(k, v)

    def unmask_next_inactive(self, *, shadow: bool = True) -> Optional[str]:
        """Activate the first masked expert; optional sandbox shadow flag."""
        for i, name in enumerate(self.adapter_names):
            if self.adapter_mask[i] < 0.5:
                self.adapter_mask[i] = 1.0
                self.is_shadow[i] = shadow
                return name
        return None

    def remask_expert(self, name: str) -> None:
        i = self._name_to_idx[name]
        self.adapter_mask[i] = 0.0
        self.is_shadow[i] = False

    def integrate_shadow(self, name: str) -> None:
        i = self._name_to_idx[name]
        self.is_shadow[i] = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        for i, (name, adapter) in enumerate(self.adapters.items()):
            if self.adapter_mask[i] >= 0.5:
                delta = adapter(x)
                if self.is_shadow[i]:
                    delta = delta.detach()
                h = h + delta
        return h
