from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MutationSignal:
    """Emitted when routing is too uniform (high epistemic uncertainty in expert choice)."""

    triggered: bool
    mean_entropy: float
    num_active_experts: int
    normalized_entropy: float  # mean_entropy / log(num_active), in [0, 1] when num_active > 1
