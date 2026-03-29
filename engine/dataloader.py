from __future__ import annotations

from typing import Iterator, Optional, Tuple

import torch


def sequential_to_features(sequence_1d: torch.Tensor, feature_dim: int) -> torch.Tensor:
    """Map (T,) float sequence to (T, D) by broadcasting the scalar channel."""
    if sequence_1d.dim() != 1:
        raise ValueError("sequence must be 1-D (T,)")
    return sequence_1d.unsqueeze(-1).expand(-1, int(feature_dim)).contiguous()


class LiquidSequenceLoader:
    """
    Yields (x, y) batches: x is trunk-shaped (B, D) with injected Gaussian noise
    (`baseline_var`) to excite stochastic routing / mutation paths; y defaults to the
    clean projection (identity denoising target).
    """

    def __init__(
        self,
        sequence_1d: torch.Tensor,
        feature_dim: int,
        batch_size: int,
        *,
        baseline_var: float = 0.05,
        device: Optional[torch.device] = None,
        shuffle: bool = True,
    ) -> None:
        self.features = sequential_to_features(sequence_1d.float(), feature_dim)
        self.targets = self.features.clone()
        self.batch_size = max(1, int(batch_size))
        self.baseline_var = float(baseline_var)
        self.device = device
        self.shuffle = shuffle

    def __len__(self) -> int:
        return max(1, (self.features.size(0) + self.batch_size - 1) // self.batch_size)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        n = self.features.size(0)
        idx = torch.randperm(n) if self.shuffle else torch.arange(n)
        for start in range(0, n, self.batch_size):
            j = idx[start : start + self.batch_size]
            noise = self.baseline_var * torch.randn_like(self.features[j])
            x = self.features[j] + noise
            y = self.targets[j]
            if self.device is not None:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
            yield x, y
