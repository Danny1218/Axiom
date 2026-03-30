from __future__ import annotations

from typing import List, Literal, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from axiom.engine.supernet import LatentSupernet

Verdict = Literal["integrate", "prune"]


class ShadowFitnessEvaluator:
    """Track mean localized loss per epoch; decide integrate vs prune after `epochs` samples."""

    def __init__(self, expert_name: str, epochs: int = 5) -> None:
        self.expert_name = expert_name
        self.epochs = epochs
        self.epoch_losses: List[float] = []

    def record_epoch_loss(self, loss: float) -> None:
        self.epoch_losses.append(float(loss))

    def verdict(self) -> Verdict:
        if len(self.epoch_losses) < self.epochs:
            raise ValueError(f"need {self.epochs} epoch losses, got {len(self.epoch_losses)}")
        if self.epoch_losses[-1] < self.epoch_losses[0]:
            return "integrate"
        return "prune"


def apply_shadow_verdict(supernet: LatentSupernet, expert_name: str, verdict: Verdict) -> None:
    if verdict == "integrate":
        supernet.integrate_shadow(expert_name)
    else:
        supernet.remask_expert(expert_name)


def localized_adapter_loss(adapter: nn.Module, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(adapter(h), target)


def run_shadow_training_epochs(
    adapter: nn.Module,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    *,
    epochs: int = 5,
    lr: float = 0.05,
) -> Tuple[List[float], Verdict]:
    """
    Train only `adapter` on MSE(adapter(h), target) for `epochs` full passes over `batches`.
    Returns per-epoch mean losses and integrate/prune from `ShadowFitnessEvaluator` rule.
    """
    opt = torch.optim.SGD(adapter.parameters(), lr=lr)
    losses: List[float] = []
    for _ in range(epochs):
        ep = 0.0
        m = max(len(batches), 1)
        for h, tgt in batches:
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(adapter(h), tgt)
            loss.backward()
            opt.step()
            ep += loss.item()
        losses.append(ep / m)
    verdict: Verdict = "integrate" if losses[-1] < losses[0] else "prune"
    return losses, verdict
