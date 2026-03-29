from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.fitness import ShadowFitnessEvaluator, Verdict, apply_shadow_verdict
from engine.meta_compiler import MetaCompiler
from engine.topology import ExecutionGraph


class EvolutionaryTrainer:
    """
    Train `ExecutionGraph` with Adam; optionally react to Sinkhorn mutation signals.
    After each epoch, records shadow localized losses and applies `ShadowFitnessEvaluator`
    when `shadow_fitness_epochs` samples are collected (integrate vs prune), then rebuilds
    the optimizer so state stays consistent with mask/shadow changes.
    """

    def __init__(
        self,
        graph: ExecutionGraph,
        *,
        lr: float = 1e-2,
        shadow_fitness_epochs: int = 5,
    ) -> None:
        self.graph = graph
        self.lr = float(lr)
        self.shadow_fitness_epochs = int(shadow_fitness_epochs)
        self.criterion = nn.MSELoss()
        self.shadow_evaluators: Dict[str, ShadowFitnessEvaluator] = {}
        self.optimizer = torch.optim.Adam(self.graph.parameters(), lr=self.lr)

    def rebuild_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(self.graph.parameters(), lr=self.lr)

    def train_epoch(
        self,
        loader,
        meta_compiler: Optional[MetaCompiler] = None,
    ) -> float:
        self.graph.train()
        total = 0.0
        count = 0
        last_xy: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        for x, y in loader:
            last_xy = (x, y)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.graph(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            total += float(loss.detach().item())
            count += 1
            if meta_compiler is not None:
                unmasked = meta_compiler.react_to_router_signals(self.graph.routers(), max_unmasks=1)
                if unmasked:
                    self.rebuild_optimizer()
        mean_loss = total / max(count, 1)
        if last_xy is not None:
            self._shadow_epoch_end(last_xy[0], last_xy[1])
        return mean_loss

    def _shadow_epoch_end(self, x: torch.Tensor, y: torch.Tensor) -> None:
        sn = self.graph.supernet
        self.graph.eval()
        with torch.no_grad():
            _ = self.graph(x)
        locs = self.graph.shadow_locals()
        for i, name in enumerate(sn.adapter_names):
            if not bool(sn.is_shadow[i].item()):
                continue
            loc = locs.get(name)
            loss_v = float(F.mse_loss(loc, y).item()) if loc is not None else 1.0
            if name not in self.shadow_evaluators:
                self.shadow_evaluators[name] = ShadowFitnessEvaluator(
                    name, epochs=self.shadow_fitness_epochs
                )
            self.shadow_evaluators[name].record_epoch_loss(loss_v)
        pending: List[Tuple[str, Verdict]] = []
        for name, ev in list(self.shadow_evaluators.items()):
            if len(ev.epoch_losses) >= ev.epochs:
                pending.append((name, ev.verdict()))
                ev.epoch_losses.clear()
        for name, verdict in pending:
            apply_shadow_verdict(sn, name, verdict)
        if pending:
            self.rebuild_optimizer()
        self.graph.train()
