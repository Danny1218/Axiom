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
    Train `ExecutionGraph` with Adam. Main MSE plus summed localized MSE on returned shadow dict
    so shadow adapters receive gradients while their contribution to `out` stays detached.
    Shadow fitness uses epoch averages of those localized losses (not a single no_grad pass).
    """

    def __init__(
        self,
        graph: ExecutionGraph,
        *,
        lr: float = 1e-2,
        shadow_fitness_epochs: int = 5,
        compile_graph: bool = False,
    ) -> None:
        self.graph = graph
        self.lr = float(lr)
        self.shadow_fitness_epochs = int(shadow_fitness_epochs)
        self.criterion = nn.MSELoss()
        self.shadow_evaluators: Dict[str, ShadowFitnessEvaluator] = {}
        self.optimizer = torch.optim.Adam(self.graph.parameters(), lr=self.lr)
        if compile_graph:
            # SinkhornRouter uses `.item()` for mutation metadata; allow scalar capture so Dynamo does not diverge on restarts.
            import torch._dynamo.config as dynamo_config

            dynamo_config.capture_scalar_outputs = True
            # aot_eager: no Inductor (portable); use default backend on Linux+GPU for Triton fusion.
            self.step_fn: nn.Module = torch.compile(graph, backend="aot_eager")
        else:
            self.step_fn = graph

    def train_epoch(
        self,
        loader,
        meta_compiler: Optional[MetaCompiler] = None,
    ) -> float:
        self.graph.train()
        total = 0.0
        count = 0
        shadow_sum: Dict[str, float] = {}
        shadow_cnt: Dict[str, int] = {}
        for x, y in loader:
            self.optimizer.zero_grad(set_to_none=True)
            out, locs = self.step_fn(x)
            loss = self.criterion(out, y)
            shadow_loss: Optional[torch.Tensor] = None
            for name, loc in locs.items():
                m = F.mse_loss(loc, y)
                shadow_loss = m if shadow_loss is None else shadow_loss + m
                key = name
                shadow_sum[key] = shadow_sum.get(key, 0.0) + float(m.detach().item())
                shadow_cnt[key] = shadow_cnt.get(key, 0) + 1
            if shadow_loss is None:
                shadow_loss = torch.zeros((), device=out.device, dtype=out.dtype)
            total_loss = loss + shadow_loss
            total_loss.backward()
            self.optimizer.step()
            total += float(loss.detach().item())
            count += 1
            if meta_compiler is not None:
                meta_compiler.react_to_router_signals(self.graph.routers(), max_unmasks=1)
        mean_loss = total / max(count, 1)
        epoch_means = {n: shadow_sum[n] / shadow_cnt[n] for n in shadow_sum if shadow_cnt.get(n, 0) > 0}
        self._shadow_epoch_end(epoch_means)
        return mean_loss

    def _shadow_epoch_end(self, epoch_shadow_mse: Dict[str, float]) -> None:
        sn = self.graph.supernet
        for i, name in enumerate(sn.adapter_names):
            if not bool(sn.is_shadow[i].item()):
                continue
            if name not in epoch_shadow_mse:
                continue
            loss_v = epoch_shadow_mse[name]
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
        self.graph.train()
