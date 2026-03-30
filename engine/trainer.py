from __future__ import annotations

import importlib.util
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.fitness import ShadowFitnessEvaluator, Verdict, apply_shadow_verdict
from engine.meta_compiler import MetaCompiler
from engine.topology import ExecutionGraph


def _compile_step_fn(graph: ExecutionGraph) -> nn.Module:
    """Prefer inductor when available; fall back to aot_eager if codegen fails (e.g. no MSVC on Windows)."""
    import torch._dynamo.config as dynamo_config

    dynamo_config.capture_dynamic_output_shape_ops = True
    p = next(graph.parameters())
    dev, dt = p.device, p.dtype
    d = graph.supernet.dim
    trial = torch.randn(2, d, device=dev, dtype=dt, requires_grad=True)
    graph.train()
    order: List[str] = []
    if importlib.util.find_spec("torch._inductor") is not None:
        order.append("inductor")
    order.append("aot_eager")
    for backend in order:
        try:
            fn = torch.compile(graph, backend=backend, fullgraph=True)
            out, _, _ = fn(trial)
            out.sum().backward()
            trial.grad = None
            return fn
        except Exception:
            continue
    return graph


class EvolutionaryTrainer:
    """
    Train `ExecutionGraph` with Adam. Main MSE plus summed localized MSE on returned shadow dict
    so shadow adapters receive gradients while their contribution to `out` stays detached.
    Shadow fitness uses epoch averages of those localized losses (not a single no_grad pass).

    With ``target_col``, only that trunk column is supervised (``out[:, c:c+1]`` vs batch targets);
    other channels stay free for latent working memory. Default ``None`` keeps full-vector MSE
    (e.g. denoising loaders where ``y`` matches ``out`` shape).

    ``train_epoch(..., device=...)`` moves each batch ``x,y`` to that device when set (e.g. CUDA graph).
    """

    def __init__(
        self,
        graph: ExecutionGraph,
        *,
        lr: float = 1e-2,
        shadow_fitness_epochs: int = 5,
        compile_graph: bool = False,
        target_col: Optional[int] = None,
    ) -> None:
        self.graph = graph
        self.lr = float(lr)
        self.shadow_fitness_epochs = int(shadow_fitness_epochs)
        self.target_col = target_col
        self.criterion = nn.MSELoss()
        self.shadow_evaluators: Dict[str, ShadowFitnessEvaluator] = {}
        self.optimizer = torch.optim.Adam(self.graph.parameters(), lr=self.lr)
        if compile_graph:
            self.step_fn = _compile_step_fn(graph)
        else:
            self.step_fn = graph

    def train_epoch(
        self,
        loader,
        meta_compiler: Optional[MetaCompiler] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> float:
        self.graph.train()
        total = 0.0
        count = 0
        shadow_sum: Dict[str, float] = {}
        shadow_cnt: Dict[str, int] = {}
        for x, y in loader:
            if device is not None:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            out, locs, signals = self.step_fn(x)
            if self.target_col is not None:
                c = self.target_col
                loss = self.criterion(out[:, c : c + 1], y.view(-1, 1))
            else:
                loss = self.criterion(out, y)
            shadow_loss: Optional[torch.Tensor] = None
            sn = self.graph.supernet
            for name, loc in locs.items():
                i = sn._name_to_idx.get(name)
                if i is None or not bool(sn.is_shadow[i].item()):
                    continue
                if self.target_col is not None:
                    c = self.target_col
                    m = F.mse_loss(loc[:, c : c + 1], y.view(-1, 1))
                else:
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
                meta_compiler.react_to_signals(
                    signals,
                    self.graph.supernet,
                    max_unmasks=1,
                    block_thresholds=self.graph.block_mutation_thresholds(),
                )
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
