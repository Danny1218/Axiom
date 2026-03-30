from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch

from engine.topology import ExecutionGraph


def _inputs_to_tensor(
    inputs: Dict[str, float],
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim < 1:
        raise ValueError("supernet dim must be positive")
    if not inputs:
        return torch.zeros(1, dim, device=device, dtype=dtype)
    keys = sorted(inputs.keys())
    if len(keys) == 1:
        v = float(inputs[keys[0]])
        return torch.full((1, dim), v, device=device, dtype=dtype)
    if len(keys) > dim:
        raise ValueError(f"too many input keys ({len(keys)}) for trunk dim {dim}")
    row = torch.zeros(1, dim, device=device, dtype=dtype)
    for i, k in enumerate(keys):
        row[0, i] = float(inputs[k])
    return row


def _batch_inputs_to_tensor(
    batch: List[Dict[str, float]],
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not batch:
        raise ValueError("predict_batch requires a non-empty list")
    union = sorted(set(k for row in batch for k in row.keys()))
    if len(union) > dim:
        raise ValueError(f"too many distinct input keys ({len(union)}) for trunk dim {dim}")
    B = len(batch)
    t = torch.zeros(B, dim, device=device, dtype=dtype)
    if len(union) == 1 and all(len(r) == 1 for r in batch):
        k0 = union[0]
        for b, row in enumerate(batch):
            t[b, :] = float(row[k0])
        return t
    for b, row in enumerate(batch):
        for j, k in enumerate(union):
            if k in row:
                t[b, j] = float(row[k])
    return t


class AxiomRunner:
    """Run a deserialized `ExecutionGraph` under inference (no grad, eval mode)."""

    def __init__(self, graph: ExecutionGraph) -> None:
        self.graph = graph

    def predict(
        self,
        inputs: Dict[str, float],
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        dev = torch.device(device) if isinstance(device, str) else device
        dt = torch.float32
        self.graph.to(dev)
        self.graph.eval()
        x = _inputs_to_tensor(inputs, self.graph.supernet.dim, device=dev, dtype=dt)
        with torch.no_grad():
            out, _, _ = self.graph(x)
        return out

    def predict_batch(
        self,
        inputs: List[Dict[str, float]],
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        dev = torch.device(device) if isinstance(device, str) else device
        dt = torch.float32
        self.graph.to(dev)
        self.graph.eval()
        x = _batch_inputs_to_tensor(inputs, self.graph.supernet.dim, device=dev, dtype=dt)
        with torch.no_grad():
            out, _, _ = self.graph(x)
        return out

    def predict_with_signals(
        self,
        inputs: Dict[str, float],
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        dev = torch.device(device) if isinstance(device, str) else device
        dt = torch.float32
        self.graph.to(dev)
        self.graph.eval()
        x = _inputs_to_tensor(inputs, self.graph.supernet.dim, device=dev, dtype=dt)
        with torch.no_grad():
            out, _, signals = self.graph(x)
        return out, signals
