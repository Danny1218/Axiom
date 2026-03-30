from __future__ import annotations

from typing import Dict, List, Tuple, Union

import torch

from axiom.engine.topology import ExecutionGraph


def _legacy_rows_to_tensor(
    rows: List[Dict[str, float]],
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deprecated path when ``graph.abi`` is empty (very old bundles)."""
    if not rows:
        raise ValueError("predict_batch requires a non-empty list")
    union = sorted(set(k for row in rows for k in row.keys()))
    if len(union) > dim:
        raise ValueError(f"too many distinct input keys ({len(union)}) for trunk dim {dim}")
    B = len(rows)
    t = torch.zeros(B, dim, device=device, dtype=dtype)
    if len(union) == 1 and all(len(r) == 1 for r in rows):
        k0 = union[0]
        for b, row in enumerate(rows):
            t[b, :] = float(row[k0])
        return t
    for b, row in enumerate(rows):
        for j, k in enumerate(union):
            if k in row:
                t[b, j] = float(row[k])
    return t


def _abi_rows_to_tensor(
    rows: List[Dict[str, float]],
    abi: Dict[str, int],
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    B = len(rows)
    t = torch.zeros(B, dim, device=device, dtype=dtype)
    for b, row in enumerate(rows):
        for name, col in abi.items():
            if col < dim:
                t[b, col] = float(row.get(name, 0.0))
    return t


def _inputs_to_tensor(
    inputs: Dict[str, float],
    abi: Dict[str, int],
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim < 1:
        raise ValueError("supernet dim must be positive")
    if not abi:
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
    return _abi_rows_to_tensor([inputs], abi, dim, device=device, dtype=dtype)


def _batch_inputs_to_tensor(
    batch: List[Dict[str, float]],
    abi: Dict[str, int],
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not batch:
        raise ValueError("predict_batch requires a non-empty list")
    if not abi:
        return _legacy_rows_to_tensor(batch, dim, device=device, dtype=dtype)
    return _abi_rows_to_tensor(batch, abi, dim, device=device, dtype=dtype)


class AxiomRunner:
    """Run a deserialized `ExecutionGraph` under inference (no grad, eval mode)."""

    def __init__(self, graph: ExecutionGraph) -> None:
        self.graph = graph
        self.device = torch.device("cpu")

    def predict(
        self,
        inputs: Dict[str, float],
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        dev = torch.device(device) if isinstance(device, str) else device
        self.device = dev
        dt = torch.float32
        self.graph.to(dev)
        self.graph.eval()
        abi = getattr(self.graph, "abi", {}) or {}
        x = _inputs_to_tensor(inputs, abi, self.graph.supernet.dim, device=dev, dtype=dt)
        x = x.to(self.device)
        with torch.no_grad():
            out, _, _ = self.graph(x)
        return out

    def predict_batch(
        self,
        inputs: List[Dict[str, float]],
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        dev = torch.device(device) if isinstance(device, str) else device
        self.device = dev
        dt = torch.float32
        self.graph.to(dev)
        self.graph.eval()
        abi = getattr(self.graph, "abi", {}) or {}
        x = _batch_inputs_to_tensor(inputs, abi, self.graph.supernet.dim, device=dev, dtype=dt)
        x = x.to(self.device)
        with torch.no_grad():
            out, _, _ = self.graph(x)
        return out

    def predict_dict(
        self,
        inputs: Dict[str, float],
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, float]:
        out_tensor = self.predict(inputs, device=device)
        abi = getattr(self.graph, "abi", {}) or {}
        return {name: float(out_tensor[0, col].item()) for name, col in abi.items()}

    def predict_dict_batch(
        self,
        inputs: List[Dict[str, float]],
        device: Union[str, torch.device] = "cpu",
    ) -> List[Dict[str, float]]:
        out_batch = self.predict_batch(inputs, device=device)
        abi = getattr(self.graph, "abi", {}) or {}
        B = out_batch.shape[0]
        return [
            {name: float(out_batch[b, col].item()) for name, col in abi.items()}
            for b in range(B)
        ]

    def predict_with_signals(
        self,
        inputs: Dict[str, float],
        device: Union[str, torch.device] = "cpu",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        dev = torch.device(device) if isinstance(device, str) else device
        self.device = dev
        dt = torch.float32
        self.graph.to(dev)
        self.graph.eval()
        abi = getattr(self.graph, "abi", {}) or {}
        x = _inputs_to_tensor(inputs, abi, self.graph.supernet.dim, device=dev, dtype=dt)
        x = x.to(self.device)
        with torch.no_grad():
            out, _, signals = self.graph(x)
        return out, signals
