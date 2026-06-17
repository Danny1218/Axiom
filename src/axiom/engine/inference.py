from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, cast

import torch

from axiom.engine.strict import StrictInferenceError, validate_predict_inputs_strict


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


def _fill_abi_row_slice(
    row_t: torch.Tensor,
    col: int,
    width: int,
    dim: int,
    val: object,
) -> None:
    w = max(1, int(width))
    end = min(col + w, dim)
    if col >= dim or end <= col:
        return
    if isinstance(val, (list, tuple)):
        for i, x in enumerate(val):
            if col + i >= dim or i >= w:
                break
            row_t[col + i] = float(x)
    else:
        fv = float(cast(float, val))
        row_t[col:end] = fv


def _abi_rows_to_tensor(
    rows: List[Dict[str, float]],
    abi: Dict[str, int],
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    abi_widths: Optional[Dict[str, int]] = None,
    strict: bool = False,
) -> torch.Tensor:
    aw = dict(abi_widths or {})
    B = len(rows)
    if strict:
        for row in rows:
            validate_predict_inputs_strict(row, abi, abi_widths=aw)
    t = torch.zeros(B, dim, device=device, dtype=dtype)
    for b, row in enumerate(rows):
        for name, col in abi.items():
            if col >= dim:
                continue
            w = max(1, int(aw.get(name, 1)))
            if name not in row:
                continue
            _fill_abi_row_slice(t[b], col, w, dim, row[name])
    return t


AbiScalarOrVec = Union[float, List[float]]


def _abi_outputs_from_trunk_row(
    row: torch.Tensor,
    abi: Dict[str, int],
    abi_widths: Dict[str, int],
) -> Dict[str, AbiScalarOrVec]:
    D = int(row.shape[-1])
    out: Dict[str, AbiScalarOrVec] = {}
    for name, col in abi.items():
        w = max(1, int(abi_widths.get(name, 1)))
        if col >= D:
            continue
        end = min(col + w, D)
        span = end - col
        if span <= 0:
            continue
        if span == 1:
            out[name] = float(row[col].item())
        else:
            out[name] = [float(row[col + i].item()) for i in range(span)]
    return out


def _inputs_to_tensor(
    inputs: Dict[str, float],
    abi: Dict[str, int],
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    abi_widths: Optional[Dict[str, int]] = None,
    strict: bool = False,
) -> torch.Tensor:
    if dim < 1:
        raise ValueError("supernet dim must be positive")
    if not abi:
        if strict and inputs:
            unknown = list(inputs.keys())
            if unknown:
                raise StrictInferenceError(f"unknown input key(s): {', '.join(sorted(map(str, unknown)))}")
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
    return _abi_rows_to_tensor(
        [inputs], abi, dim, device=device, dtype=dtype, abi_widths=abi_widths, strict=strict
    )


def _batch_inputs_to_tensor(
    batch: List[Dict[str, float]],
    abi: Dict[str, int],
    dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    abi_widths: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    if not batch:
        raise ValueError("predict_batch requires a non-empty list")
    if not abi:
        return _legacy_rows_to_tensor(batch, dim, device=device, dtype=dtype)
    return _abi_rows_to_tensor(
        batch, abi, dim, device=device, dtype=dtype, abi_widths=abi_widths
    )


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
        abi_w = getattr(self.graph, "abi_widths", {}) or {}
        x = _inputs_to_tensor(
            inputs, abi, self.graph.supernet.dim, device=dev, dtype=dt, abi_widths=abi_w
        )
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
        abi_w = getattr(self.graph, "abi_widths", {}) or {}
        x = _batch_inputs_to_tensor(
            inputs, abi, self.graph.supernet.dim, device=dev, dtype=dt, abi_widths=abi_w
        )
        x = x.to(self.device)
        with torch.no_grad():
            out, _, _ = self.graph(x)
        return out

    def predict_dict(
        self,
        inputs: Dict[str, float],
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, AbiScalarOrVec]:
        out_tensor = self.predict(inputs, device=device)
        abi = getattr(self.graph, "abi", {}) or {}
        abi_w = getattr(self.graph, "abi_widths", {}) or {}
        return _abi_outputs_from_trunk_row(out_tensor[0], abi, abi_w)

    def predict_dict_batch(
        self,
        inputs: List[Dict[str, float]],
        device: Union[str, torch.device] = "cpu",
    ) -> List[Dict[str, AbiScalarOrVec]]:
        out_batch = self.predict_batch(inputs, device=device)
        abi = getattr(self.graph, "abi", {}) or {}
        abi_w = getattr(self.graph, "abi_widths", {}) or {}
        B = out_batch.shape[0]
        return [_abi_outputs_from_trunk_row(out_batch[b], abi, abi_w) for b in range(B)]

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
        abi_w = getattr(self.graph, "abi_widths", {}) or {}
        x = _inputs_to_tensor(
            inputs, abi, self.graph.supernet.dim, device=dev, dtype=dt, abi_widths=abi_w
        )
        x = x.to(self.device)
        with torch.no_grad():
            out, _, signals = self.graph(x)
        return out, signals
