"""High-level Python API: load ``.axb`` bundles and predict with dicts / batches / DataFrames.

``AxiomModel`` is the primary programmatic contract for bundles. Higher-level agents (e.g. a
semantic copilot backed by an external LLM) should call ``predict`` / ``explain`` / ``export_report``
here rather than duplicating trunk/ABI logic—see ``plan.md`` § Next target (not implemented).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from axiom.compiler.deserializer import load_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _abi_outputs_from_trunk_row, _inputs_to_tensor
from axiom.tools.html_exporter import export_html_report


def _env_tensor_to_python(t: torch.Tensor) -> Union[float, List[float]]:
    """Map batch-1 env tensor to float or list of floats (first row only)."""
    x = t.detach().cpu()
    if x.dim() >= 1:
        x = x[0]
    if x.dim() == 0:
        return float(x.item())
    flat = x.flatten().tolist()
    if len(flat) == 1:
        return float(flat[0])
    return [float(v) for v in flat]


def _trunk_dim_from_block_abi(block: InterpretedBlock) -> int:
    abi, aw = block.abi, getattr(block, "abi_widths", {}) or {}
    return max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=16)


def load(
    bundle_path: str | Path,
    custom_neural_registry: Optional[Dict[str, nn.Module]] = None,
) -> AxiomModel:
    """Load a ``.axb`` bundle. Pass ``custom_neural_registry`` if training used non-default ``neural()`` nets."""
    return AxiomModel(load_bundle(bundle_path, custom_neural_registry=custom_neural_registry))


class AxiomModel:
    """Scikit-learn-style wrapper around ``InterpretedBlock`` (dict in → dict out)."""

    def __init__(self, block: InterpretedBlock) -> None:
        self.block = block

    def predict(self, data: Any) -> Any:
        block = self.block
        block.eval()
        dim = _trunk_dim_from_block_abi(block)
        dev = torch.device("cpu")
        dt = torch.float32
        abi = block.abi
        aw = dict(getattr(block, "abi_widths", {}) or {})

        if type(data).__name__ == "DataFrame":
            data = data.to_dict(orient="records")

        if isinstance(data, list):
            if not data:
                raise ValueError("predict requires a non-empty list of dicts")
            rows: List[torch.Tensor] = []
            for row in data:
                if not isinstance(row, dict):
                    raise TypeError("batch rows must be dicts")
                rows.append(
                    _inputs_to_tensor(row, abi, dim, device=dev, dtype=dt, abi_widths=aw)
                )
            batched_h = torch.cat(rows, dim=0)
            with torch.no_grad():
                out_trunk = block(batched_h)
            return [
                _abi_outputs_from_trunk_row(out_trunk[i], abi, aw)
                for i in range(out_trunk.shape[0])
            ]

        if not isinstance(data, dict):
            raise TypeError("data must be a dict, list of dicts, or pandas.DataFrame")
        h = _inputs_to_tensor(data, abi, dim, device=dev, dtype=dt, abi_widths=aw)
        with torch.no_grad():
            out_trunk = block(h)
        return _abi_outputs_from_trunk_row(out_trunk[0], abi, aw)

    def explain(self, data: dict) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of symbolic variables after one forward (batch size 1)."""
        if not isinstance(data, dict):
            raise TypeError("explain expects a single dict (one row of features)")
        block = self.block
        block.eval()
        dim = _trunk_dim_from_block_abi(block)
        dev = torch.device("cpu")
        dt = torch.float32
        abi = block.abi
        aw = dict(getattr(block, "abi_widths", {}) or {})
        h = _inputs_to_tensor(data, abi, dim, device=dev, dtype=dt, abi_widths=aw)
        with torch.no_grad():
            _out, env = block(h, return_env=True)
        trace: Dict[str, Any] = {}
        for k, v in env.items():
            if k.startswith("_"):
                continue
            if isinstance(v, torch.Tensor):
                trace[k] = _env_tensor_to_python(v)
        ex = getattr(block, "_last_expert_trace", None)
        if ex:
            trace["expert_calls"] = list(ex)
        return trace

    def export_report(self, data: dict, output_path: str, source_code: str | None = None) -> None:
        """Write a standalone HTML Glass Box report (``explain`` + ``predict``) to ``output_path``."""
        export_html_report(self, data, output_path, source_code)
