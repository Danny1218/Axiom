"""High-level Python API: load ``.axb`` bundles and predict with dicts / batches / DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import torch

from axiom.compiler.deserializer import load_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _abi_outputs_from_trunk_row, _inputs_to_tensor


def _trunk_dim_from_block_abi(block: InterpretedBlock) -> int:
    abi, aw = block.abi, getattr(block, "abi_widths", {}) or {}
    return max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=16)


def load(bundle_path: str | Path) -> AxiomModel:
    return AxiomModel(load_bundle(bundle_path))


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
