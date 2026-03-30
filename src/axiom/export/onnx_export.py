"""AOT ONNX export for ``InterpretedBlock`` ``.axb`` bundles (inference tensor I/O only).

This path does not preserve Glass Box / ``explain`` semantics or symbolic audit traces—only a
traced ``forward`` suitable for deployment runtimes that accept ONNX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from axiom.compiler.deserializer import load_bundle
from axiom.engine.block_executor import InterpretedBlock


class OnnxExportError(Exception):
    """Raised when a bundle cannot be exported to ONNX or tracing fails."""


def _trunk_dim(block: InterpretedBlock) -> int:
    abi, aw = block.abi, getattr(block, "abi_widths", {}) or {}
    if not abi:
        raise OnnxExportError(
            "empty ABI: cannot determine dense input width for ONNX (unsupported bundle layout)"
        )
    return max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=16)


def _expr_has_op_expert(expr: List[Any]) -> bool:
    for tup in expr:
        if not isinstance(tup, tuple) or not tup:
            continue
        op = tup[0]
        if op == "OP_EXPERT":
            return True
        if op == "OP_NEURAL" and len(tup) >= 3 and _expr_has_op_expert(list(tup[2])):
            return True
        if op == "OP_CALL":
            for a in tup[2]:
                if _expr_has_op_expert(list(a)):
                    return True
    return False


def _stmt_has_op_expert(stmt: tuple) -> bool:
    if not isinstance(stmt, tuple) or not stmt:
        return False
    op = stmt[0]
    if op == "OP_ASSIGN" and _expr_has_op_expert(list(stmt[2])):
        return True
    if op == "OP_BLEND_ASSIGN":
        return _expr_has_op_expert(list(stmt[2])) or _expr_has_op_expert(list(stmt[3]))
    if op == "OP_EXPR_STMT":
        return _expr_has_op_expert(list(stmt[1]))
    if op == "OP_CONDITIONAL":
        if _expr_has_op_expert(list(stmt[1])):
            return True
        for s in stmt[2]:
            if _stmt_has_op_expert(tuple(s) if isinstance(s, list) else s):
                return True
        for s in stmt[3]:
            if _stmt_has_op_expert(tuple(s) if isinstance(s, list) else s):
                return True
        return False
    if op == "OP_LOOP":
        if _expr_has_op_expert(list(stmt[1])):
            return True
        for s in stmt[2]:
            if _stmt_has_op_expert(tuple(s) if isinstance(s, list) else s):
                return True
        return False
    return False


def interpreted_block_ir_contains_expert(block: InterpretedBlock) -> bool:
    for st in block.ir_stmts:
        if _stmt_has_op_expert(tuple(st) if isinstance(st, list) else st):
            return True
    return False


class _TrunkWrapper(nn.Module):
    """Dense (B, D) in → dense (B, D) out via ``InterpretedBlock`` without ``return_env``."""

    def __init__(self, block: InterpretedBlock) -> None:
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if isinstance(out, tuple):
            raise OnnxExportError("internal: InterpretedBlock returned a tuple (use forward without return_env)")
        return out


def export_interpreted_block_to_onnx(
    block: InterpretedBlock,
    output_path: str | Path,
    *,
    opset_version: int = 17,
) -> None:
    """Trace ``block`` and write an ONNX file. Inference-only; may not match full Python semantics."""
    try:
        import onnx
    except ImportError as e:
        raise OnnxExportError(
            'Missing package "onnx". Install with: pip install -e ".[export]"'
        ) from e

    block.eval()
    if interpreted_block_ir_contains_expert(block):
        raise OnnxExportError(
            "bundle IR uses expert() (OP_EXPERT); ONNX export is not supported for external expert calls"
        )
    dim = _trunk_dim(block)
    dummy = torch.zeros(1, dim, dtype=torch.float32)
    wrapper = _TrunkWrapper(block)
    wrapper.eval()
    out_path = Path(output_path)
    parent = out_path.parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=opset_version,
            dynamic_axes={
                "input": {0: "batch"},
                "output": {0: "batch"},
            },
        )
    except Exception as e:
        raise OnnxExportError(
            f"ONNX export failed (IR may use control flow or ops not supported by the ONNX exporter): {e}"
        ) from e
    model = onnx.load(str(out_path))
    onnx.checker.check_model(model)


def export_bundle_to_onnx(
    bundle_path: str | Path,
    output_path: str | Path,
    *,
    opset_version: int = 17,
    custom_neural_registry: Optional[Dict[str, nn.Module]] = None,
) -> None:
    """Load an ``.axb`` with ``load_bundle`` and export if it is an ``InterpretedBlock`` bundle."""
    block = load_bundle(bundle_path, custom_neural_registry=custom_neural_registry)
    if not isinstance(block, InterpretedBlock):
        raise OnnxExportError("bundle did not load as InterpretedBlock")
    export_interpreted_block_to_onnx(block, output_path, opset_version=opset_version)
