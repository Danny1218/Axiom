from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from axiom.compiler.ir import extract_neural_node_specs
from axiom.engine.interpreter import collect_load_names_from_stmts, exec_stmt
from axiom.engine.ssm import LiquidKANNode
from axiom.primitives.liquid_tensor import LiquidFeatureReadout

Stmt = Tuple


def _neural_mlp(in_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


class _KanNeuralReadout(nn.Module):
    """Spline / RBF path from ``LiquidKANNode`` + linear scalar readout (Phase 43)."""

    def __init__(self, in_dim: int, *, num_basis: int = 8, max_unroll: int = 8) -> None:
        super().__init__()
        self.kan = LiquidKANNode(in_dim, num_basis=num_basis, max_unroll=max(1, int(max_unroll)))
        self.readout = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.readout(self.kan(x))


def build_neural_module(in_dim: int, arch: str, *, max_unroll: int = 8) -> nn.Module:
    a = str(arch).strip().lower()
    if a == "kan":
        u = max(1, min(int(max_unroll), 16))
        return _KanNeuralReadout(in_dim, num_basis=8, max_unroll=u)
    if a == "liquid":
        return LiquidFeatureReadout(in_dim)
    return _neural_mlp(in_dim)


class InterpretedBlock(nn.Module):
    """Runs a list of IR statements on the trunk tensor (symbolic path); other columns pass through.

    With ``forward(h, return_env=True)``, returns ``(out, env)`` so callers can inspect per-variable tensors
    after execution (Phase 41 explainability).
    """

    def __init__(
        self,
        ir_stmts: List[Stmt],
        abi: Dict[str, int],
        *,
        max_unroll: int = 8,
        abi_widths: Optional[Dict[str, int]] = None,
        custom_neural_registry: Optional[Dict[str, nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.ir_stmts = list(ir_stmts)
        self.abi = dict(abi)
        self.abi_widths: Dict[str, int] = dict(abi_widths or {})
        self.max_unroll = int(max_unroll)
        spec = extract_neural_node_specs(self.ir_stmts, self.abi_widths)
        custom = dict(custom_neural_registry or {})
        built: Dict[str, nn.Module] = {}
        for nid, (w, arch) in spec.items():
            if nid in custom:
                built[nid] = custom[nid]
            else:
                built[nid] = build_neural_module(w, arch, max_unroll=self.max_unroll)
        self.neural_registry: nn.ModuleDict = nn.ModuleDict(built)

    def forward(
        self, h: torch.Tensor, return_env: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.ir_stmts:
            return (h, {}) if return_env else h
        if h.dim() != 2:
            raise ValueError("InterpretedBlock expects h (B, D)")
        B, D = h.shape
        device, dtype = h.device, h.dtype
        names: Set[str] = set(self.abi.keys()) | set(collect_load_names_from_stmts(self.ir_stmts))
        env: Dict[str, torch.Tensor] = {}
        z = torch.zeros(B, device=device, dtype=dtype)
        for name in names:
            if name in self.abi:
                col = self.abi[name]
                w = max(1, int(self.abi_widths.get(name, 1)))
                if col + w <= D:
                    if w == 1:
                        env[name] = h[:, col].clone()
                    else:
                        env[name] = h[:, col : col + w].clone()
                else:
                    env[name] = z.clone() if w == 1 else torch.zeros(B, w, device=device, dtype=dtype)
            else:
                env[name] = z.clone()
        for stmt in self.ir_stmts:
            exec_stmt(
                env,
                stmt,
                B=B,
                dim=D,
                max_unroll=self.max_unroll,
                device=device,
                dtype=dtype,
                abi_widths=self.abi_widths,
                neural_registry=self.neural_registry,
            )
        out = h.clone()
        for name, col in self.abi.items():
            if name not in env:
                continue
            w = max(1, int(self.abi_widths.get(name, 1)))
            t = env[name]
            if col + w > D:
                continue
            if w == 1 and t.dim() == 1:
                out[:, col] = t
            elif t.dim() == 2 and t.shape[1] == w:
                out[:, col : col + w] = t
            elif t.dim() == 1 and w == 1:
                out[:, col] = t
        return (out, env) if return_env else out
