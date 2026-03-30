from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from axiom.engine.interpreter import collect_load_names_from_stmts, exec_stmt

Stmt = Tuple


class InterpretedBlock(nn.Module):
    """Runs a list of IR statements on the trunk tensor (symbolic path); other columns pass through."""

    def __init__(
        self,
        ir_stmts: List[Stmt],
        abi: Dict[str, int],
        *,
        max_unroll: int = 8,
        abi_widths: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.ir_stmts = list(ir_stmts)
        self.abi = dict(abi)
        self.abi_widths: Dict[str, int] = dict(abi_widths or {})
        self.max_unroll = int(max_unroll)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self.ir_stmts:
            return h
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
        return out
