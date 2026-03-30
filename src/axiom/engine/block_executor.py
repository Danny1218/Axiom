from __future__ import annotations

from typing import Dict, List, Set, Tuple

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
    ) -> None:
        super().__init__()
        self.ir_stmts = list(ir_stmts)
        self.abi = dict(abi)
        self.col_to_name = {col: name for name, col in self.abi.items()}
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
                env[name] = h[:, col].clone() if col < D else z.clone()
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
        cols: List[torch.Tensor] = []
        for i in range(D):
            name = self.col_to_name.get(i)
            if name is not None and name in env:
                cols.append(env[name])
            else:
                cols.append(h[:, i])
        return torch.stack(cols, dim=1)
