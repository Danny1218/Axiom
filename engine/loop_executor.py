from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from engine.interpreter import run_loop_snapshots
from engine.ssm import LiquidKANNode

Stmt = Tuple
ExprIR = List[Tuple]


class InterpretedLiquidLoop(nn.Module):
    """Runs IR cond/body in the interpreter; feeds snapshot sequence into `LiquidKANNode.forward_sequence_tensors`."""

    def __init__(
        self,
        dim: int,
        cond_ir: ExprIR,
        body_ir: List[Stmt],
        prelude_stmts: List[Stmt],
        seed_map: Dict[int, str],
        *,
        num_basis: int = 8,
        max_unroll: int = 8,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.cond_ir = cond_ir
        self.body_ir = body_ir
        self.prelude_stmts = prelude_stmts
        self.seed_map = dict(seed_map)
        self.max_unroll = max_unroll
        self.kan = LiquidKANNode(dim, num_basis=num_basis, max_unroll=max_unroll)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        lead = h.shape[:-1]
        d = h.shape[-1]
        flat = h.reshape(-1, d)
        outs: List[torch.Tensor] = []
        for b in range(flat.size(0)):
            seq = run_loop_snapshots(
                flat[b],
                self.cond_ir,
                self.body_ir,
                dim=d,
                max_unroll=self.max_unroll,
                seed_map=self.seed_map,
                prelude_stmts=self.prelude_stmts,
            )
            row = flat[b : b + 1]
            if seq.size(0) == 0:
                outs.append(self.kan.forward(row))
            else:
                outs.append(self.kan.forward_sequence_tensors(seq.unsqueeze(0), h_init=row))
        y = torch.cat(outs, dim=0)
        return y.reshape(*lead, d)
