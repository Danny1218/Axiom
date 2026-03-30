from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from engine.interpreter import run_loop_snapshots
from engine.ssm import LiquidKANNode

Stmt = Tuple
ExprIR = List[Tuple]


class InterpretedLiquidLoop(nn.Module):
    """Vectorized IR interpreter over batch; feeds (B, T, D) into `LiquidKANNode.forward_sequence_tensors`."""

    def __init__(
        self,
        dim: int,
        cond_ir: ExprIR,
        body_ir: List[Stmt],
        prelude_stmts: List[Stmt],
        seed_map: Dict[str, int],
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

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        lead = h.shape[:-1]
        d = h.shape[-1]
        flat = h.reshape(-1, d)
        seq, seq_mask = run_loop_snapshots(
            flat,
            self.cond_ir,
            self.body_ir,
            dim=d,
            max_unroll=self.max_unroll,
            seed_map=self.seed_map,
            prelude_stmts=self.prelude_stmts,
            device=h.device,
            dtype=h.dtype,
            trunk_dim=flat.shape[-1],
        )
        if seq.shape[1] == 0:
            y = self.kan.forward(flat)
        else:
            y = self.kan.forward_sequence_tensors(seq, h_init=flat, mask=seq_mask)
        return y.reshape(*lead, d), {}, {}
