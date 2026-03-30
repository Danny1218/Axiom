from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from axiom.compiler.ir import extract_neural_node_specs

from axiom.engine.block_executor import build_neural_module
from axiom.engine.interpreter import run_loop_snapshots
from axiom.engine.ssm import LiquidKANNode

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
        abi_widths: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.cond_ir = cond_ir
        self.body_ir = body_ir
        self.prelude_stmts = prelude_stmts
        self.seed_map = dict(seed_map)
        self.abi_widths: Dict[str, int] = dict(abi_widths or {})
        self.max_unroll = max_unroll
        self.kan = LiquidKANNode(dim, num_basis=num_basis, max_unroll=max_unroll)
        combined: List[Stmt] = list(prelude_stmts) + [("OP_LOOP", list(cond_ir), list(body_ir))]
        spec = extract_neural_node_specs(combined, self.abi_widths)
        self.neural_registry: nn.ModuleDict = nn.ModuleDict(
            {
                nid: build_neural_module(w, arch, max_unroll=max_unroll)
                for nid, (w, arch) in spec.items()
            }
        )

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
            abi_widths=self.abi_widths,
            neural_registry=self.neural_registry,
        )
        if seq.shape[1] == 0:
            y = self.kan.forward(flat)
        else:
            y = self.kan.forward_sequence_tensors(seq, h_init=flat, mask=seq_mask)
        return y.reshape(*lead, d), {}, {}
