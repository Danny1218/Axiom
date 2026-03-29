"""Compile-time wiring: IR → execution DAG (Sinkhorn at OP_CONDITIONAL)."""

from __future__ import annotations

from typing import Sequence, Tuple

from compiler.ir import IRList
from engine.supernet import LatentSupernet
from engine.topology import ExecutionGraph, build_execution_graph_from_ir

ExpertPair = Tuple[str, str]


def wire_execution_graph(
    ir: IRList,
    supernet: LatentSupernet,
    conditional_experts: Sequence[ExpertPair],
    *,
    router_iters: int = 8,
    router_eps: float = 0.1,
) -> ExecutionGraph:
    """Each OP_CONDITIONAL in IR gets a Sinkhorn router node over the paired LoRA experts."""
    return build_execution_graph_from_ir(
        ir,
        supernet,
        conditional_experts,
        router_iters=router_iters,
        router_eps=router_eps,
    )
