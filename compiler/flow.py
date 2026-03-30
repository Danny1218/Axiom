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
    mutation_entropy_norm_threshold: float = 0.92,
    loop_max_unroll: int = 8,
    loop_num_basis: int = 8,
) -> ExecutionGraph:
    """OP_CONDITIONAL → Sinkhorn block; OP_LOOP → InterpretedLiquidLoop (IR + liquid sequence)."""
    # ABI is attached on the graph inside ``build_execution_graph_from_ir`` (``extract_global_abi``).
    return build_execution_graph_from_ir(
        ir,
        supernet,
        conditional_experts,
        router_iters=router_iters,
        router_eps=router_eps,
        mutation_entropy_norm_threshold=mutation_entropy_norm_threshold,
        loop_max_unroll=loop_max_unroll,
        loop_num_basis=loop_num_basis,
    )
