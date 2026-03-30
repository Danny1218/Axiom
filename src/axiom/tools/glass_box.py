"""Pure helpers for the Glass Box visualizer (testable without Streamlit)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import graphviz
import torch

from axiom.engine.topology import ConditionalSinkhornBlock, ExecutionGraph


def execution_graph_to_graphviz(graph: ExecutionGraph) -> graphviz.Digraph:
    """Convert `ExecutionGraph.dag` to a Graphviz digraph with opcode color coding."""
    dot = graphviz.Digraph(name="axiom_dag", engine="dot")
    dot.attr(rankdir="LR", fontsize="10")
    for node_id in graph.dag.nodes:
        data: Dict[str, Any] = dict(graph.dag.nodes[node_id])
        kind = data.get("kind", "")
        op = data.get("op", "") or ""
        label = f"{node_id}\n{op}" if op else str(node_id)
        if kind == "conditional" or op == "OP_CONDITIONAL":
            fillcolor = "#fff2a8"
        elif kind == "loop" or op == "OP_LOOP":
            fillcolor = "#9ec5ff"
        elif kind == "stmt" or op in ("OP_ASSIGN", "OP_EXPR_STMT"):
            fillcolor = "#b8f5b8"
        else:
            fillcolor = "#dddddd"
        dot.node(
            str(node_id),
            label=label,
            style="filled",
            fillcolor=fillcolor,
            shape="box",
        )
    for u, v in graph.dag.edges:
        dot.edge(str(u), str(v))
    return dot


def routing_trace_entries(
    graph: ExecutionGraph,
    signals: Dict[str, torch.Tensor],
) -> List[Dict[str, Any]]:
    """Structured routing rows for each `ConditionalSinkhornBlock` (entropy + mean expert weights)."""
    rows: List[Dict[str, Any]] = []
    for name in graph.topo_names:
        if name not in graph.node_modules:
            continue
        mod = graph.node_modules[name]
        if not isinstance(mod, ConditionalSinkhornBlock):
            continue
        ent_t = signals.get(name)
        w_t = signals.get(f"{name}_weights")
        entropy: Optional[float] = None
        if ent_t is not None and ent_t.dim() == 0:
            entropy = float(ent_t.detach().cpu().item())
        mean_w: Optional[List[float]] = None
        if w_t is not None:
            w2 = w_t.detach().reshape(-1, w_t.shape[-1]).mean(dim=0).cpu()
            mean_w = [float(w2[i].item()) for i in range(w2.shape[0])]
        rows.append(
            {
                "block": name,
                "normalized_routing_entropy": entropy,
                "expert_then": mod.expert_then,
                "expert_else": mod.expert_else,
                "mean_router_weights_then_else": mean_w,
            }
        )
    return rows


def tensor_preview_dict(t: torch.Tensor, max_elems: int = 64) -> Dict[str, Any]:
    """Small JSON-friendly summary of an output tensor for the UI."""
    t0 = t.detach().cpu()
    flat = t0.reshape(-1)
    n = min(flat.numel(), max_elems)
    return {
        "shape": list(t0.shape),
        "dtype": str(t0.dtype),
        "flat_head": [float(flat[i].item()) for i in range(n)],
    }
