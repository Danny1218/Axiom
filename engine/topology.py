from __future__ import annotations

from typing import List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn

from engine.router import SinkhornRouter
from engine.supernet import LatentSupernet

IRList = List[tuple]
ExpertPair = Tuple[str, str]


class ConditionalSinkhornBlock(nn.Module):
    """Sinkhorn-balanced mix of two supernet adapters (then / else experts)."""

    def __init__(
        self,
        supernet: LatentSupernet,
        expert_then: str,
        expert_else: str,
        *,
        num_iters: int = 8,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        if expert_then not in supernet.adapters or expert_else not in supernet.adapters:
            raise KeyError("both experts must exist on the supernet")
        self.supernet = supernet
        self.expert_then = expert_then
        self.expert_else = expert_else
        self.router = SinkhornRouter(supernet.dim, 2, num_iters=num_iters, epsilon=epsilon)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        it = self.supernet._name_to_idx[self.expert_then]
        ie = self.supernet._name_to_idx[self.expert_else]
        mask = torch.zeros(2, device=h.device, dtype=torch.bool)
        if self.supernet.adapter_mask[it] >= 0.5:
            mask[0] = True
        if self.supernet.adapter_mask[ie] >= 0.5:
            mask[1] = True
        if not mask.any():
            return h
        w = self.router(h, expert_mask=mask)
        *lead, _ = w.shape
        w2 = w.reshape(-1, 2)
        h_flat = h.reshape(-1, h.shape[-1])
        y0 = self.supernet.adapters[self.expert_then](h_flat)
        y1 = self.supernet.adapters[self.expert_else](h_flat)
        out = h_flat + w2[:, 0:1] * y0 + w2[:, 1:2] * y1
        return out.reshape(*lead, h.shape[-1])


class ExecutionGraph(nn.Module):
    """NetworkX DAG of logical IR steps; forward walks topo order after shared trunk."""

    def __init__(
        self,
        dag: nx.DiGraph,
        supernet: LatentSupernet,
        node_modules: nn.ModuleDict,
        topo_names: Tuple[str, ...],
        entry: str = "src",
    ) -> None:
        super().__init__()
        self.dag = dag
        self.supernet = supernet
        self.topo_names = topo_names
        self.entry = entry
        self.add_module("supernet", supernet)
        self.add_module("node_modules", node_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.supernet.trunk(x)
        for name in self.topo_names:
            if name == self.entry:
                continue
            h = self.node_modules[name](h)
        return h

    def node_kind(self, name: str) -> str:
        return self.dag.nodes[name].get("kind", "unknown")


def build_execution_graph_from_ir(
    ir: IRList,
    supernet: LatentSupernet,
    conditional_experts: Sequence[ExpertPair],
    *,
    router_iters: int = 8,
    router_eps: float = 0.1,
) -> ExecutionGraph:
    """
    Map IR to a DAG: each statement is a node; OP_CONDITIONAL inserts ConditionalSinkhornBlock
    (Sinkhorn router + two LoRA experts). Other ops use Identity to preserve linear flow.
    """
    conds = [i for i in ir if i[0] == "OP_CONDITIONAL"]
    if len(conditional_experts) != len(conds):
        raise ValueError(
            f"need one expert pair per OP_CONDITIONAL: got {len(conditional_experts)} pairs, "
            f"{len(conds)} conditionals in IR"
        )

    G = nx.DiGraph()
    G.add_node("src", kind="source")
    prev = "src"
    modules: dict[str, nn.Module] = {}
    cidx = 0
    order: List[str] = []

    for instr in ir:
        op = instr[0]
        if op == "OP_CONDITIONAL":
            name = f"cond_{cidx}"
            then_e, else_e = conditional_experts[cidx]
            modules[name] = ConditionalSinkhornBlock(
                supernet,
                then_e,
                else_e,
                num_iters=router_iters,
                epsilon=router_eps,
            )
            G.add_node(name, kind="conditional", op="OP_CONDITIONAL")
            G.add_edge(prev, name)
            order.append(name)
            prev = name
            cidx += 1
        else:
            name = f"stmt_{len(order)}_{op}"
            modules[name] = nn.Identity()
            G.add_node(name, kind="stmt", op=op)
            G.add_edge(prev, name)
            order.append(name)
            prev = name

    md = nn.ModuleDict(modules)
    topo = tuple(n for n in nx.topological_sort(G) if n != "src")
    if tuple(order) != topo:
        raise RuntimeError("IR linear chain order mismatch vs topological sort")
    return ExecutionGraph(G, supernet, md, topo)
