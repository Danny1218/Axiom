from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import torch
import torch.nn as nn

from compiler.ir import extract_global_abi
from engine.block_executor import InterpretedBlock
from engine.loop_executor import InterpretedLiquidLoop
from engine.router import SinkhornRouter
from engine.supernet import LatentSupernet

IRList = List[tuple]
ExpertPair = Tuple[str, str]

_PRELUDE_OPS = frozenset({"OP_ASSIGN", "OP_EXPR_STMT"})


def _absorbed_prelude_indices(ir: IRList) -> Set[int]:
    absorbed: Set[int] = set()
    for k, instr in enumerate(ir):
        if instr[0] != "OP_LOOP":
            continue
        j = k - 1
        while j >= 0 and ir[j][0] in _PRELUDE_OPS:
            absorbed.add(j)
            j -= 1
    return absorbed


def _prelude_stmts_before_loop(ir: IRList, k: int) -> List[tuple]:
    stmts: List[tuple] = []
    j = k - 1
    while j >= 0 and ir[j][0] in _PRELUDE_OPS:
        stmts.insert(0, ir[j])
        j -= 1
    return stmts


class ConditionalSinkhornBlock(nn.Module):
    """Sinkhorn-balanced mix: symbolic IR on each branch + TT-LoRA residual, then weight blend."""

    def __init__(
        self,
        supernet: LatentSupernet,
        expert_then: str,
        expert_else: str,
        *,
        block_name: str = "cond",
        num_iters: int = 8,
        epsilon: float = 0.1,
        mutation_entropy_norm_threshold: float = 0.92,
        then_ir: Optional[List[tuple]] = None,
        else_ir: Optional[List[tuple]] = None,
        abi: Optional[Dict[str, int]] = None,
        block_max_unroll: int = 8,
    ) -> None:
        super().__init__()
        if expert_then not in supernet.adapters or expert_else not in supernet.adapters:
            raise KeyError("both experts must exist on the supernet")
        self.supernet = supernet
        self.expert_then = expert_then
        self.expert_else = expert_else
        self.block_name = block_name
        self.router = SinkhornRouter(
            supernet.dim,
            2,
            num_iters=num_iters,
            epsilon=epsilon,
            mutation_entropy_norm_threshold=mutation_entropy_norm_threshold,
        )
        abi_d = dict(abi or {})
        then_l = list(then_ir) if then_ir else []
        else_l = list(else_ir) if else_ir else []
        self.then_block = (
            InterpretedBlock(then_l, abi_d, max_unroll=block_max_unroll) if then_l else None
        )
        self.else_block = (
            InterpretedBlock(else_l, abi_d, max_unroll=block_max_unroll) if else_l else None
        )

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        local_shadows: Dict[str, torch.Tensor] = {}
        it = self.supernet._name_to_idx[self.expert_then]
        ie = self.supernet._name_to_idx[self.expert_else]
        mask = torch.stack(
            [
                (self.supernet.adapter_mask[it] >= 0.5).to(device=h.device, dtype=torch.bool),
                (self.supernet.adapter_mask[ie] >= 0.5).to(device=h.device, dtype=torch.bool),
            ],
            dim=0,
        )
        h_then = self.then_block(h) if self.then_block is not None else h
        h_else = self.else_block(h) if self.else_block is not None else h
        w, entropy_tensor = self.router(h, expert_mask=mask)
        *lead, _ = w.shape
        w2 = w.reshape(-1, 2)
        h_flat = h.reshape(-1, h.shape[-1])
        h_then_f = h_then.reshape(-1, h.shape[-1])
        h_else_f = h_else.reshape(-1, h.shape[-1])
        y0 = self.supernet.adapters[self.expert_then](h_flat)
        y1 = self.supernet.adapters[self.expert_else](h_flat)
        y0_mix = torch.where(self.supernet.is_shadow[it] >= 0.5, y0.detach(), y0)
        y1_mix = torch.where(self.supernet.is_shadow[ie] >= 0.5, y1.detach(), y1)
        branch0 = h_then_f + y0_mix
        branch1 = h_else_f + y1_mix
        out_flat = w2[:, 0:1] * branch0 + w2[:, 1:2] * branch1
        # Always expose raw adapter outputs; trainer applies localized MSE only when is_shadow.
        local_shadows[self.expert_then] = y0
        local_shadows[self.expert_else] = y1
        return out_flat.reshape(*lead, h.shape[-1]), local_shadows, {self.block_name: entropy_tensor}


class ExecutionGraph(nn.Module):
    """NetworkX DAG of logical IR steps; forward walks topo order after shared trunk."""

    def __init__(
        self,
        dag: nx.DiGraph,
        supernet: LatentSupernet,
        node_modules: nn.ModuleDict,
        topo_names: Tuple[str, ...],
        entry: str = "src",
        *,
        abi: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.dag = dag
        self.supernet = supernet
        self.topo_names = topo_names
        self.entry = entry
        self.abi: Dict[str, int] = dict(abi or {})
        self.add_module("supernet", supernet)
        self.add_module("node_modules", node_modules)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        h = self.supernet.trunk(x)
        all_shadows: Dict[str, torch.Tensor] = {}
        all_signals: Dict[str, torch.Tensor] = {}
        for name in self.topo_names:
            if name == self.entry:
                continue
            mod = self.node_modules[name]
            if isinstance(mod, (ConditionalSinkhornBlock, InterpretedLiquidLoop)):
                h, shadows, signals = mod(h)
                all_shadows.update(shadows)
                all_signals.update(signals)
            else:
                h = mod(h)
        return h, all_shadows, all_signals

    def block_mutation_thresholds(self) -> Dict[str, float]:
        """Per conditional node: router entropy threshold for `MetaCompiler.react_to_signals`."""
        out: Dict[str, float] = {}
        for n in self.topo_names:
            m = self.node_modules[n]
            if isinstance(m, ConditionalSinkhornBlock):
                out[n] = m.router.mutation_entropy_norm_threshold
        return out

    def node_kind(self, name: str) -> str:
        return self.dag.nodes[name].get("kind", "unknown")

    def conditional_blocks(self) -> List[ConditionalSinkhornBlock]:
        return [
            self.node_modules[n]
            for n in self.topo_names
            if isinstance(self.node_modules[n], ConditionalSinkhornBlock)
        ]

    def routers(self) -> List[SinkhornRouter]:
        return [b.router for b in self.conditional_blocks()]


def build_execution_graph_from_ir(
    ir: IRList,
    supernet: LatentSupernet,
    conditional_experts: Sequence[ExpertPair],
    *,
    router_iters: int = 8,
    router_eps: float = 0.1,
    mutation_entropy_norm_threshold: float = 0.92,
    loop_max_unroll: int = 8,
    loop_num_basis: int = 8,
    global_abi: Optional[Dict[str, int]] = None,
) -> ExecutionGraph:
    """
    Map IR to a DAG: OP_CONDITIONAL → ConditionalSinkhornBlock; OP_LOOP → InterpretedLiquidLoop
    (IR cond/body + contiguous prelude assigns); other statements → Identity.
    OP_ASSIGN/OP_EXPR_STMT immediately before a loop are absorbed into the loop node (no duplicate stmt nodes).
    """
    conds = [i for i in ir if i[0] == "OP_CONDITIONAL"]
    if len(conditional_experts) != len(conds):
        raise ValueError(
            f"need one expert pair per OP_CONDITIONAL: got {len(conditional_experts)} pairs, "
            f"{len(conds)} conditionals in IR"
        )
    if global_abi is None:
        global_abi = extract_global_abi(ir, max_vars=supernet.dim)

    G = nx.DiGraph()
    G.add_node("src", kind="source")
    prev = "src"
    modules: dict[str, nn.Module] = {}
    cidx = 0
    lidx = 0
    order: List[str] = []
    absorbed = _absorbed_prelude_indices(ir)

    for idx, instr in enumerate(ir):
        if idx in absorbed:
            continue
        op = instr[0]
        if op == "OP_CONDITIONAL":
            name = f"cond_{cidx}"
            then_e, else_e = conditional_experts[cidx]
            _, _cond_ir, then_ir_list, else_ir_list = instr
            then_ir = list(then_ir_list)
            else_ir = list(else_ir_list)
            modules[name] = ConditionalSinkhornBlock(
                supernet,
                then_e,
                else_e,
                block_name=name,
                num_iters=router_iters,
                epsilon=router_eps,
                mutation_entropy_norm_threshold=mutation_entropy_norm_threshold,
                then_ir=then_ir,
                else_ir=else_ir,
                abi=global_abi,
                block_max_unroll=loop_max_unroll,
            )
            G.add_node(
                name,
                kind="conditional",
                op="OP_CONDITIONAL",
                then_ir=then_ir,
                else_ir=else_ir,
            )
            G.add_edge(prev, name)
            order.append(name)
            prev = name
            cidx += 1
        elif op == "OP_LOOP":
            name = f"loop_{lidx}"
            _, cond_ir, body_ir = instr
            prelude = _prelude_stmts_before_loop(ir, idx)
            modules[name] = InterpretedLiquidLoop(
                supernet.dim,
                cond_ir,
                body_ir,
                prelude,
                global_abi,
                num_basis=loop_num_basis,
                max_unroll=loop_max_unroll,
            )
            G.add_node(
                name,
                kind="loop",
                op="OP_LOOP",
                cond_ir=cond_ir,
                body_ir=body_ir,
                prelude_stmts=prelude,
            )
            G.add_edge(prev, name)
            order.append(name)
            prev = name
            lidx += 1
        else:
            name = f"stmt_{len(order)}_{op}"
            modules[name] = InterpretedBlock(
                [instr], global_abi, max_unroll=loop_max_unroll
            )
            G.add_node(name, kind="stmt", op=op, ir=instr)
            G.add_edge(prev, name)
            order.append(name)
            prev = name

    md = nn.ModuleDict(modules)
    topo = tuple(n for n in nx.topological_sort(G) if n != "src")
    if tuple(order) != topo:
        raise RuntimeError("IR linear chain order mismatch vs topological sort")
    return ExecutionGraph(G, supernet, md, topo, abi=global_abi)
