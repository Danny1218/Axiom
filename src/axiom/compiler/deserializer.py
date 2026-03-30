"""Rehydrate `ExecutionGraph` from `save_execution_bundle` outputs (.pt + *_topology.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import torch
import torch.nn as nn

from axiom.compiler.ir import extract_abi_widths, extract_global_abi
from axiom.compiler.serializer import load_state_dict
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.loop_executor import InterpretedLiquidLoop
from axiom.engine.supernet import LatentSupernet
from axiom.engine.topology import ConditionalSinkhornBlock, ExecutionGraph


def _ir_from_json(obj: Any) -> Any:
    """JSON lists → tuples so IR matches parser-shaped structures."""
    if isinstance(obj, list):
        return tuple(_ir_from_json(x) for x in obj)
    return obj


def _node_attrs(record: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in record.items() if k != "id"}


def _ir_program_from_json(obj: Any) -> list:
    """Top-level IR list from JSON (nested lists → stmt tuples)."""
    if not isinstance(obj, list):
        return []
    return [_ir_from_json(x) for x in obj]


def _ir_stmt_list(obj: Any) -> Optional[List[tuple]]:
    if not obj:
        return None
    if not isinstance(obj, list):
        return None
    return [_ir_from_json(s) for s in obj]


def _resolve_global_abi(data: Dict[str, Any], dim: int) -> Dict[str, int]:
    raw = data.get("abi")
    if isinstance(raw, dict) and raw:
        return {str(k): int(v) for k, v in raw.items()}
    if data.get("ir") is not None:
        ir_prog = _ir_program_from_json(data["ir"])
        return extract_global_abi(ir_prog, max_vars=dim)
    return {}


def _resolve_abi_widths(data: Dict[str, Any], dim: int) -> Dict[str, int]:
    raw = data.get("abi_widths")
    if isinstance(raw, dict) and raw:
        return {str(k): int(v) for k, v in raw.items()}
    if data.get("ir") is not None:
        ir_prog = _ir_program_from_json(data["ir"])
        return extract_abi_widths(ir_prog, max_vars=dim)
    return {}


def load_bundle(
    path: str | Path,
    custom_neural_registry: Optional[Dict[str, nn.Module]] = None,
) -> InterpretedBlock:
    """Load ``.axb`` written by ``save_bundle`` (``InterpretedBlock`` + optional neural weights).

    If the bundle was trained with ``custom_neural_registry``, pass the same mapping here so
    ``load_state_dict`` matches module shapes.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(p, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("invalid .axb payload")
    from axiom.security.genetic_lock import unlock_payload

    payload = unlock_payload(payload)
    topo = payload.get("topology") or {}
    if topo.get("kind") != "interpreted_block":
        raise ValueError("bundle topology is not an interpreted_block")
    ir_raw = topo.get("ir")
    if not isinstance(ir_raw, list):
        raise ValueError("bundle missing IR list")
    ir_prog = _ir_program_from_json(ir_raw)
    abi = {str(k): int(v) for k, v in (topo.get("abi") or {}).items()}
    max_unroll = int(topo.get("max_unroll", 8))
    abi_widths_raw = payload.get("abi_widths") or {}
    abi_widths = {str(k): int(v) for k, v in abi_widths_raw.items()}
    block = InterpretedBlock(
        ir_prog,
        abi,
        max_unroll=max_unroll,
        abi_widths=abi_widths,
        custom_neural_registry=custom_neural_registry,
    )
    nw = payload.get("neural_weights")
    if nw:
        block.neural_registry.load_state_dict(nw, strict=True)
    return block


def load_execution_bundle(path_prefix: str | Path) -> ExecutionGraph:
    prefix = Path(path_prefix)
    jpath = Path(str(prefix) + "_topology.json")
    pt_path = Path(str(prefix) + ".pt")
    if not jpath.is_file():
        raise FileNotFoundError(jpath)
    if not pt_path.is_file():
        raise FileNotFoundError(pt_path)
    data = json.loads(jpath.read_text(encoding="utf-8"))
    sc = data.get("supernet_config") or {}
    dim = int(sc["dim"])
    adapter_names = tuple(sc["adapter_names"])
    rank = int(sc.get("rank", 4))
    if not adapter_names:
        raise ValueError("topology JSON missing non-empty supernet_config.adapter_names")

    rc = data.get("router_config") or {}
    num_iters = int(rc.get("num_iters", 8))
    epsilon = float(rc.get("epsilon", 0.1))
    mut_thr = float(rc.get("mutation_entropy_norm_threshold", 0.92))
    lc = data.get("loop_config") or {}
    default_num_basis = int(lc.get("num_basis", 8))
    default_max_unroll = int(lc.get("max_unroll", 8))

    global_abi = _resolve_global_abi(data, dim)
    global_abi_widths = _resolve_abi_widths(data, dim)
    span = max(
        (int(global_abi[n]) + max(1, int(global_abi_widths.get(n, 1))) for n in global_abi),
        default=0,
    )
    if span > dim:
        raise ValueError(
            f"ABI spans columns 0..{span - 1} but supernet dim is {dim} "
            "(regenerate bundle with matching --dim or fix script)."
        )

    sn = LatentSupernet(dim, adapter_names, rank=rank)
    G = nx.DiGraph()
    for rec in data["nodes"]:
        G.add_node(rec["id"], **_node_attrs(rec))
    for e in data["edges"]:
        G.add_edge(e["source"], e["target"])

    topo: Tuple[str, ...] = tuple(data["topo_order"])
    modules: Dict[str, nn.Module] = {}
    for name in topo:
        attr: Dict[str, Any] = dict(G.nodes[name])
        kind = attr.get("kind")
        if kind == "conditional":
            then_e = attr["expert_then"]
            else_e = attr["expert_else"]
            then_ir = _ir_stmt_list(attr.get("then_ir"))
            else_ir = _ir_stmt_list(attr.get("else_ir"))
            then_ir_l = list(then_ir) if then_ir else []
            else_ir_l = list(else_ir) if else_ir else []
            modules[name] = ConditionalSinkhornBlock(
                sn,
                then_e,
                else_e,
                block_name=name,
                num_iters=num_iters,
                epsilon=epsilon,
                mutation_entropy_norm_threshold=mut_thr,
                then_ir=then_ir_l or None,
                else_ir=else_ir_l or None,
                abi=global_abi,
                block_max_unroll=default_max_unroll,
                abi_widths=global_abi_widths,
            )
        elif kind == "loop":
            cond_ir = _ir_from_json(attr["cond_ir"])
            body_ir: List = list(_ir_from_json(attr["body_ir"]))
            prelude = list(_ir_from_json(attr["prelude_stmts"]))
            num_basis = int(attr.get("loop_num_basis", default_num_basis))
            max_unroll = int(attr.get("loop_max_unroll", default_max_unroll))
            modules[name] = InterpretedLiquidLoop(
                dim,
                cond_ir,
                body_ir,
                prelude,
                global_abi,
                num_basis=num_basis,
                max_unroll=max_unroll,
                abi_widths=global_abi_widths,
            )
        elif kind == "stmt":
            ir_one = attr.get("ir")
            if ir_one is not None:
                stmt_t = _ir_from_json(ir_one)
                modules[name] = InterpretedBlock(
                    [stmt_t],
                    global_abi,
                    max_unroll=default_max_unroll,
                    abi_widths=global_abi_widths,
                )
            else:
                modules[name] = nn.Identity()
        else:
            raise ValueError(f"unknown node kind {kind!r} for {name!r}")

    md = nn.ModuleDict(modules)
    graph = ExecutionGraph(G, sn, md, topo, abi=global_abi, abi_widths=global_abi_widths)
    sd = load_state_dict(pt_path)
    graph.load_state_dict(sd, strict=True)
    return graph
