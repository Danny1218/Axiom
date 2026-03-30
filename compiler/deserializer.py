"""Rehydrate `ExecutionGraph` from `save_execution_bundle` outputs (.pt + *_topology.json)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import torch.nn as nn

from compiler.ir import extract_global_abi
from compiler.serializer import load_state_dict
from engine.loop_executor import InterpretedLiquidLoop
from engine.supernet import LatentSupernet
from engine.topology import ConditionalSinkhornBlock, ExecutionGraph


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


def _resolve_global_abi(data: Dict[str, Any], dim: int) -> Dict[str, int]:
    raw = data.get("abi")
    if isinstance(raw, dict) and raw:
        return {str(k): int(v) for k, v in raw.items()}
    if data.get("ir") is not None:
        ir_prog = _ir_program_from_json(data["ir"])
        return extract_global_abi(ir_prog, max_vars=dim)
    return {}


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
            modules[name] = ConditionalSinkhornBlock(
                sn,
                then_e,
                else_e,
                block_name=name,
                num_iters=num_iters,
                epsilon=epsilon,
                mutation_entropy_norm_threshold=mut_thr,
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
            )
        elif kind == "stmt":
            modules[name] = nn.Identity()
        else:
            raise ValueError(f"unknown node kind {kind!r} for {name!r}")

    md = nn.ModuleDict(modules)
    graph = ExecutionGraph(G, sn, md, topo, abi=global_abi)
    sd = load_state_dict(pt_path)
    graph.load_state_dict(sd, strict=True)
    return graph
