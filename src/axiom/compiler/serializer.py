from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.loop_executor import InterpretedLiquidLoop
from axiom.engine.topology import ConditionalSinkhornBlock, ExecutionGraph

AXB_BUNDLE_VERSION = 2
AXB_WEIGHTS_SUFFIX = ".weights.pt"


def _supernet_rank(graph: ExecutionGraph) -> int:
    names = graph.supernet.adapter_names
    if not names:
        return 4
    return graph.supernet.adapters[names[0]].rank


def _jsonable_ir(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, tuple):
        return [_jsonable_ir(x) for x in obj]
    if isinstance(obj, list):
        return [_jsonable_ir(x) for x in obj]
    return str(obj)


def _sanitize_node_attr(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        try:
            return _jsonable_ir(list(value))
        except Exception:
            return str(value)[:8000]
    return str(value)[:8000]


def execution_topology_to_dict(graph: ExecutionGraph) -> Dict[str, Any]:
    G = graph.dag
    sn = graph.supernet
    adapter_names = list(sn.adapter_names)
    router_config = {
        "num_iters": 8,
        "epsilon": 0.1,
        "mutation_entropy_norm_threshold": 0.92,
    }
    loop_config = {"num_basis": 8, "max_unroll": 8}
    seen_router = False
    seen_loop = False
    nodes = []
    for n, attr in G.nodes(data=True):
        row: Dict[str, Any] = {"id": n, **{k: _sanitize_node_attr(v) for k, v in attr.items()}}
        if n in graph.node_modules:
            mod = graph.node_modules[n]
            if isinstance(mod, ConditionalSinkhornBlock):
                row["expert_then"] = mod.expert_then
                row["expert_else"] = mod.expert_else
                if not seen_router:
                    r = mod.router
                    router_config = {
                        "num_iters": r.num_iters,
                        "epsilon": r.epsilon,
                        "mutation_entropy_norm_threshold": r.mutation_entropy_norm_threshold,
                    }
                    seen_router = True
            elif isinstance(mod, InterpretedLiquidLoop):
                row["loop_num_basis"] = mod.kan.num_basis
                row["loop_max_unroll"] = mod.max_unroll
                if not seen_loop:
                    loop_config = {
                        "num_basis": mod.kan.num_basis,
                        "max_unroll": mod.max_unroll,
                    }
                    seen_loop = True
        nodes.append(row)
    edges = [{"source": u, "target": v} for u, v in G.edges()]
    return {
        "directed": True,
        "topo_order": list(graph.topo_names),
        "nodes": nodes,
        "edges": edges,
        "supernet_config": {
            "dim": sn.dim,
            "adapter_names": adapter_names,
            "rank": _supernet_rank(graph),
        },
        "router_config": router_config,
        "loop_config": loop_config,
        "abi": {k: int(v) for k, v in getattr(graph, "abi", {}).items()},
        "abi_widths": {k: int(v) for k, v in getattr(graph, "abi_widths", {}).items()},
    }


def save_execution_bundle(
    graph: ExecutionGraph,
    path_prefix: str | Path,
    *,
    ir: Optional[List[tuple]] = None,
) -> None:
    """
    Persist weights (`*.pt` torch state_dict) and NetworkX-friendly topology JSON (`*_topology.json`).
    """
    prefix = Path(path_prefix)
    parent = prefix.parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph.state_dict(), str(prefix) + ".pt")
    topo = execution_topology_to_dict(graph)
    if ir is not None:
        topo["ir"] = _jsonable_ir(ir)
    with open(str(prefix) + "_topology.json", "w", encoding="utf-8") as f:
        json.dump(topo, f, indent=2)


def interpreted_block_topology_dict(block: InterpretedBlock) -> Dict[str, Any]:
    """Symbolic payload for ``save_bundle`` / ``.axb`` (IR + ABI + loop unroll)."""
    return {
        "kind": "interpreted_block",
        "ir": _jsonable_ir(list(block.ir_stmts)),
        "abi": {str(k): int(v) for k, v in block.abi.items()},
        "max_unroll": int(block.max_unroll),
    }


def save_bundle(
    block: InterpretedBlock,
    path: str | Path,
    *,
    lock_mode: Optional[str] = None,
) -> None:
    """Persist a trained ``InterpretedBlock`` (IR + ABI + ``neural_registry`` weights) in one ``.axb`` file.

    Optional ``lock_mode``: ``none`` (default), ``device``, ``host``, ``env-secret`` — encrypts only the
    serialized neural weight blob (topology stays readable). Requires ``pip install -e ".[lock]"`` when not ``none``.
    """
    p = Path(path)
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    topology = interpreted_block_topology_dict(block)
    abi_widths = {str(k): int(v) for k, v in block.abi_widths.items()}
    neural_weights: Optional[Dict[str, torch.Tensor]]
    if hasattr(block, "neural_registry") and len(block.neural_registry) > 0:
        neural_weights = block.neural_registry.state_dict()
    else:
        neural_weights = None
    payload: Dict[str, Any] = {
        "version": AXB_BUNDLE_VERSION,
        "topology": topology,
        "abi_widths": abi_widths,
        "neural_weights": None,
    }
    weights_to_save = neural_weights
    lm = lock_mode
    if lm is not None and str(lm).lower().strip() not in ("none", ""):
        from axiom.security.genetic_lock import apply_lock_to_payload

        payload["neural_weights"] = weights_to_save
        apply_lock_to_payload(payload, lm)
        weights_to_save = None
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    weights_path = Path(str(p) + AXB_WEIGHTS_SUFFIX)
    if weights_to_save:
        torch.save(weights_to_save, str(weights_path))
    elif weights_path.is_file():
        weights_path.unlink()


def _is_v1_pickle_bundle(raw: bytes) -> bool:
    return len(raw) >= 2 and raw[0:2] == b"\x80\x04"


def load_state_dict(path: str | Path) -> Dict[str, torch.Tensor]:
    p = str(path)
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(p, map_location="cpu")
