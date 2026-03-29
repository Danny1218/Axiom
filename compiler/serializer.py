from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from engine.topology import ExecutionGraph


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
    nodes = []
    for n, attr in G.nodes(data=True):
        nodes.append({"id": n, **{k: _sanitize_node_attr(v) for k, v in attr.items()}})
    edges = [{"source": u, "target": v} for u, v in G.edges()]
    return {
        "directed": True,
        "topo_order": list(graph.topo_names),
        "nodes": nodes,
        "edges": edges,
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


def load_state_dict(path: str | Path) -> Dict[str, torch.Tensor]:
    p = str(path)
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(p, map_location="cpu")
