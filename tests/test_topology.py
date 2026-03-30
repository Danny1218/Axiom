import networkx as nx
import pytest
import torch

from axiom.engine.supernet import LatentSupernet
from axiom.engine.topology import (
    ConditionalSinkhornBlock,
    ExecutionGraph,
    build_execution_graph_from_ir,
)


def _tiny_ir_with_cond():
    return [
        ("OP_ASSIGN", "a", [("OP_CONST", 1)]),
        ("OP_CONDITIONAL", [("OP_CONST", 0)], [], []),
    ]


def test_build_graph_has_conditional_node():
    sn = LatentSupernet(4, ("t", "e"), rank=2)
    ir = _tiny_ir_with_cond()
    g = build_execution_graph_from_ir(ir, sn, [("t", "e")])
    kinds = [g.node_kind(n) for n in g.topo_names]
    assert "stmt" in kinds[0]
    assert g.node_kind(g.topo_names[1]) == "conditional"
    assert isinstance(g.dag, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(g.dag)


def test_build_graph_wrong_pair_count():
    sn = LatentSupernet(3, ("t", "e"))
    with pytest.raises(ValueError):
        build_execution_graph_from_ir(_tiny_ir_with_cond(), sn, [])


def test_execution_forward_and_autograd():
    torch.manual_seed(0)
    dim = 4
    sn = LatentSupernet(dim, ("t", "e"), rank=2)
    sn.set_masks({"t": 1.0, "e": 1.0})
    ir = _tiny_ir_with_cond()
    g = build_execution_graph_from_ir(ir, sn, [("t", "e")], router_iters=10)
    x = torch.randn(3, dim, requires_grad=True)
    y, _, _ = g(x)
    assert y.shape == x.shape
    loss = y.pow(2).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_conditional_block_routing_changes_output():
    torch.manual_seed(1)
    dim = 5
    sn = LatentSupernet(dim, ("t", "e"), rank=2)
    sn.set_masks({"t": 1.0, "e": 1.0})
    blk = ConditionalSinkhornBlock(sn, "t", "e", num_iters=12)
    h = torch.randn(2, dim)
    y, _, sig = blk(h)
    assert "cond" in sig and sig["cond"].shape == ()
    base = sn.trunk(h)
    assert not torch.allclose(y, base)


def test_linear_dag_matches_ir_order():
    sn = LatentSupernet(3, ("a", "b"))
    ir = [
        ("OP_EXPR_STMT", []),
        ("OP_CONDITIONAL", [], [], []),
    ]
    g = build_execution_graph_from_ir(ir, sn, [("a", "b")])
    assert len(g.topo_names) == 2
