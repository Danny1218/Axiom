import networkx as nx
import pytest
import torch
import torch.nn as nn

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


def _piecewise_ir():
    """if (a > 0) { b = 10; } else { b = -10; } — ``a`` read from trunk ABI column."""
    return [
        (
            "OP_CONDITIONAL",
            [("OP_LOAD", "a"), ("OP_CONST", 0.0), ("OP_CMP_GT",)],
            [("OP_ASSIGN", "b", [("OP_CONST", 10.0)])],
            [("OP_ASSIGN", "b", [("OP_CONST", -10.0)])],
        ),
    ]


def test_conditional_block_stores_cond_ir():
    sn = LatentSupernet(4, ("t", "e"), rank=2)
    g = build_execution_graph_from_ir(_piecewise_ir(), sn, [("t", "e")])
    blk = g.conditional_blocks()[0]
    assert blk.cond_ir
    assert g.dag.nodes[g.topo_names[-1]].get("cond_ir") == blk.cond_ir


def test_execution_graph_symbolic_conditional_with_neutral_adapters():
    """Top-level if/else follows compiled predicate when router/adapters are neutral."""
    torch.manual_seed(7)
    dim = 4
    trunk = nn.Linear(dim, dim, bias=False)
    nn.init.eye_(trunk.weight)
    sn = LatentSupernet(dim, ("then_0", "else_0"), rank=2, trunk=trunk)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    for p in sn.adapters.values():
        nn.init.zeros_(p.U)
        nn.init.zeros_(p.V)
        nn.init.zeros_(p.W)
    ir = _piecewise_ir()
    g = build_execution_graph_from_ir(ir, sn, [("then_0", "else_0")], router_iters=8)
    blk = g.conditional_blocks()[0]
    nn.init.zeros_(blk.router.proj.weight)
    nn.init.zeros_(blk.router.proj.bias)
    ac, bc = g.abi["a"], g.abi["b"]
    x_pos = torch.zeros(1, dim)
    x_pos[0, ac] = 2.0
    x_neg = torch.zeros(1, dim)
    x_neg[0, ac] = -3.0
    with torch.no_grad():
        out_pos, _, _ = g(x_pos)
        out_neg, _, _ = g(x_neg)
    assert out_pos[0, bc].item() == pytest.approx(10.0, abs=1e-4)
    assert out_neg[0, bc].item() == pytest.approx(-10.0, abs=1e-4)
