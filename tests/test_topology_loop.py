import torch

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax, reset_parser
from engine.ssm import LiquidKANNode
from engine.supernet import LatentSupernet
from engine.topology import build_execution_graph_from_ir


def test_op_loop_instantiates_liquid_kan():
    sn = LatentSupernet(5, ("a", "b"), rank=2)
    ir = [("OP_LOOP", [("OP_CONST", 1)], [("OP_ASSIGN", "x", [("OP_CONST", 0)])])]
    g = build_execution_graph_from_ir(ir, sn, [], loop_max_unroll=3, loop_num_basis=4)
    loop_mod = g.node_modules["loop_0"]
    assert isinstance(loop_mod, LiquidKANNode)
    assert g.dag.nodes["loop_0"].get("body_ir") is not None
    x = torch.randn(2, 5)
    y = g(x)
    assert y.shape == (2, 5)


def test_wire_while_ax():
    reset_parser()
    ir = ast_to_ir(parse_ax("while (1) { k = 0; }"))
    sn = LatentSupernet(4, ("e1", "e2"), rank=2)
    g = wire_execution_graph(ir, sn, [], loop_max_unroll=2)
    assert any(g.node_kind(n) == "loop" for n in g.topo_names)
