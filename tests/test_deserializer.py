"""Round-trip and coverage for `compiler.deserializer.load_execution_bundle`."""

from pathlib import Path

import pytest
import torch

from axiom.compiler.deserializer import _ir_from_json, load_bundle, load_execution_bundle
from axiom.compiler.flow import wire_execution_graph
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.compiler.serializer import execution_topology_to_dict, save_bundle, save_execution_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.supernet import LatentSupernet


def test_ir_from_json_nested_assign():
    nested = ["OP_ASSIGN", "x", [["OP_CONST", 1]]]
    assert _ir_from_json(nested) == ("OP_ASSIGN", "x", (("OP_CONST", 1),))


def test_topology_dict_has_supernet_router_loop_config():
    reset_parser()
    ir = ast_to_ir(parse_ax("while (1) { k = 0; }"))
    sn = LatentSupernet(4, ("a", "b"), rank=3)
    g = wire_execution_graph(ir, sn, [], loop_max_unroll=4, loop_num_basis=5)
    d = execution_topology_to_dict(g)
    assert d["supernet_config"]["dim"] == 4
    assert d["supernet_config"]["adapter_names"] == ["a", "b"]
    assert d["supernet_config"]["rank"] == 3
    assert d["router_config"]["num_iters"] == 8
    assert d["loop_config"]["max_unroll"] == 4
    assert d["loop_config"]["num_basis"] == 5
    assert "abi" in d and isinstance(d["abi"], dict)
    assert "abi_widths" in d and isinstance(d["abi_widths"], dict)
    loop_node = next(n for n in d["nodes"] if n.get("kind") == "loop")
    assert loop_node["loop_max_unroll"] == 4
    assert loop_node["loop_num_basis"] == 5


def test_load_execution_bundle_conditional_and_loop_matches_original(tmp_path):
    reset_parser()
    ax = """
x = 1;
if (x > 0) {
  y = 1;
}
i = 1;
while (i > 0) {
  i = 0;
}
"""
    ir = ast_to_ir(parse_ax(ax))
    sn = LatentSupernet(5, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], loop_max_unroll=3, loop_num_basis=4)
    torch.manual_seed(42)
    x = torch.randn(4, 5)
    with torch.no_grad():
        y0, s0, z0 = g(x)
    prefix = tmp_path / "bundle"
    save_execution_bundle(g, prefix, ir=ir)
    assert Path(str(prefix) + "_topology.json").is_file()
    g2 = load_execution_bundle(prefix)
    with torch.no_grad():
        y1, s1, z1 = g2(x)
    assert torch.allclose(y0, y1, atol=0, rtol=0)
    assert set(s0.keys()) == set(s1.keys())
    for k in s0:
        assert torch.allclose(s0[k], s1[k], atol=0, rtol=0)
    assert set(z0.keys()) == set(z1.keys())
    for k in z0:
        assert torch.allclose(z0[k], z1[k], atol=0, rtol=0)


def test_load_execution_bundle_stmt_only(tmp_path):
    reset_parser()
    ir = ast_to_ir(parse_ax("x = 1;"))
    sn = LatentSupernet(4, ("p", "q"), rank=2)
    g = wire_execution_graph(ir, sn, [])
    prefix = tmp_path / "stmt"
    save_execution_bundle(g, prefix, ir=ir)
    g2 = load_execution_bundle(prefix)
    x = torch.randn(2, 4)
    o1, sh1, sg1 = g(x)
    o2, sh2, sg2 = g2(x)
    assert torch.allclose(o1, o2)
    assert set(sh1.keys()) == set(sh2.keys())
    for k in sh1:
        assert torch.allclose(sh1[k], sh2[k], atol=0, rtol=0)
    assert set(sg1.keys()) == set(sg2.keys())
    for k in sg1:
        assert torch.allclose(sg1[k], sg2[k], atol=0, rtol=0)


def test_load_bundle_restores_neural_weights(tmp_path):
    reset_parser()
    ir = ast_to_ir(parse_ax("z = neural([0.5]);"))
    gabi = extract_global_abi(ir, max_vars=16)
    gaw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(ir, gabi, abi_widths=gaw)
    opt = torch.optim.SGD(b.parameters(), lr=0.1)
    for _ in range(5):
        x = torch.randn(2, 16)
        opt.zero_grad()
        b(x).sum().backward()
        opt.step()
    p = tmp_path / "n.axb"
    save_bundle(b, p)
    b2 = load_bundle(p)
    x2 = torch.randn(4, 16)
    with torch.no_grad():
        assert torch.allclose(b(x2), b2(x2), atol=0, rtol=0)


def test_load_execution_bundle_raises_missing_json(tmp_path):
    p = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        load_execution_bundle(p)


def test_conditional_node_carries_expert_names_in_topology_json(tmp_path):
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], router_iters=5, router_eps=0.05)
    d = execution_topology_to_dict(g)
    cnode = next(n for n in d["nodes"] if n.get("kind") == "conditional")
    assert cnode["expert_then"] == "then_0"
    assert cnode["expert_else"] == "else_0"
    assert d["router_config"]["num_iters"] == 5
    assert d["router_config"]["epsilon"] == pytest.approx(0.05)
    prefix = tmp_path / "c"
    save_execution_bundle(g, prefix, ir=ir)
    g2 = load_execution_bundle(prefix)
    x = torch.randn(1, 5)
    o1, sh1, sg1 = g(x)
    o2, sh2, sg2 = g2(x)
    assert torch.allclose(o1, o2)
    assert set(sh1.keys()) == set(sh2.keys())
    for k in sh1:
        assert torch.allclose(sh1[k], sh2[k], atol=0, rtol=0)
    assert set(sg1.keys()) == set(sg2.keys())
    for k in sg1:
        assert torch.allclose(sg1[k], sg2[k], atol=0, rtol=0)
