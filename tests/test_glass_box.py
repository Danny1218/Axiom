"""Phase 20: Glass Box helpers, router weights in signals, Graphviz DAG export."""

import torch
import torch.nn as nn

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax, reset_parser
from compiler.serializer import save_execution_bundle
from engine.inference import AxiomRunner
from engine.meta_compiler import MetaCompiler
from engine.router import SinkhornRouter
from engine.supernet import LatentSupernet
from tools.glass_box import (
    execution_graph_to_graphviz,
    routing_trace_entries,
    tensor_preview_dict,
)


def test_conditional_signals_include_entropy_and_weights():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], mutation_entropy_norm_threshold=0.99)
    x = torch.randn(3, 5)
    _, _, sig = g(x)
    assert "cond_0" in sig and sig["cond_0"].shape == ()
    assert "cond_0_weights" in sig
    assert sig["cond_0_weights"].shape[-1] == 2
    assert sig["cond_0_weights"].shape[0] == 3


def test_routing_trace_entries_matches_conditional_blocks():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")])
    x = torch.randn(2, 5)
    _, _, sig = g(x)
    rows = routing_trace_entries(g, sig)
    assert len(rows) == 1
    assert rows[0]["block"] == "cond_0"
    assert rows[0]["expert_then"] == "then_0"
    assert rows[0]["normalized_routing_entropy"] is not None
    assert len(rows[0]["mean_router_weights_then_else"]) == 2


def test_execution_graph_to_graphviz_color_tags():
    reset_parser()
    ir = ast_to_ir(parse_ax("x=1; if (x>0) { y=1; } else { y=2; }"))
    sn = LatentSupernet(4, ("t", "e"), rank=2)
    sn.set_masks({"t": 1.0, "e": 1.0})
    g = wire_execution_graph(ir, sn, [("t", "e")])
    src = execution_graph_to_graphviz(g).source
    assert "OP_CONDITIONAL" in src or "cond_0" in src
    assert "#fff2a8" in src
    assert "#b8f5b8" in src


def test_graphviz_loop_node_is_blue():
    reset_parser()
    ir = ast_to_ir(parse_ax("while (1) { k = 0; }"))
    sn = LatentSupernet(4, ("a", "b"), rank=2)
    g = wire_execution_graph(ir, sn, [], loop_max_unroll=2)
    src = execution_graph_to_graphviz(g).source
    assert "OP_LOOP" in src
    assert "#9ec5ff" in src


def test_tensor_preview_dict_shape():
    t = torch.randn(2, 4)
    p = tensor_preview_dict(t, max_elems=4)
    assert p["shape"] == [2, 4]
    assert len(p["flat_head"]) == 4


def test_meta_compiler_ignores_nonscalar_signal_tensors():
    sn = LatentSupernet(4, ("a", "b", "c"), rank=2)
    sn.set_masks({"a": 1.0, "b": 1.0})
    r = SinkhornRouter(4, 2, mutation_entropy_norm_threshold=0.5)
    nn.init.zeros_(r.proj.weight)
    nn.init.zeros_(r.proj.bias)
    _, ent = r(torch.randn(6, 4))
    mc = MetaCompiler(sn)
    mixed = {"cond_0": ent, "cond_0_weights": torch.randn(3, 2), "junk": torch.randn(2, 2)}
    names = mc.react_to_signals(mixed, sn, max_unmasks=1, block_thresholds={"cond_0": 0.5})
    assert names == ["c"]


def test_runner_predict_with_signals_after_bundle_roundtrip(tmp_path):
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")])
    prefix = tmp_path / "gb"
    save_execution_bundle(g, prefix, ir=ir)
    from compiler.deserializer import load_execution_bundle

    g2 = load_execution_bundle(prefix)
    runner = AxiomRunner(g2)
    out, sig = runner.predict_with_signals({"a": 1.0}, device="cpu")
    assert out.shape == (1, 5)
    assert "cond_0" in sig and "cond_0_weights" in sig
