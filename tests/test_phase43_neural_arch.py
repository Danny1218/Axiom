"""Phase 43: string literal arch for ``neural(expr, "kan"|"liquid"|"mlp")``."""

import torch

from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi, extract_neural_node_specs
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.engine.block_executor import InterpretedBlock, build_neural_module
from axiom.primitives.liquid_tensor import LiquidFeatureReadout


def test_neural_two_arg_ir_has_arch():
    reset_parser()
    ir = ast_to_ir(parse_ax('y = neural([1.0, 2.0], "kan");'))
    tup = ir[0][2][0]
    assert tup[0] == "OP_NEURAL" and tup[3] == "kan"


def test_neural_single_arg_defaults_mlp():
    reset_parser()
    ir = ast_to_ir(parse_ax("y = neural([1.0]);"))
    assert ir[0][2][0][3] == "mlp"


def test_extract_neural_node_specs_returns_width_and_arch():
    reset_parser()
    ir = ast_to_ir(parse_ax('a = neural([0.0, 0.0], "liquid");'))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    spec = extract_neural_node_specs(ir, aw)
    nid = next(iter(spec))
    w, arch = spec[nid]
    assert w == 2 and arch == "liquid"
    assert abi["a"] >= 0


def test_build_neural_module_kan_and_liquid():
    k = build_neural_module(3, "kan", max_unroll=4)
    assert hasattr(k, "kan") and hasattr(k, "readout")
    ell = build_neural_module(3, "liquid")
    assert isinstance(ell, LiquidFeatureReadout)
    mlp = build_neural_module(3, "mlp")
    x = torch.randn(5, 3)
    assert k(x).shape == (5, 1)
    assert ell(x).shape == (5, 1)
    assert mlp(x).shape == (5, 1)


def test_interpreted_block_uses_arch_from_source():
    reset_parser()
    ir = ast_to_ir(parse_ax('z = neural([1.0], "liquid");'))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    nid = next(iter(b.neural_registry.keys()))
    assert isinstance(b.neural_registry[nid], LiquidFeatureReadout)


def test_neural_arch_single_quotes():
    reset_parser()
    ir = ast_to_ir(parse_ax("y = neural([1.0], 'kan');"))
    assert ir[0][2][0][3] == "kan"
