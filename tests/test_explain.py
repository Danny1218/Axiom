"""Phase 41: ``InterpretedBlock(..., return_env=True)`` and ``AxiomModel.explain``."""

import pytest
import torch

import axiom
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock


def test_interpreted_block_forward_return_env():
    reset_parser()
    ir = ast_to_ir(parse_ax("y = x * 2.0;"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    h = torch.zeros(1, 16)
    h[0, abi["x"]] = 3.0
    out, env = b(h, return_env=True)
    assert out.shape == (1, 16)
    assert "x" in env and "y" in env
    assert float(env["y"][0].item()) == pytest.approx(6.0)


def test_interpreted_block_forward_no_env_default():
    reset_parser()
    ir = ast_to_ir(parse_ax("y = x * 2.0;"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    h = torch.zeros(1, 16)
    h[0, abi["x"]] = 2.0
    out = b(h)
    assert out.shape == (1, 16)
    assert not isinstance(out, tuple)


def test_interpreted_block_empty_ir_returns_empty_env():
    b = InterpretedBlock([], {}, abi_widths={})
    h = torch.randn(2, 8)
    out, env = b(h, return_env=True)
    assert torch.allclose(out, h)
    assert env == {}


def test_axiom_explain_single_row(tmp_path):
    reset_parser()
    ir = ast_to_ir(parse_ax("y = x * 2.0;"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    save_bundle(block, tmp_path / "e.axb")
    model = axiom.load(tmp_path / "e.axb")
    trace = model.explain({"x": 2.5})
    assert isinstance(trace["y"], float)
    assert trace["y"] == pytest.approx(5.0)
    assert isinstance(trace["x"], float)


def test_explain_rejects_non_dict(tmp_path):
    reset_parser()
    ir = ast_to_ir(parse_ax("y = x;"))
    abi = extract_global_abi(ir, max_vars=8)
    aw = extract_abi_widths(ir, max_vars=8)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    save_bundle(block, tmp_path / "x.axb")
    model = axiom.load(tmp_path / "x.axb")
    with pytest.raises(TypeError):
        model.explain([])  # type: ignore[arg-type]
