"""Masked early return inside user functions (Phase 33 / Part C)."""

import pytest
import torch

from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.engine.block_executor import InterpretedBlock


def test_early_return_if_then_branch_batched():
    reset_parser()
    ir = ast_to_ir(
        parse_ax(
            "def pick(c) { if (c > 0) { return 1.0; } return 0.0; } x = pick(c);"
        )
    )
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    h = torch.zeros(4, 16)
    h[:, abi["c"]] = torch.tensor([1.0, -1.0, 0.0, 2.0])
    out = block(h)
    xc = abi["x"]
    assert torch.allclose(out[:, xc], torch.tensor([1.0, 0.0, 0.0, 1.0]))


def test_early_return_without_else():
    reset_parser()
    ir = ast_to_ir(
        parse_ax("def f(a) { if (a > 0) { return 10.0; } return 1.0; } y = f(a);")
    )
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    h = torch.zeros(3, 16)
    h[:, abi["a"]] = torch.tensor([-1.0, 3.0, 0.0])
    out = block(h)
    assert torch.allclose(out[:, abi["y"]], torch.tensor([1.0, 10.0, 1.0]))


def test_tail_return_only_still_simple_inline():
    reset_parser()
    ir = ast_to_ir(parse_ax("def add(a, b) { return a + b; } z = add(3, 4);"))
    assert any(
        isinstance(s, tuple) and s[0] == "OP_ASSIGN" and str(s[1]) == "z"
        for s in ir
    )
    assert any(
        isinstance(s, tuple) and s[0] == "OP_ASSIGN" and "ret" in str(s[1])
        for s in ir
    )


def test_return_inside_while_rejected():
    reset_parser()
    with pytest.raises(ValueError, match="while"):
        ast_to_ir(
            parse_ax("def bad() { while (1) { return 1; } return 0; } x = bad();")
        )


def test_while_without_return_in_body_allowed_with_early_return_elsewhere():
    reset_parser()
    ir = ast_to_ir(
        parse_ax(
            "def g(x) { if (x > 0) { return 2.0; } while (0) { x = x; } return 1.0; } r = g(x);"
        )
    )
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    h = torch.zeros(2, 16)
    h[:, abi["x"]] = torch.tensor([5.0, -1.0])
    out = block(h)
    assert torch.allclose(out[:, abi["r"]], torch.tensor([2.0, 1.0]))


def test_nested_if_early_return():
    reset_parser()
    ir = ast_to_ir(
        parse_ax(
            "def h(a, b) { if (a > 0) { if (b > 0) { return 3.0; } return 2.0; } return 1.0; } o = h(a, b);"
        )
    )
    abi = extract_global_abi(ir, max_vars=24)
    aw = extract_abi_widths(ir, max_vars=24)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    h = torch.zeros(4, 24)
    h[:, abi["a"]] = torch.tensor([-1.0, 1.0, 1.0, 1.0])
    h[:, abi["b"]] = torch.tensor([0.0, 0.0, 1.0, -1.0])
    out = block(h)
    assert torch.allclose(out[:, abi["o"]], torch.tensor([1.0, 2.0, 3.0, 2.0]))


def test_blend_assign_and_accum_grad():
    reset_parser()
    ir = ast_to_ir(
        parse_ax("def pick(c) { if (c > 0) { return 1.0; } return 0.0; } x = pick(c);")
    )
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    base = torch.zeros(2, 16)
    base[:, abi["c"]] = torch.tensor([1.0, -1.0])
    h = base.clone().requires_grad_(True)
    out = block(h)
    out[:, abi["x"]].sum().backward()
    assert h.grad is not None
