"""Macro inlining: user functions expand to flat IR and execute on the graph."""

import pytest
import torch

from axiom.compiler.flow import wire_execution_graph
from axiom.compiler.ir import (
    ast_to_ir,
    expand_function_calls,
    parse_program,
    RESERVED_MATH_BUILTINS,
    RESERVED_REDUCTION_BUILTINS,
)
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.engine.supernet import LatentSupernet


def test_expand_nested_calls_mangles_names():
    reset_parser()
    src = """
def add(a, b) { return a + b; }
def dbl(x) { return add(x, x); }
y = dbl(3);
"""
    tree = parse_ax(src)
    funcs, main = parse_program(tree)
    assert "add" in funcs and "dbl" in funcs
    ir = expand_function_calls(main, funcs)
    assert any(s[0] == "OP_ASSIGN" and str(s[1]).startswith("_inline_") for s in ir)
    assert not any(
        isinstance(s, tuple) and s and s[0] == "OP_CALL" for s in _all_stmts_flat(ir)
    )


def _all_stmts_flat(stmts):
    for s in stmts:
        yield s
        if isinstance(s, tuple) and s[0] == "OP_CONDITIONAL":
            yield from _all_stmts_flat(s[2])
            yield from _all_stmts_flat(s[3])
        elif isinstance(s, tuple) and s[0] == "OP_LOOP":
            yield from _all_stmts_flat(s[2])


def test_wire_graph_with_inlined_function_runs():
    reset_parser()
    ir = ast_to_ir(
        parse_ax("def add(a, b) { return a + b; } x = add(1.0, 2.0);")
    )
    sn = LatentSupernet(8, ("latent_0", "latent_1"), rank=2)
    g = wire_execution_graph(ir, sn, [])
    x = torch.zeros(2, 8)
    y, _, _ = g(x)
    xc = g.abi["x"]
    assert torch.allclose(y[:, xc], torch.tensor([3.0, 3.0]))


def test_reserved_reduction_names_cannot_be_user_functions():
    for name in RESERVED_REDUCTION_BUILTINS:
        with pytest.raises(ValueError, match="reserved"):
            ast_to_ir(parse_ax(f"def {name}(x) {{ return x; }}"))


def test_reserved_math_names_cannot_be_user_functions():
    for name in RESERVED_MATH_BUILTINS:
        with pytest.raises(ValueError, match="reserved"):
            ast_to_ir(parse_ax(f"def {name}(x) {{ return x; }}"))


def test_double_call_two_independent_mangles():
    reset_parser()
    ir = ast_to_ir(
        parse_ax(
            "def id(z) { return z; } a = id(1); b = id(2);"
        )
    )
    names = [s[1] for s in ir if s[0] == "OP_ASSIGN" and str(s[1]).startswith("_inline_id_")]
    assert len(set(names)) >= 4  # args + rets for two calls
