"""Interval analysis certificates — arithmetic, branches, fuzz soundness."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import pytest
import torch

from axiom.api import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.expert_registry import ExpertRuntimeRegistry
from axiom.verify.interval import Interval, certify, eval_expr_interval, exec_stmt_interval

Bounds = Tuple[float, float]


def _block(source: str) -> InterpretedBlock:
    reset_parser()
    ir = ast_to_ir(parse_ax(source))
    abi = extract_global_abi(ir, max_vars=32)
    aw = extract_abi_widths(ir, max_vars=32)
    return InterpretedBlock(ir, abi, abi_widths=aw, max_unroll=8)


def _concrete(model: AxiomModel, row: Dict[str, float], out_key: str) -> float:
    return float(model.predict(row)[out_key])


def test_interval_add_mul_div():
    env: Dict[str, Bounds] = {"x": (1.0, 2.0), "y": (3.0, 4.0)}
    unsupported: List[str] = []
    ir = [
        ("OP_LOAD", "x"),
        ("OP_LOAD", "y"),
        ("OP_ADD",),
    ]
    iv = eval_expr_interval(env, ir, node_bounds={}, unsupported=unsupported)
    assert iv.lo == pytest.approx(4.0)
    assert iv.hi == pytest.approx(6.0)


def test_interval_div_by_zero_interval_is_unknown():
    env = {"a": (1.0, 2.0), "b": (-1.0, 1.0)}
    unsupported: List[str] = []
    ir = [("OP_LOAD", "a"), ("OP_LOAD", "b"), ("OP_DIV",)]
    iv = eval_expr_interval(env, ir, node_bounds={}, unsupported=unsupported)
    assert iv.lo == float("-inf")
    assert iv.hi == float("inf")


def test_branch_union_soundness():
    """One branch alone would be tighter but unsound without union."""
    source = (
        "if (x > 0.0) {\n"
        "  y = 10.0;\n"
        "} else {\n"
        "  y = -10.0;\n"
        "}\n"
    )
    block = _block(source)
    cert = certify(block, {"x": (-1.0, 1.0)})
    lo, hi = cert.proven_output_bounds["y"]
    assert lo <= -10.0 + 1e-9
    assert hi >= 10.0 - 1e-9


def test_expert_unknown_without_assumption():
    source = 'y = expert("m", [x]);'
    block = _block(source)
    cert = certify(block, {"x": (0.0, 1.0)})
    lo, hi = cert.proven_output_bounds["y"]
    assert lo == float("-inf") and hi == float("inf")


def test_expert_bounded_with_node_bounds():
    source = 'y = expert("m", [x]);'
    block = _block(source)
    cert = certify(block, {"x": (0.0, 1.0)}, node_bounds={"m": (0.0, 1.0)})
    assert cert.proven_output_bounds["y"] == (0.0, 1.0)


def test_loop_refused():
    source = (
        "step = 0.0;\n"
        "while (step < 2.0) {\n"
        "  step = step + 1.0;\n"
        "}\n"
    )
    block = _block(source)
    cert = certify(block, {"step": (0.0, 0.0)})
    assert cert.status == "unsupported"
    assert "OP_LOOP" in cert.unsupported_ops


def test_neural_unknown():
    source = "y = neural([x]);"
    block = _block(source)
    cert = certify(block, {"x": (0.0, 1.0)})
    assert "OP_NEURAL" in cert.unsupported_ops[0]


@pytest.mark.parametrize(
    "source,out_key,region",
    [
        ("y = x + 1.0;", "y", {"x": (-2.0, 2.0)}),
        ("y = min(x, 0.5);", "y", {"x": (0.0, 1.0)}),
        (
            "if (x > 0.0) { y = x; } else { y = 0.0; }",
            "y",
            {"x": (-1.0, 1.0)},
        ),
    ],
)
def test_fuzz_soundness(source: str, out_key: str, region: Dict[str, Bounds]) -> None:
    block = _block(source)
    cert = certify(block, region)
    reg = ExpertRuntimeRegistry()
    model = AxiomModel(block)
    rng = random.Random(99)
    lo_r, hi_r = region["x"]
    for _ in range(2000):
        x = rng.uniform(lo_r, hi_r)
        val = _concrete(model, {"x": x}, out_key)
        blo, bhi = cert.proven_output_bounds[out_key]
        assert blo - 1e-6 <= val <= bhi + 1e-6
