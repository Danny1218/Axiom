"""Minimum viable happy path: parse → compile → bundle → predict → explain → report."""

from __future__ import annotations

from pathlib import Path

import pytest

import axiom
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


def test_smoke_parse_compile_predict_explain_report(tmp_path: Path) -> None:
    """One chained path a new contributor can mirror without live LLM or optional servers."""
    source = "y = x * 2.0;"
    tree = parse_ax(source)
    ir = ast_to_ir(tree)
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    block.eval()

    axb = tmp_path / "smoke.axb"
    save_bundle(block, axb)
    model = axiom.load(axb)

    pred = model.predict({"x": 2.0})
    assert pred["y"] == pytest.approx(4.0)

    trace = model.explain({"x": 2.0})
    assert isinstance(trace, dict)
    assert "trace" in trace or "steps" in trace or trace

    report_path = tmp_path / "smoke_report.html"
    model.export_report({"x": 2.0}, str(report_path))
    html = report_path.read_text(encoding="utf-8")
    assert "Axiom Glass Box Execution Report" in html
    assert "4.0" in html
