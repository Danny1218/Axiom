"""Smoke test for Titanic guarded expert wrap (no sklearn in test path)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom.api import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.expert_registry import ExpertRuntimeRegistry
from axiom.verify.interval import certify

ROOT = Path(__file__).resolve().parents[1]
AX_PATH = ROOT / "examples" / "titanic_guarded.ax"


def _stub_handler(_name: str, features: list[float]) -> float:
    return min(1.0, max(0.0, 0.3 + 0.001 * sum(features)))


def _compile_stub() -> AxiomModel:
    reset_parser()
    ir = ast_to_ir(parse_ax_file(AX_PATH))
    abi = extract_global_abi(ir, max_vars=32)
    aw = extract_abi_widths(ir, max_vars=32)
    reg = ExpertRuntimeRegistry()
    reg.register("tabular_model", _stub_handler)
    block = InterpretedBlock(ir, abi, abi_widths=aw, max_unroll=8, expert_registry=reg)
    block.eval()
    return AxiomModel(block)


def test_guarded_enforces_clamp_on_rule_region() -> None:
    model = _compile_stub()
    row = {"Sex": 0.0, "Pclass": 3.0, "Fare": 10.0, "Age": 30.0}
    trace = model.explain(row)
    assert float(trace["raw_prob"]) > 0.15
    assert float(trace["survived_prob"]) <= 0.15 + 1e-6


def test_guarded_passthrough_outside_rule_region() -> None:
    model = _compile_stub()
    row = {"Sex": 1.0, "Pclass": 1.0, "Fare": 50.0, "Age": 30.0}
    trace = model.explain(row)
    assert float(trace["survived_prob"]) == pytest.approx(float(trace["raw_prob"]))


def test_certificate_proves_clamp(tmp_path: Path) -> None:
    reset_parser()
    ir = ast_to_ir(parse_ax_file(AX_PATH))
    abi = extract_global_abi(ir, max_vars=32)
    aw = extract_abi_widths(ir, max_vars=32)
    block = InterpretedBlock(ir, abi, abi_widths=aw, max_unroll=8)
    cert = certify(
        block,
        {
            "Pclass": (3.0, 3.0),
            "Sex": (0.0, 0.0),
            "Age": (18.0, 100.0),
            "Fare": (0.0, 600.0),
        },
        node_bounds={"tabular_model": (0.0, 1.0)},
        source_path=AX_PATH,
    )
    payload = cert.to_dict()
    assert payload["status"] == "ok"
    assert float(payload["proven_output_bounds"]["survived_prob"][1]) <= 0.15 + 1e-9
    out = tmp_path / "cert.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    assert json.loads(out.read_text(encoding="utf-8"))["axiom_version"]
