"""Phase 72: ``ExpertRuntimeRegistry`` and :class:`~axiom.api.AxiomModel` wiring for ``OP_EXPERT``."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import axiom
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.expert_call import ExpertRuntimeError
from axiom.engine.expert_registry import (
    ExpertRuntimeRegistry,
    collect_expert_backend_names_from_stmts,
    expert_runtime_wiring_sufficient,
    interpreted_block_ir_contains_expert,
)


def _expert_ir():
    reset_parser()
    return ast_to_ir(parse_ax('e = expert("demo", [x, 1.0]);'))


def test_registry_register_resolve_clear():
    r = ExpertRuntimeRegistry()
    assert r.resolve("a") is None
    r.register("a", lambda _n, f: float(f[0]) * 2)
    assert r.resolve("a") is not None
    assert float(r.resolve("a")("a", [3.0])) == pytest.approx(6.0)
    r.clear()
    assert r.resolve("a") is None


def test_collect_names_and_contains():
    ir = _expert_ir()
    names = collect_expert_backend_names_from_stmts(ir)
    assert names == frozenset({"demo"})
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    assert interpreted_block_ir_contains_expert(b) is True


def test_wiring_sufficient_fallback():
    ir = _expert_ir()
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(ir, abi, abi_widths=aw, expert_fallback=0.0)
    assert expert_runtime_wiring_sufficient(b) is True


def test_wiring_sufficient_handler():
    ir = _expert_ir()
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(ir, abi, abi_widths=aw, expert_handler=lambda _n, f: 1.0)
    assert expert_runtime_wiring_sufficient(b) is True


def test_wiring_sufficient_registry_complete():
    ir = _expert_ir()
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    r = ExpertRuntimeRegistry()
    r.register("demo", lambda _n, f: 1.0)
    b = InterpretedBlock(ir, abi, abi_widths=aw, expert_registry=r)
    assert expert_runtime_wiring_sufficient(b) is True


def test_wiring_insufficient_partial_registry():
    ir = _expert_ir()
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    r = ExpertRuntimeRegistry()
    r.register("other", lambda _n, f: 0.0)
    b = InterpretedBlock(ir, abi, abi_widths=aw, expert_registry=r)
    assert expert_runtime_wiring_sufficient(b) is False


def test_load_attach_registry_predict_explain(tmp_path: Path):
    ir = _expert_ir()
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    p = tmp_path / "e.axb"
    save_bundle(b, p)
    reg = ExpertRuntimeRegistry()
    reg.register("demo", lambda _n, f: float(f[0]) + float(f[1]))
    model = axiom.load(p, expert_registry=reg)
    out = model.predict({"x": 2.0})
    assert out["e"] == pytest.approx(3.0)
    tr = model.explain({"x": 2.0})
    assert tr["e"] == pytest.approx(3.0)
    assert tr["expert_calls"] == [{"op": "expert", "backend": "demo"}]


def test_set_expert_registry_from_dict(tmp_path: Path):
    ir = _expert_ir()
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    p = tmp_path / "e2.axb"
    save_bundle(b, p)
    model = axiom.load(p)
    model.set_expert_registry({"demo": lambda _n, f: 7.0})
    assert model.predict({"x": 0.0})["e"] == pytest.approx(7.0)
    model.set_expert_registry(None)
    with pytest.raises(ExpertRuntimeError):
        model.predict({"x": 0.0})


def test_registry_wins_over_handler_for_same_name():
    ir = _expert_ir()
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    r = ExpertRuntimeRegistry()
    r.register("demo", lambda _n, f: 99.0)
    b = InterpretedBlock(
        ir,
        abi,
        abi_widths=aw,
        expert_registry=r,
        expert_handler=lambda _n, f: -1.0,
    )
    h = torch.zeros(1, 16)
    h[0, abi["x"]] = 1.0
    out = b(h)
    assert float(out[0, abi["e"]].item()) == pytest.approx(99.0)
