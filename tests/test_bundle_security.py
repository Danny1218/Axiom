"""Security and strict-mode regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from axiom.compiler.deserializer import load_bundle
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _inputs_to_tensor
from axiom.engine.strict import StrictInferenceError, validate_predict_inputs_strict
from axiom.security.bundle_trust import BundleTrustError, resolve_report_output_path
from axiom.security.genetic_lock import BundleUnlockError, apply_lock_to_payload, unlock_payload


def _simple_block() -> InterpretedBlock:
    reset_parser()
    ir = ast_to_ir(parse_ax("y = x * 2.0;"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    return InterpretedBlock(ir, abi, abi_widths=aw)


def test_v2_bundle_is_json_not_pickle(tmp_path: Path):
    b = _simple_block()
    p = tmp_path / "m.axb"
    save_bundle(b, p)
    raw = p.read_bytes()
    assert raw[:1] == b"{"
    loaded = load_bundle(p)
    x = torch.zeros(1, 16)
    with torch.no_grad():
        assert torch.allclose(b(x), loaded(x))


def test_legacy_pickle_requires_trust(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    b = _simple_block()
    p = tmp_path / "legacy.axb"
    payload = {
        "version": 1,
        "topology": {
            "kind": "interpreted_block",
            "ir": [["OP_ASSIGN", "y", [["OP_LOAD", "x"], ["OP_CONST", 2.0], ["OP_MUL"]]]],
            "abi": {"x": 0, "y": 1},
            "max_unroll": 8,
        },
        "abi_widths": {"x": 1, "y": 1},
        "neural_weights": None,
    }
    torch.save(payload, str(p))
    monkeypatch.delenv("AXIOM_TRUST_BUNDLE", raising=False)
    with pytest.raises(BundleTrustError):
        load_bundle(p)
    loaded = load_bundle(p, trusted=True)
    assert loaded.abi["x"] == 0


def test_locked_bundle_tamper_raises(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("cryptography")
    monkeypatch.setenv("AXIOM_BUNDLE_SECRET", "tamper-test")
    payload: dict = {
        "version": 2,
        "topology": {"kind": "interpreted_block", "ir": [], "abi": {}, "max_unroll": 8},
        "abi_widths": {},
        "neural_weights": None,
    }
    apply_lock_to_payload(payload, "env-secret")
    payload["lock"]["ciphertext_hex"] = "00" + payload["lock"]["ciphertext_hex"][2:]
    with pytest.raises(BundleUnlockError, match="tampered"):
        unlock_payload(payload)


def test_report_path_sandbox_rejects_escape(tmp_path: Path):
    sandbox = tmp_path / "reports"
    sandbox.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        resolve_report_output_path("../outside.html", sandbox)
    out = resolve_report_output_path("run/report.html", sandbox)
    assert str(out).startswith(str(sandbox.resolve()))


def test_strict_missing_abi_input():
    with pytest.raises(StrictInferenceError, match="missing"):
        validate_predict_inputs_strict({"x": 1.0}, {"x": 0, "y": 1})


def test_strict_unknown_input_key():
    with pytest.raises(StrictInferenceError, match="unknown"):
        validate_predict_inputs_strict({"x": 1.0, "z": 9.0}, {"x": 0})


def test_strict_divide_by_zero():
    reset_parser()
    ir = ast_to_ir(parse_ax("y = x / 0.0;"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw, strict=True)
    h = torch.zeros(1, 16)
    h[0, abi["x"]] = 3.0
    with pytest.raises(StrictInferenceError, match="division by zero"):
        block(h)


def test_strict_index_out_of_range():
    reset_parser()
    ir = ast_to_ir(parse_ax("y = arr[5];"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw, strict=True)
    h = torch.zeros(1, 16)
    with pytest.raises(StrictInferenceError, match="index out of range"):
        block(h)


def test_lenient_default_unchanged():
    row = {"x": 2.0}
    h = _inputs_to_tensor(row, {"x": 0, "y": 1}, 16, device=torch.device("cpu"), dtype=torch.float32)
    assert h.shape == (1, 16)
