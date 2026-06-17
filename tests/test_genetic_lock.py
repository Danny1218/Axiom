"""Genetic lock for ``.axb`` bundles (AES-256-CTR on neural weights)."""

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("cryptography")

from axiom.cli import main
from axiom.compiler.deserializer import load_bundle
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.compiler.serializer import AXB_BUNDLE_VERSION, interpreted_block_topology_dict, save_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.security.genetic_lock import (
    BundleLockError,
    BundleUnlockError,
    lock_bundle_file,
    unlock_payload,
)


def _make_block() -> InterpretedBlock:
    reset_parser()
    ir = ast_to_ir(parse_ax("y = neural([1.0, 2.0]);"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    return InterpretedBlock(ir, abi, abi_widths=aw)


def test_unlock_payload_pass_through_without_lock():
    payload = {"version": 1, "topology": {}, "neural_weights": None}
    assert unlock_payload(payload) is payload


def test_lock_bundle_file_missing_input(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as ei:
        lock_bundle_file(tmp_path / "missing.axb", tmp_path / "out.axb", "env-secret")
    msg = str(ei.value)
    assert "train_portfolio" in msg
    assert "gitignored" in msg.lower()


def test_roundtrip_unlocked_bundle(tmp_path: Path):
    b0 = _make_block()
    p = tmp_path / "u.axb"
    save_bundle(b0, p)
    b1 = load_bundle(p)
    x = torch.zeros(1, 16)
    with torch.no_grad():
        assert torch.allclose(b0(x), b1(x))


def test_env_secret_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AXIOM_BUNDLE_SECRET", "unit-test-secret-xyz")
    b0 = _make_block()
    p = tmp_path / "e.axb"
    save_bundle(b0, p, lock_mode="env-secret")
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["lock"]["encrypted"] is True
    assert raw["lock"]["lock_mode"] == "env-secret"
    assert raw["neural_weights"] is None
    b1 = load_bundle(p)
    x = torch.zeros(1, 16)
    with torch.no_grad():
        assert torch.allclose(b0(x), b1(x))


def test_env_secret_wrong_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AXIOM_BUNDLE_SECRET", "correct")
    b0 = _make_block()
    p = tmp_path / "locked.axb"
    save_bundle(b0, p, lock_mode="env-secret")
    monkeypatch.setenv("AXIOM_BUNDLE_SECRET", "wrong")
    with pytest.raises(BundleUnlockError, match="fingerprint"):
        load_bundle(p)


def test_env_secret_missing_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AXIOM_BUNDLE_SECRET", "s")
    b0 = _make_block()
    p = tmp_path / "l.axb"
    save_bundle(b0, p, lock_mode="env-secret")
    monkeypatch.delenv("AXIOM_BUNDLE_SECRET", raising=False)
    with pytest.raises(BundleUnlockError, match="AXIOM_BUNDLE_SECRET"):
        load_bundle(p)


def test_host_lock_roundtrip(tmp_path: Path):
    b0 = _make_block()
    p = tmp_path / "h.axb"
    save_bundle(b0, p, lock_mode="host")
    b1 = load_bundle(p)
    x = torch.zeros(1, 16)
    with torch.no_grad():
        assert torch.allclose(b0(x), b1(x))


def test_lock_bundle_file_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AXIOM_BUNDLE_SECRET", "cli-secret")
    b0 = _make_block()
    src = tmp_path / "a.axb"
    dst = tmp_path / "b.axb"
    save_bundle(b0, src)
    lock_bundle_file(src, dst, "env-secret")
    b1 = load_bundle(dst)
    x = torch.zeros(1, 16)
    with torch.no_grad():
        assert torch.allclose(b0(x), b1(x))


def test_apply_lock_topology_still_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AXIOM_BUNDLE_SECRET", "k")
    b0 = _make_block()
    p = tmp_path / "t.axb"
    save_bundle(b0, p, lock_mode="env-secret")
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert obj["topology"]["kind"] == "interpreted_block"
    assert "ir" in obj["topology"]


def test_device_lock_save_fails_without_cuda_cpu_only(tmp_path: Path):
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present; cannot test CPU-only device-lock failure")
    b0 = _make_block()
    with pytest.raises(BundleLockError, match="CUDA"):
        save_bundle(b0, tmp_path / "d.axb", lock_mode="device")


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA required",
)
def test_device_lock_roundtrip_cuda(tmp_path: Path):
    b0 = _make_block()
    p = tmp_path / "cuda.axb"
    save_bundle(b0, p, lock_mode="device")
    b1 = load_bundle(p)
    x = torch.zeros(1, 16).cuda()
    with torch.no_grad():
        o0 = b0(x.cpu())
        o1 = b1(x.cpu())
    assert torch.allclose(o0, o1)


def test_cli_lock_bundle_help_exits_ok():
    with pytest.raises(SystemExit) as exc:
        main(["lock-bundle", "--help"])
    assert exc.value.code == 0


def test_backward_compat_payload_v1_shape(tmp_path: Path):
    """Legacy payload dict without ``lock`` still loads."""
    b0 = _make_block()
    abi = dict(b0.abi)
    p = tmp_path / "legacy.axb"
    payload = {
        "version": AXB_BUNDLE_VERSION,
        "topology": interpreted_block_topology_dict(b0),
        "abi_widths": {str(k): int(v) for k, v in b0.abi_widths.items()},
        "neural_weights": b0.neural_registry.state_dict(),
    }
    torch.save(payload, str(p))
    b1 = load_bundle(p)
    assert b1.abi == abi
