"""Smoke test for Titanic hybrid constraint audit."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from benchmarks.titanic_hybrid.run_hybrid_audit import run_audit

ROOT = Path(__file__).resolve().parents[1]


def test_titanic_hybrid_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_json = tmp_path / "titanic.json"
    out_md = tmp_path / "titanic.md"
    monkeypatch.setattr(
        "benchmarks.titanic_hybrid.run_hybrid_audit.EVIDENCE_JSON",
        out_json,
    )
    monkeypatch.setattr(
        "benchmarks.titanic_hybrid.run_hybrid_audit.EVIDENCE_MD",
        out_md,
    )
    monkeypatch.setattr("benchmarks.titanic_hybrid.run_hybrid_audit.EPOCHS", 3)
    t0 = time.perf_counter()
    summary = run_audit()
    elapsed = time.perf_counter() - t0
    assert elapsed < 120.0
    assert out_json.is_file() and out_md.is_file()
    ax = next(m for m in summary["models"] if m["name"] == "axiom_hybrid")
    assert ax["constraint_violations"] == 0
    assert summary["explain_sample"]["trace"]
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "titanic_hybrid"
