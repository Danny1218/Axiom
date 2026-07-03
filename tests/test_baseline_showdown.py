"""Smoke test for baseline_showdown harness (fast subset)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from benchmarks.baseline_showdown.run_showdown import run_showdown

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_JSON = ROOT / "docs" / "evidence" / "baseline_showdown.json"


def test_baseline_showdown_smoke_two_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_json = tmp_path / "showdown.json"
    out_md = tmp_path / "showdown.md"
    monkeypatch.setattr(
        "benchmarks.baseline_showdown.run_showdown.EVIDENCE_JSON",
        out_json,
    )
    monkeypatch.setattr(
        "benchmarks.baseline_showdown.run_showdown.EVIDENCE_MD",
        out_md,
    )
    t0 = time.perf_counter()
    summary = run_showdown(task_ids=["affine_slope_2", "sabotage_sin"])
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0
    assert out_json.is_file() and out_md.is_file()
    assert summary["benchmark"] == "baseline_showdown"
    assert len(summary["tasks"]) == 2
    ax = next(c for t in summary["tasks"] for c in t["contenders"] if c["name"] == "axiom")
    assert "declined" in ax
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
