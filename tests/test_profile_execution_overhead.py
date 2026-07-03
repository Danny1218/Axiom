"""``scripts/profile_execution_overhead.py`` JSON contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_profile_execution_overhead_json_contract(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "overhead.json"
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "profile_execution_overhead.py"), "--json-out", str(out), "--repeats", "2", "--warmup", "0"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    names = {b["name"] for b in payload["benchmarks"]}
    assert {
        "parse_to_ir",
        "interpreted_block_forward",
        "loop_block_forward",
        "conditional_graph_forward",
        "torch_compile_backend",
    }.issubset(names)
    compile_entry = next(b for b in payload["benchmarks"] if b["name"] == "torch_compile_backend")
    assert "backend" in compile_entry
    assert "errors" in compile_entry
    assert isinstance(compile_entry["errors"], dict)
    timed = [b for b in payload["benchmarks"] if b["name"] != "torch_compile_backend"]
    for entry in timed:
        assert "p50_seconds" in entry
        assert "p95_seconds" in entry
        assert "mean_seconds" in entry
