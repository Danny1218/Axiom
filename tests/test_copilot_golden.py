"""Golden-task acceptance for Semantic Copilot (offline fake experts + evaluate_program)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from axiom.compiler.parser import reset_parser
from axiom.copilot.benchmarks import BenchmarkTask, default_neg_mse_score_fn, metric_success
from axiom.copilot.evaluator import evaluate_program
from axiom.copilot.models import ProgramCandidate
from axiom.experts.base import ExpertDraftRequest

_GOLDEN_JSON = Path(__file__).resolve().parent / "fixtures" / "copilot_golden_tasks.json"


def _load_golden_families() -> List[Dict[str, Any]]:
    raw = json.loads(_GOLDEN_JSON.read_text(encoding="utf-8"))
    families = raw.get("families")
    if not isinstance(families, list):
        raise ValueError("copilot_golden_tasks.json must contain a 'families' array.")
    return families


def _evaluate_ax(
    source: str,
    *,
    mode: str,
    example_input_rows: List[Dict[str, Any]],
    expected_rows: List[Dict[str, Any]],
    max_unroll: int = 8,
):
    return evaluate_program(
        ProgramCandidate(source=source, id="golden"),
        mode=mode,  # type: ignore[arg-type]
        input_rows=example_input_rows,
        expected_rows=expected_rows,
        score_fn=default_neg_mse_score_fn(),
        max_unroll=max_unroll,
    )


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


@pytest.mark.parametrize("family", _load_golden_families(), ids=lambda f: str(f["id"]))
def test_copilot_golden_family_passes(family: Dict[str, Any]) -> None:
    """Each family: goal-shaped reference .ax must compile and meet predict metrics."""
    family_id = str(family["id"])
    tasks = family.get("tasks")
    assert isinstance(tasks, list) and tasks, f"family {family_id} has no tasks"
    for task in tasks:
        tid = str(task["id"])
        mode = str(task.get("evaluation_mode", "compile_only"))
        rows_in = [dict(r) for r in task.get("example_input_rows") or []]
        rows_exp = [dict(r) for r in task.get("expected_rows") or []]
        ref = str(task["reference_ax"])
        report = _evaluate_ax(ref, mode=mode, example_input_rows=rows_in, expected_rows=rows_exp)
        assert report.success, f"{family_id}/{tid}: evaluate_program failed: {report.failures}"
        assert report.compile_stage_reached in ("block", "predict")
        if mode == "predict_rows":
            mpm = task.get("metric_pass_min")
            mpm_t = (str(mpm[0]), float(mpm[1])) if mpm is not None else None
            bench_task = BenchmarkTask(
                id=tid,
                title=str(task.get("title", tid)),
                goal=str(task.get("goal", tid)),
                domain_context="",
                evaluation_mode="predict_rows",
                example_input_rows=tuple(rows_in),
                expected_rows=tuple(rows_exp),
                metric_pass_min=mpm_t,
            )
            assert metric_success(bench_task, report), f"{family_id}/{tid}: metric_success failed: {report.metrics}"


def test_copilot_golden_invalid_syntax_repair_path() -> None:
    """Broken draft fails compile; repaired reference passes predict_rows."""
    families = _load_golden_families()
    repair = next(f for f in families if f["id"] == "invalid_syntax_repair")
    task = repair["tasks"][0]
    broken = str(task["broken_ax"])
    ref = str(task["reference_ax"])
    rows_in = [dict(r) for r in task["example_input_rows"]]
    rows_exp = [dict(r) for r in task["expected_rows"]]
    bad = _evaluate_ax(broken, mode="compile_only", example_input_rows=[], expected_rows=[])
    assert bad.success is False
    assert bad.compile_stage_reached == "parse"
    good = _evaluate_ax(ref, mode="predict_rows", example_input_rows=rows_in, expected_rows=rows_exp)
    assert good.success is True


def test_copilot_golden_dispatch_expert_emits_reference_ax() -> None:
    """BenchmarkDispatchExpert path: draft → evaluate_program for bundled double-x reference."""
    from axiom.copilot.benchmarks import BenchmarkDispatchExpert

    expert = BenchmarkDispatchExpert(
        sources={"golden_double_x": "y = x + x;\n"},
    )
    req = ExpertDraftRequest(
        goal="double x",
        context={"benchmark_task_id": "golden_double_x"},
    )
    resp = expert.draft_program(req)
    report = evaluate_program(
        ProgramCandidate(source=resp.ax_source, id="golden_double_x"),
        mode="predict_rows",
        input_rows=[{"x": 2.0}],
        expected_rows=[{"y": 4.0}],
        score_fn=default_neg_mse_score_fn(),
    )
    assert report.success
    assert report.metrics.get("neg_mse", -1.0) >= -1e-6


def test_copilot_golden_conformance_summary_by_family(capsys: pytest.CaptureFixture[str]) -> None:
    """Single entry point: print pass/fail by family (used by CI smoke)."""
    results: Dict[str, bool] = {}
    for family in _load_golden_families():
        fid = str(family["id"])
        ok = True
        for task in family.get("tasks") or []:
            mode = str(task.get("evaluation_mode", "compile_only"))
            rows_in = [dict(r) for r in task.get("example_input_rows") or []]
            rows_exp = [dict(r) for r in task.get("expected_rows") or []]
            report = _evaluate_ax(
                str(task["reference_ax"]),
                mode=mode,
                example_input_rows=rows_in,
                expected_rows=rows_exp,
            )
            if not report.success:
                ok = False
        results[fid] = ok
    for fid, passed in results.items():
        print(f"GOLDEN_FAMILY {fid}: {'PASS' if passed else 'FAIL'}")
    assert all(results.values()), results
