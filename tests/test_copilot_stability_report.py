"""Phase 83 / 83b: stability report over copilot artifact dirs and pipeline JSON."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom.cli import main
from axiom.copilot.artifacts import BEST_AX_NAME, SEARCH_REPORT_JSON_NAME
from axiom.copilot.stability_report import (
    DEFAULT_NEAR_NEG_MSE,
    build_aggregate,
    collect_run_snapshots,
    collect_stability_report,
    format_discovery_failure_message,
    ingest_path,
    resolve_input_paths,
    stability_report_to_dict,
)


def _minimal_search_report(
    *,
    neg_mse: float | None = None,
    success: bool = True,
    mode: str = "predict_rows",
    convergence_reason: str = "metric_threshold_met",
    adjusted: float | None = None,
) -> dict:
    metrics = {}
    if neg_mse is not None:
        metrics["neg_mse"] = neg_mse
    be: dict = {
        "success": success,
        "source": "y = x * 2.0;\n",
        "compile_stage_reached": "block",
        "mode": mode,
        "failures": [],
        "warnings": [],
        "metrics": metrics,
        "program_metrics": [],
        "ranking_penalty": 0.0,
    }
    if adjusted is not None:
        be["adjusted_sort_score"] = adjusted
    return {
        "schema_version": 1,
        "kind": "axiom.copilot.search_report",
        "goal": "g",
        "converged": True,
        "convergence_reason": convergence_reason,
        "evaluation_mode": mode,
        "score_sort_key": "neg_mse" if neg_mse is not None else None,
        "artifact_files": {"best_ax": BEST_AX_NAME, "iterations": "iterations.json", "search_report": SEARCH_REPORT_JSON_NAME},
        "best_evaluation": be,
        "final_evaluation": be,
    }


def _minimal_pipeline_summary(*, artifact_dir: str | None = None) -> dict:
    be = _minimal_search_report(neg_mse=0.0)["best_evaluation"]
    return {
        "disclaimer": "x",
        "converged": True,
        "convergence_reason": "metric_threshold_met",
        "best_evaluation": be,
        "final_evaluation": be,
        "iterations": [],
        "final_validation": None,
        "artifact_dir": artifact_dir,
        "restarts": {
            "total": 1,
            "winning_index": 0,
            "per_restart": [
                {
                    "index": 0,
                    "converged": True,
                    "convergence_reason": "compile_success",
                    "iteration_count": 1,
                    "best_evaluation": be,
                    "final_evaluation": be,
                }
            ],
        },
    }


def test_single_search_exact_and_near_hit(tmp_path: Path) -> None:
    d = tmp_path / "run_a"
    d.mkdir()
    (d / BEST_AX_NAME).write_text("y = x * 2.0;\n", encoding="utf-8")
    rep = _minimal_search_report(neg_mse=0.0, convergence_reason="metric_threshold_met")
    (d / SEARCH_REPORT_JSON_NAME).write_text(json.dumps(rep), encoding="utf-8")
    runs, _ = collect_run_snapshots([d], None, DEFAULT_NEAR_NEG_MSE)
    assert len(runs) == 1
    assert runs[0].exact_hit and runs[0].near_hit
    agg = build_aggregate(runs)
    assert agg["exact_hit_count"] == 1
    assert agg["near_hit_count"] == 1
    assert agg["convergence_reason_counts"]["metric_threshold_met"] == 1


def test_single_search_not_exact_but_near(tmp_path: Path) -> None:
    d = tmp_path / "run_b"
    d.mkdir()
    rep = _minimal_search_report(neg_mse=-1e-12)
    (d / SEARCH_REPORT_JSON_NAME).write_text(json.dumps(rep), encoding="utf-8")
    runs, _ = collect_run_snapshots([d], None, DEFAULT_NEAR_NEG_MSE)
    assert runs[0].near_hit and not runs[0].exact_hit


def test_parent_recursive_finds_nested_search_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "sweeps"
    root.mkdir()
    for name, neg in [("t1", 0.0), ("t2", -1.0)]:
        sd = root / name
        sd.mkdir()
        rep = _minimal_search_report(neg_mse=neg, convergence_reason="compile_success")
        (sd / SEARCH_REPORT_JSON_NAME).write_text(json.dumps(rep), encoding="utf-8")
    runs, meta = collect_run_snapshots([], root, DEFAULT_NEAR_NEG_MSE)
    assert len(runs) == 2
    assert sum(r.exact_hit for r in runs) == 1
    assert meta["search_report_paths_total"] >= 2


def test_recursive_finds_top_level_pipeline_json_files(tmp_path: Path) -> None:
    sweep = tmp_path / "pipeline_sweeps"
    sweep.mkdir()
    for name in ("risk_score_run_1.json", "risk_score_run_2.json"):
        (sweep / name).write_text(json.dumps(_minimal_pipeline_summary()), encoding="utf-8")
    runs, meta = collect_run_snapshots([sweep], None, DEFAULT_NEAR_NEG_MSE)
    assert len(runs) == 2
    assert meta["pipeline_json_matched"] == 2
    kinds = {r.kind for r in runs}
    assert kinds == {"pipeline_json"}


def test_dedup_pipeline_json_covers_matching_artifact_dir(tmp_path: Path) -> None:
    art = tmp_path / "bundle"
    art.mkdir()
    (art / SEARCH_REPORT_JSON_NAME).write_text(json.dumps(_minimal_search_report(neg_mse=0.0)), encoding="utf-8")
    summ = tmp_path / "summary.json"
    summ.write_text(json.dumps(_minimal_pipeline_summary(artifact_dir=str(art.resolve()))), encoding="utf-8")
    runs, meta = collect_run_snapshots([tmp_path], None, DEFAULT_NEAR_NEG_MSE)
    assert len(runs) == 1
    assert runs[0].kind == "pipeline_json"
    assert meta["search_dir_skipped_covered"] >= 1


def test_mixed_tree_pipeline_json_and_standalone_search(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    root.mkdir()
    (root / "pipe.json").write_text(json.dumps(_minimal_pipeline_summary()), encoding="utf-8")
    solo = root / "solo"
    solo.mkdir()
    (solo / SEARCH_REPORT_JSON_NAME).write_text(json.dumps(_minimal_search_report(neg_mse=-0.5)), encoding="utf-8")
    runs, meta = collect_run_snapshots([root], None, DEFAULT_NEAR_NEG_MSE)
    assert len(runs) == 2
    assert meta["pipeline_json_matched"] >= 1
    assert meta["search_dir_ingested"] >= 1


def test_pipeline_restart_dir_picks_winning_restart(tmp_path: Path) -> None:
    pipe = tmp_path / "pipe"
    for idx, neg in [(0, -0.5), (1, 0.0)]:
        rd = pipe / f"restart_{idx}"
        rd.mkdir(parents=True)
        rep = _minimal_search_report(neg_mse=neg, convergence_reason="metric_budget_exhausted")
        (rd / SEARCH_REPORT_JSON_NAME).write_text(json.dumps(rep), encoding="utf-8")
        (rd / BEST_AX_NAME).write_text(f"y = x * {idx};\n", encoding="utf-8")
    snap = ingest_path(pipe, DEFAULT_NEAR_NEG_MSE)
    assert snap is not None
    assert snap.kind == "pipeline_dir"
    assert snap.winning_restart_index == 1
    assert snap.primary_metric == 0.0
    assert snap.best_ax_path and "restart_1" in snap.best_ax_path.replace("\\", "/")


def test_pipeline_json_winning_restart_counts(tmp_path: Path) -> None:
    p = tmp_path / "summary.json"
    doc = {
        "disclaimer": "x",
        "converged": True,
        "convergence_reason": "metric_threshold_met",
        "best_evaluation": _minimal_search_report(neg_mse=0.0)["best_evaluation"],
        "final_evaluation": _minimal_search_report(neg_mse=0.0)["best_evaluation"],
        "iterations": [],
        "final_validation": None,
        "artifact_dir": None,
        "restarts": {
            "total": 3,
            "winning_index": 2,
            "per_restart": [
                {"index": 0, "artifact_subdir": str(tmp_path / "r0")},
                {"index": 1, "artifact_subdir": str(tmp_path / "r1")},
                {"index": 2, "artifact_subdir": str(tmp_path / "r2")},
            ],
        },
    }
    (tmp_path / "r2").mkdir()
    (tmp_path / "r2" / BEST_AX_NAME).write_text("ok\n", encoding="utf-8")
    p.write_text(json.dumps(doc), encoding="utf-8")
    snap = ingest_path(p, DEFAULT_NEAR_NEG_MSE)
    assert snap is not None
    assert snap.winning_restart_index == 2
    assert snap.best_ax_path


def test_mixed_runs_best_metric(tmp_path: Path) -> None:
    d1 = tmp_path / "a"
    d1.mkdir()
    (d1 / SEARCH_REPORT_JSON_NAME).write_text(json.dumps(_minimal_search_report(neg_mse=-0.1)), encoding="utf-8")
    d2 = tmp_path / "b"
    d2.mkdir()
    (d2 / SEARCH_REPORT_JSON_NAME).write_text(json.dumps(_minimal_search_report(neg_mse=0.0)), encoding="utf-8")
    runs, agg, text, _ = collect_stability_report([d1, d2])
    assert agg["best"]["primary_metric"] == 0.0
    assert "Best: neg_mse=0" in text


def test_resolve_input_paths_dedupes(tmp_path: Path) -> None:
    a = tmp_path / "x"
    a.mkdir()
    out = resolve_input_paths([a, a], None)
    assert len(out) == 1


def test_stability_report_to_dict_has_summary_text(tmp_path: Path) -> None:
    d = tmp_path / "s"
    d.mkdir()
    (d / SEARCH_REPORT_JSON_NAME).write_text(json.dumps(_minimal_search_report(neg_mse=0.0)), encoding="utf-8")
    runs, agg, _, disc = collect_stability_report([d])
    doc = stability_report_to_dict(runs, agg, discovery=disc)
    assert "runs" in doc and "aggregate" in doc
    assert "discovery" in doc
    assert "summary_text" in doc["aggregate"]
    assert doc["aggregate"]["run_count"] == 1


def test_format_discovery_failure_message_includes_counts() -> None:
    msg = format_discovery_failure_message(
        {
            "scan_roots": ["/tmp/x"],
            "search_report_paths_total": 3,
            "search_report_paths_skipped_under_restart": 2,
            "json_files_probed_pipeline": 5,
            "pipeline_json_matched": 0,
            "patterns": "pat",
        }
    )
    assert "Scan root" in msg and "3" in msg and "5" in msg and "pat" in msg


def test_cli_copilot_stability_report_writes_json(tmp_path: Path, capsys) -> None:
    d = tmp_path / "cli_run"
    d.mkdir()
    (d / SEARCH_REPORT_JSON_NAME).write_text(json.dumps(_minimal_search_report(neg_mse=0.0)), encoding="utf-8")
    outj = tmp_path / "out.json"
    main(["copilot-stability-report", str(d), "--json-out", str(outj)])
    data = json.loads(outj.read_text(encoding="utf-8"))
    assert data["aggregate"]["run_count"] == 1
    assert "discovery" in data
    assert "Copilot stability" in capsys.readouterr().out


def test_cli_no_inputs_exits_1(capsys) -> None:
    with pytest.raises(SystemExit) as ei:
        main(["copilot-stability-report"])
    assert ei.value.code == 1
    err = capsys.readouterr().err
    assert "No copilot runs matched after discovery" in err
    assert "Scan root" in err
