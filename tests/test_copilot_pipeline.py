"""NL→``.ax`` pipeline (Phase 71) + multi-restart best-of-N (Phase 80)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import axiom.cli as cli_mod
from axiom.cli import main
from axiom.compiler.parser import reset_parser
from axiom.copilot.artifacts import BEST_AX_NAME, ITERATIONS_JSON_NAME, SEARCH_REPORT_JSON_NAME
from axiom.copilot.benchmarks import default_neg_mse_score_fn
from axiom.copilot.pipeline import (
    PIPELINE_DISCLAIMER,
    CopilotPipelineConfig,
    copilot_pipeline_summary_dict,
    run_copilot_pipeline,
)
from axiom.copilot.search import CopilotSearchConfig
from axiom.experts.base import ExpertDraftRequest, ExpertDraftResponse, ExpertRepairRequest


@pytest.fixture(autouse=True)
def _fresh_parser() -> None:
    reset_parser()
    yield
    reset_parser()


class _GoodExpert:
    def __init__(self) -> None:
        self._ax = "y = neural([x]);\n"

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        return ExpertDraftResponse(ax_source=self._ax, backend_name="fake", explanation="ok")

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        return ExpertDraftResponse(ax_source=self._ax, backend_name="fake")


class _BrokenAxExpert:
    """Always emits syntactically invalid Axiom source (final compile pass should fail)."""

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        return ExpertDraftResponse(ax_source="not_axiom {{{\n", backend_name="fake")

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        return ExpertDraftResponse(ax_source="still_bad {{{\n", backend_name="fake")


def _search_cfg(expert, *, artifact_dir: Path | None, max_iterations: int = 2, **kwargs) -> CopilotSearchConfig:
    return CopilotSearchConfig(
        expert=expert,
        goal="test goal",
        max_iterations=max_iterations,
        mode="compile_only",
        artifact_dir=artifact_dir,
        **kwargs,
    )


def test_pipeline_writes_artifacts_and_passes_final_validation(tmp_path: Path) -> None:
    art = tmp_path / "run1"
    expert = _GoodExpert()
    cfg = _search_cfg(expert, artifact_dir=art)
    pcfg = CopilotPipelineConfig(search=cfg, final_validate=True)
    result = run_copilot_pipeline(pcfg)
    assert (art / BEST_AX_NAME).is_file()
    assert (art / ITERATIONS_JSON_NAME).is_file()
    assert (art / SEARCH_REPORT_JSON_NAME).is_file()
    assert result.final_validation is not None
    assert result.final_validation.success is True
    assert "neural" in (art / BEST_AX_NAME).read_text(encoding="utf-8")


def test_pipeline_summary_json_shape(tmp_path: Path) -> None:
    expert = _GoodExpert()
    cfg = _search_cfg(expert, artifact_dir=tmp_path / "a")
    result = run_copilot_pipeline(CopilotPipelineConfig(search=cfg))
    doc = copilot_pipeline_summary_dict(result, artifact_dir_resolved=result.artifact_dir)
    assert doc["disclaimer"] == PIPELINE_DISCLAIMER
    assert doc["converged"] is True
    assert doc["convergence_reason"] == "compile_success"
    assert doc["metric_repair"]["enabled"] is False
    assert doc["best_evaluation"]["success"] is True
    assert doc["final_validation"]["success"] is True
    assert doc["restarts"]["total"] == 1
    assert doc["restarts"]["winning_index"] == 0
    assert len(doc["restarts"]["per_restart"]) == 1
    json.dumps(doc)


def test_pipeline_extra_best_ax_path(tmp_path: Path) -> None:
    expert = _GoodExpert()
    cfg = _search_cfg(expert, artifact_dir=tmp_path / "bundle")
    extra = tmp_path / "extra.ax"
    run_copilot_pipeline(CopilotPipelineConfig(search=cfg, best_ax_path=extra))
    assert extra.read_text(encoding="utf-8").strip() == "y = neural([x]);"


def test_pipeline_skips_final_validation_when_disabled(tmp_path: Path) -> None:
    expert = _GoodExpert()
    cfg = _search_cfg(expert, artifact_dir=None)
    result = run_copilot_pipeline(CopilotPipelineConfig(search=cfg, final_validate=False))
    assert result.final_validation is None
    doc = copilot_pipeline_summary_dict(result)
    assert doc["final_validation"] is None


def test_pipeline_final_validation_surfaces_compile_failure(tmp_path: Path, monkeypatch) -> None:
    expert = _GoodExpert()
    cfg = _search_cfg(expert, artifact_dir=None)

    def _fake_validate(candidate, *, max_unroll: int = 8):
        from axiom.copilot.models import ProgramFailure, ProgramValidationReport

        return ProgramValidationReport(
            success=False,
            source=candidate.source,
            compile_stage_reached="parse",
            failures=[
                ProgramFailure(stage="parse", kind="syntax", message="forced failure", detail="Test")
            ],
        )

    monkeypatch.setattr("axiom.copilot.pipeline.validate_program", _fake_validate)
    result = run_copilot_pipeline(CopilotPipelineConfig(search=cfg, final_validate=True))
    assert result.final_validation is not None
    assert result.final_validation.success is False
    doc = copilot_pipeline_summary_dict(result)
    assert doc["final_validation"]["success"] is False
    assert doc["final_validation"]["failures"]


def test_broken_champion_final_validation_fails_without_mock(tmp_path: Path) -> None:
    expert = _BrokenAxExpert()
    cfg = _search_cfg(expert, artifact_dir=None, max_iterations=2)
    result = run_copilot_pipeline(CopilotPipelineConfig(search=cfg, final_validate=True))
    assert result.final_validation is not None
    assert result.final_validation.success is False


class _DraftSeqExpert:
    """Deterministic drafts: first `y = x * 1.0`, second `y = x * 2.0` (for double-x)."""

    def __init__(self, sources: list[str]) -> None:
        self._q = list(sources)

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        ax = self._q.pop(0)
        return ExpertDraftResponse(ax_source=ax, backend_name="seq", metadata={})

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        return ExpertDraftResponse(ax_source="y = 1.0;\n", backend_name="seq")


def test_pipeline_multi_restart_picks_best_symbolic(tmp_path: Path) -> None:
    ex = _DraftSeqExpert(["y = x * 1.0;\n", "y = x * 2.0;\n"])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="double x",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=[{"x": 1.0}],
        expected_rows=[{"y": 2.0}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    result = run_copilot_pipeline(CopilotPipelineConfig(search=cfg, restarts=2, final_validate=False))
    assert result.restarts == 2
    assert result.winning_restart_index == 1
    assert "x * 2.0" in result.search_result.best_source
    assert "x * 2.0" in result.per_restart[1]["best_source"]


def test_pipeline_multi_restart_artifact_subdirs(tmp_path: Path) -> None:
    ex = _DraftSeqExpert(["y = x * 1.0;\n", "y = x * 2.0;\n"])
    root = tmp_path / "multi"
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="double x",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=[{"x": 1.0}],
        expected_rows=[{"y": 2.0}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
        artifact_dir=root,
    )
    run_copilot_pipeline(CopilotPipelineConfig(search=cfg, restarts=2, final_validate=False))
    assert (root / "restart_0" / BEST_AX_NAME).is_file()
    assert (root / "restart_1" / BEST_AX_NAME).is_file()


def test_pipeline_multi_restart_summary_json_per_restart_has_artifact_subdir(tmp_path: Path) -> None:
    ex = _DraftSeqExpert(["y = x * 1.0;\n", "y = x * 2.0;\n"])
    root = tmp_path / "sum_art"
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="double x",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=[{"x": 1.0}],
        expected_rows=[{"y": 2.0}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
        artifact_dir=root,
    )
    result = run_copilot_pipeline(CopilotPipelineConfig(search=cfg, restarts=2, final_validate=False))
    doc = copilot_pipeline_summary_dict(result, artifact_dir_resolved=result.artifact_dir)
    pr = doc["restarts"]["per_restart"]
    assert len(pr) == 2
    assert Path(pr[0]["artifact_subdir"]).name == "restart_0"
    assert Path(pr[1]["artifact_subdir"]).name == "restart_1"
    assert doc["restarts"]["winning_index"] == 1


def test_pipeline_multi_restart_risk_score_exact_blend_beats_linear(tmp_path: Path) -> None:
    """Restart 0: valid symbolic but wrong coefficients; restart 1: exact clamp+blend (risk_score-style)."""
    mediocre = "risk_score = 0.5 * risk_a + 0.5 * risk_b;\n"
    exact = "risk_score = max(0.0, min(1.0, 0.7 * risk_a + 0.3 * risk_b));\n"
    ex = _DraftSeqExpert([mediocre, exact])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="risk_score from risk_a and risk_b",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=[{"risk_a": 1.0, "risk_b": 0.0}],
        expected_rows=[{"risk_score": 0.7}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    result = run_copilot_pipeline(CopilotPipelineConfig(search=cfg, restarts=2, final_validate=False))
    assert result.winning_restart_index == 1
    assert "max(0.0" in result.search_result.best_source
    assert "0.7 * risk_a" in result.search_result.best_source


def test_cli_copilot_run_help_exits_ok() -> None:
    with pytest.raises(SystemExit) as e:
        main(["copilot-run", "--help"])
    assert e.value.code == 0


def test_cli_copilot_run_smoke(tmp_path: Path, monkeypatch) -> None:
    expert = _GoodExpert()
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: expert)
    art = tmp_path / "cli_run"
    summ = tmp_path / "summary.json"
    best = tmp_path / "out.ax"
    main(
        [
            "copilot-run",
            "--backend",
            "onyx-qwen",
            "--goal",
            "smoke",
            "--expert-url",
            "http://127.0.0.1:9/v1/",
            "--expert-model",
            "fake",
            "--iterations",
            "2",
            "--compile-only",
            "--artifact-dir",
            str(art),
            "--summary-out",
            str(summ),
            "--out",
            str(best),
        ]
    )
    assert (art / BEST_AX_NAME).is_file()
    assert summ.is_file()
    doc = json.loads(summ.read_text(encoding="utf-8"))
    assert "disclaimer" in doc
    assert doc["restarts"]["total"] == 1
    assert doc["final_validation"]["success"] is True
    assert best.read_text(encoding="utf-8").strip() == "y = neural([x]);"


def test_cli_copilot_run_restarts_summary_json(tmp_path: Path, monkeypatch) -> None:
    ex_path = tmp_path / "ex.json"
    ex_path.write_text(
        json.dumps([{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_mod,
        "_make_copilot_expert",
        lambda _a: _DraftSeqExpert(["y = x * 1.0;\n", "y = x * 2.0;\n"]),
    )
    summ = tmp_path / "summary_r.json"
    main(
        [
            "copilot-run",
            "--backend",
            "onyx-qwen",
            "--goal",
            "double x",
            "--expert-url",
            "http://127.0.0.1:9/v1/",
            "--expert-model",
            "fake",
            "--iterations",
            "1",
            "--examples-json",
            str(ex_path),
            "--restarts",
            "2",
            "--summary-out",
            str(summ),
            "--no-final-validate",
        ]
    )
    doc = json.loads(summ.read_text(encoding="utf-8"))
    assert doc["restarts"]["total"] == 2
    assert doc["restarts"]["winning_index"] == 1
    assert len(doc["restarts"]["per_restart"]) == 2
    assert "x * 2.0" in doc["best_source"]


def test_cli_copilot_run_restarts_writes_restart_subdirs(tmp_path: Path, monkeypatch) -> None:
    ex_path = tmp_path / "ex_sub.json"
    ex_path.write_text(
        json.dumps([{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli_mod,
        "_make_copilot_expert",
        lambda _a: _DraftSeqExpert(["y = x * 1.0;\n", "y = x * 2.0;\n"]),
    )
    art = tmp_path / "cli_restart_dirs"
    summ = tmp_path / "sum_sub.json"
    main(
        [
            "copilot-run",
            "--backend",
            "onyx-qwen",
            "--goal",
            "double x",
            "--expert-url",
            "http://127.0.0.1:9/v1/",
            "--expert-model",
            "fake",
            "--iterations",
            "1",
            "--examples-json",
            str(ex_path),
            "--restarts",
            "2",
            "--artifact-dir",
            str(art),
            "--summary-out",
            str(summ),
            "--no-final-validate",
        ]
    )
    assert (art / "restart_0" / BEST_AX_NAME).is_file()
    assert (art / "restart_1" / BEST_AX_NAME).is_file()
    doc = json.loads(summ.read_text(encoding="utf-8"))
    assert doc["restarts"]["winning_index"] == 1
    assert Path(doc["restarts"]["per_restart"][0]["artifact_subdir"]).name == "restart_0"
