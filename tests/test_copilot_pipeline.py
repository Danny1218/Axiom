"""Phase 71: NL→``.ax`` pipeline (search + artifacts + final compile check)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import axiom.cli as cli_mod
from axiom.cli import main
from axiom.compiler.parser import reset_parser
from axiom.copilot.artifacts import BEST_AX_NAME, ITERATIONS_JSON_NAME, SEARCH_REPORT_JSON_NAME
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
    assert doc["best_evaluation"]["success"] is True
    assert doc["final_validation"]["success"] is True
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
    assert doc["final_validation"]["success"] is True
    assert best.read_text(encoding="utf-8").strip() == "y = neural([x]);"
