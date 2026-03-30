"""Copilot artifact persistence (Phase 62)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom.compiler.parser import reset_parser
from axiom.copilot.artifacts import (
    BEST_AX_NAME,
    COPILOT_ARTIFACT_SCHEMA_VERSION,
    ITERATIONS_JSON_NAME,
    SEARCH_REPORT_JSON_NAME,
    build_iterations_document,
    build_search_report_document,
    evaluation_report_to_dict,
    expert_response_to_dict,
    json_safe,
    persist_copilot_artifacts,
)
from axiom.copilot.search import CopilotSearchConfig, run_copilot_search
from axiom.experts import ExpertDraftRequest, ExpertDraftResponse, ExpertRepairRequest, ExpertTraceSummaryRequest


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


class _StubExpert:
    def __init__(self) -> None:
        self.n = 0

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        self.n += 1
        return ExpertDraftResponse(
            ax_source="y = 1.0;\n",
            backend_name="stub",
            metadata={"n": self.n, "tag": ("a", "b")},
        )

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        self.n += 1
        return ExpertDraftResponse(ax_source="y = 2.0;\n", backend_name="stub", metadata={"n": self.n})

    def summarize_trace(self, *args, **kwargs) -> str:
        return ""


def test_json_safe_tuple_becomes_list():
    assert json_safe({"k": (1, 2)}) == {"k": [1, 2]}


def test_expert_response_to_dict_roundtrip_json():
    r = ExpertDraftResponse(ax_source="x", backend_name="b", explanation="e", metadata={"z": 1})
    d = expert_response_to_dict(r, "draft")
    json.dumps(d)
    assert d["expert_call"] == "draft" and d["backend_name"] == "b" and d["metadata"]["z"] == 1


def test_persist_copilot_artifacts_writes_three_files(tmp_path: Path):
    cfg = CopilotSearchConfig(expert=_StubExpert(), goal="g1", domain_context="dc", max_iterations=1, mode="compile_only")
    res = run_copilot_search(cfg)
    ad = tmp_path / "run1"
    persist_copilot_artifacts(cfg, res, ad)
    assert (ad / BEST_AX_NAME).is_file()
    assert (ad / ITERATIONS_JSON_NAME).is_file()
    assert (ad / SEARCH_REPORT_JSON_NAME).is_file()
    assert "y = 1.0" in (ad / BEST_AX_NAME).read_text(encoding="utf-8")
    it = json.loads((ad / ITERATIONS_JSON_NAME).read_text(encoding="utf-8"))
    assert it["schema_version"] == COPILOT_ARTIFACT_SCHEMA_VERSION
    assert it["kind"] == "axiom.copilot.iterations"
    assert it["goal"] == "g1"
    assert it["domain_context"] == "dc"
    assert it["backend_name"] == "stub"
    assert it["iteration_count"] == 1
    assert len(it["iterations"]) == 1
    row0 = it["iterations"][0]
    assert row0["success"] is True
    assert row0["candidate_source"] == "y = 1.0;\n"
    assert row0["producing_expert"]["expert_call"] == "draft"
    assert row0["failure_summaries"] == []
    sr = json.loads((ad / SEARCH_REPORT_JSON_NAME).read_text(encoding="utf-8"))
    assert sr["schema_version"] == COPILOT_ARTIFACT_SCHEMA_VERSION
    assert sr["kind"] == "axiom.copilot.search_report"
    assert sr["converged"] is True
    assert sr["failures_metrics_summary"]["best"]["success"] is True
    assert sr["artifact_files"]["best_ax"] == BEST_AX_NAME


def test_run_copilot_search_artifact_dir_same_as_persist(tmp_path: Path):
    ad = tmp_path / "auto"
    cfg = CopilotSearchConfig(
        expert=_StubExpert(),
        goal="g2",
        max_iterations=1,
        mode="compile_only",
        artifact_dir=ad,
    )
    run_copilot_search(cfg)
    assert (ad / BEST_AX_NAME).is_file() and (ad / ITERATIONS_JSON_NAME).is_file()


def test_build_documents_include_failure_summaries():
    ex = _StubExpert()
    ex.draft_program = lambda req: ExpertDraftResponse(ax_source="y = ++++ ;\n", backend_name="stub")  # type: ignore[method-assign]
    cfg = CopilotSearchConfig(expert=ex, goal="gf", max_iterations=1, mode="compile_only")
    res = run_copilot_search(cfg)
    doc = build_iterations_document(cfg, res)
    assert doc["iterations"][0]["success"] is False
    assert doc["iterations"][0]["failure_summaries"]
    rep = build_search_report_document(cfg, res)
    assert rep["failures_metrics_summary"]["per_iteration"][0]["failure_count"] >= 1


def test_evaluation_report_to_dict_matches_program_failure():
    from axiom.copilot.models import ProgramEvaluationReport, ProgramFailure

    rep = ProgramEvaluationReport(
        success=False,
        source="s",
        compile_stage_reached="parse",
        mode="compile_only",
        failures=[ProgramFailure("parse", "syntax", "msg", "E")],
    )
    d = evaluation_report_to_dict(rep)
    assert d["failures"][0]["kind"] == "syntax"
