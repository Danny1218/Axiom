"""``axiom.copilot.summarize`` — expert trace narration helpers (Phase 64)."""

from __future__ import annotations

import pytest

from axiom.compiler.parser import reset_parser
from axiom.copilot.models import ProgramEvaluationReport, ProgramFailure, ProgramMetric
from axiom.copilot.summarize import (
    safe_summarize_evaluation,
    summary_context_from_report,
    trace_and_metrics_for_summary,
)
from axiom.experts import ExpertTraceSummaryRequest, SemanticExpert


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


def _report(
    *,
    success: bool = True,
    trace: dict | None = None,
    metrics: dict | None = None,
    failures: list | None = None,
    program_metrics: list | None = None,
) -> ProgramEvaluationReport:
    return ProgramEvaluationReport(
        success=success,
        source="y = 1.0;",
        compile_stage_reached="block",
        mode="compile_only",
        failures=failures or [],
        warnings=["w1"],
        metrics=metrics or {},
        program_metrics=program_metrics or [],
        trace_snippet=trace,
    )


def test_trace_and_metrics_for_summary_empty_trace():
    r = _report(trace=None, metrics={"q": 1.5})
    tr, m = trace_and_metrics_for_summary(r)
    assert tr == {} and m == {"q": 1.5}


def test_trace_and_metrics_for_summary_json_safe_nested():
    r = _report(trace={"a": (1, 2)}, metrics={})
    tr, _ = trace_and_metrics_for_summary(r)
    assert tr == {"a": [1, 2]}


def test_summary_context_from_report_failures_and_program_metrics():
    r = _report(
        success=False,
        failures=[ProgramFailure("parse", "syntax", "bad", "E")],
        program_metrics=[ProgramMetric("neg_mse", -0.5)],
    )
    ctx = summary_context_from_report(r)
    assert ctx["evaluation_success"] is False
    assert ctx["failure_summaries"][0]["kind"] == "syntax"
    assert ctx["program_metrics"] == [{"name": "neg_mse", "value": -0.5}]
    assert ctx["warnings"] == ["w1"]


class _OkExpert:
    def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
        assert request.goal == "g"
        assert "y = 1.0" in request.program
        assert isinstance(request.trace, dict)
        assert isinstance(request.metrics, dict)
        assert "failure_summaries" in request.context
        return "  summary line  "


def test_safe_summarize_evaluation_success():
    ex: SemanticExpert = _OkExpert()
    r = _report(trace={"x": 1.0}, metrics={"m": 0.0})
    assert safe_summarize_evaluation(ex, goal="g", program=r.source, report=r) == "summary line"


def test_safe_summarize_evaluation_returns_none_on_expert_error():
    class _Bad:
        def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
            raise RuntimeError("network down")

    r = _report()
    assert safe_summarize_evaluation(_Bad(), goal="g", program=r.source, report=r) is None


def test_safe_summarize_evaluation_returns_none_on_empty_response():
    class _Empty:
        def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
            return "   \n"

    r = _report()
    assert safe_summarize_evaluation(_Empty(), goal="g", program=r.source, report=r) is None
