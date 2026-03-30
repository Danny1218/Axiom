"""Optional expert narration of copilot traces (explain output + metrics + failures).

Uses :meth:`~axiom.experts.base.SemanticExpert.summarize_trace`. Failures here never affect
evaluation outcomes — callers should treat a ``None`` return as “summary skipped”.
"""

from __future__ import annotations

from typing import Any, Optional

from axiom.copilot.models import ProgramEvaluationReport
from axiom.experts.base import ExpertTraceSummaryRequest, SemanticExpert


def trace_and_metrics_for_summary(report: ProgramEvaluationReport) -> tuple[dict[str, Any], dict[str, float]]:
    from axiom.copilot.artifacts import json_safe

    trace: dict[str, Any] = json_safe(dict(report.trace_snippet)) if report.trace_snippet else {}
    metrics = {str(k): float(v) for k, v in report.metrics.items()}
    return trace, metrics


def summary_context_from_report(report: ProgramEvaluationReport) -> dict[str, Any]:
    from axiom.copilot.artifacts import json_safe

    return {
        "failure_summaries": [
            {"stage": f.stage, "kind": f.kind, "message": f.message, "detail": f.detail}
            for f in report.failures
        ],
        "program_metrics": [{"name": m.name, "value": float(m.value)} for m in report.program_metrics],
        "warnings": list(report.warnings),
        "compile_stage_reached": report.compile_stage_reached,
        "evaluation_success": report.success,
        "predictions_sample": json_safe(report.predictions_sample) if report.predictions_sample else [],
    }


def safe_summarize_evaluation(
    expert: SemanticExpert,
    *,
    goal: str,
    program: str,
    report: ProgramEvaluationReport,
) -> Optional[str]:
    """Call ``expert.summarize_trace``; return ``None`` on any error or empty string."""
    try:
        trace, metrics = trace_and_metrics_for_summary(report)
        req = ExpertTraceSummaryRequest(
            goal=goal.strip(),
            program=program,
            trace=trace,
            metrics=metrics,
            context=summary_context_from_report(report),
        )
        out = expert.summarize_trace(req)
        s = out.strip() if isinstance(out, str) else str(out).strip()
        return s or None
    except Exception:
        return None


__all__ = [
    "safe_summarize_evaluation",
    "summary_context_from_report",
    "trace_and_metrics_for_summary",
]
