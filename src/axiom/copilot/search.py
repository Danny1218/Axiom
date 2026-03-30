"""Draft → evaluate → repair loop over ``.ax`` programs (expert backend is injectable; no network here)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from axiom.copilot.evaluator import evaluate_program
from axiom.copilot.models import (
    EvaluationMode,
    ProgramCandidate,
    ProgramEvaluationReport,
    ProgramFailure,
    ProgramMetric,
)
from axiom.experts.base import ExpertDraftRequest, ExpertRepairRequest, SemanticExpert

ExpertRequestPayload = Dict[str, Any]


def build_draft_context(
    *,
    domain_context: Optional[str],
    example_input_rows: Optional[Sequence[Mapping[str, Any]]],
    expected_rows: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Structured, JSON-serializable context for :class:`ExpertDraftRequest` (inspectable, deterministic)."""
    return {
        "domain_context": domain_context or "",
        "example_input_rows": [dict(r) for r in example_input_rows] if example_input_rows else [],
        "expected_outputs": [dict(r) for r in expected_rows] if expected_rows else [],
    }


def format_failures_for_repair(failures: Sequence[ProgramFailure]) -> str:
    lines = ["## Structured compile / evaluation failures", ""]
    for i, f in enumerate(failures):
        lines.append(f"{i + 1}. stage={f.stage!r} kind={f.kind!r} detail={f.detail!r}")
        lines.append(f"   message: {f.message}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_metrics_for_repair(metrics: Mapping[str, float], program_metrics: Sequence[ProgramMetric]) -> str:
    body = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "program_metrics": [{"name": m.name, "value": m.value} for m in program_metrics],
    }
    return "## Metric report (program runs but may be suboptimal)\n\n```json\n" + json.dumps(
        body, indent=2, sort_keys=True
    ) + "\n```\n"


def build_repair_error_report(
    *,
    goal: str,
    domain_context: Optional[str],
    current_ax: str,
    evaluation: ProgramEvaluationReport,
) -> str:
    """Repair prompt: goal, context, current source, failures and/or metrics, fix instructions."""
    parts: List[str] = [
        "## Goal",
        goal.strip(),
        "",
        "## Domain context",
        (domain_context or "").strip() or "(none)",
        "",
        "## Current .ax program",
        "```ax",
        current_ax.rstrip(),
        "```",
        "",
    ]
    if evaluation.failures:
        parts.append(format_failures_for_repair(evaluation.failures))
        parts.append("")
    if evaluation.metrics or evaluation.program_metrics:
        parts.append(format_metrics_for_repair(evaluation.metrics, evaluation.program_metrics))
        parts.append("")
    parts.append(
        "## Instructions\n"
        "Return a **corrected full** Axiom (.ax) program as plain source only "
        "(no markdown fences unless the program itself needs them). "
        "Preserve the user goal and I/O intent."
    )
    return "\n".join(parts)


def build_repair_context(
    *,
    example_input_rows: Optional[Sequence[Mapping[str, Any]]],
    expected_rows: Optional[Sequence[Mapping[str, Any]]],
    evaluation_mode: EvaluationMode,
) -> Dict[str, Any]:
    return {
        "example_input_rows": [dict(r) for r in example_input_rows] if example_input_rows else [],
        "expected_outputs": [dict(r) for r in expected_rows] if expected_rows else [],
        "evaluation_mode": evaluation_mode,
    }


@dataclass
class CopilotSearchConfig:
    """Inputs for :func:`run_copilot_search`."""

    expert: SemanticExpert
    goal: str
    domain_context: Optional[str] = None
    example_input_rows: Optional[Sequence[Mapping[str, Any]]] = None
    expected_rows: Optional[Sequence[Mapping[str, Any]]] = None
    max_iterations: int = 8
    mode: EvaluationMode = "compile_only"
    max_unroll: int = 8
    score_fn: Optional[
        Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, float]]
    ] = None
    score_sort_key: Optional[str] = None
    repair_valid_with_metrics: bool = False
    metric_repair_if_below: Optional[float] = None
    predictions_sample_limit: int = 3
    include_trace_snippet: bool = True


@dataclass
class CopilotIterationRecord:
    index: int
    source: str
    evaluation: ProgramEvaluationReport
    producing_payload: ExpertRequestPayload
    outgoing_repair_error_report: Optional[str] = None


@dataclass
class CopilotSearchResult:
    best_source: str
    best_evaluation: ProgramEvaluationReport
    final_report: ProgramEvaluationReport
    converged: bool
    iterations: List[CopilotIterationRecord] = field(default_factory=list)


def _repair_payload_dict(req: ExpertRepairRequest) -> ExpertRequestPayload:
    return {
        "type": "repair",
        "goal": req.goal,
        "current_program": req.current_program,
        "error_report": req.error_report,
        "context": dict(req.context),
    }


def _draft_payload_dict(req: ExpertDraftRequest) -> ExpertRequestPayload:
    return {"type": "draft", "goal": req.goal, "context": dict(req.context)}


def _metric_value(report: ProgramEvaluationReport, sort_key: Optional[str]) -> Optional[float]:
    if not report.metrics:
        return None
    keys = list(report.metrics.keys())
    key = sort_key
    if key is None:
        if len(keys) == 1:
            key = keys[0]
        else:
            return None
    if key not in report.metrics:
        return None
    return float(report.metrics[key])


def _score_for_sort(
    report: ProgramEvaluationReport,
    sort_key: Optional[str],
) -> Optional[float]:
    if not report.success:
        return None
    return _metric_value(report, sort_key)


def _is_better(
    cand: ProgramEvaluationReport,
    best: Optional[ProgramEvaluationReport],
    sort_key: Optional[str],
) -> bool:
    if best is None:
        return True
    c_ok, b_ok = cand.success, best.success
    if c_ok and not b_ok:
        return True
    if not c_ok and b_ok:
        return False
    if not c_ok and not b_ok:
        return False
    cs = _score_for_sort(cand, sort_key)
    bs = _score_for_sort(best, sort_key)
    if cs is not None and bs is not None:
        return cs > bs
    if cs is not None and bs is None:
        return True
    if cs is None and bs is not None:
        return False
    return False


def _needs_metric_repair(config: CopilotSearchConfig, report: ProgramEvaluationReport) -> bool:
    if not report.success or not config.repair_valid_with_metrics:
        return False
    if not report.metrics and not report.program_metrics:
        return False
    thr = config.metric_repair_if_below
    if thr is None:
        return True
    v = _metric_value(report, config.score_sort_key)
    return v is not None and v < thr


def run_copilot_search(config: CopilotSearchConfig) -> CopilotSearchResult:
    ctx = build_draft_context(
        domain_context=config.domain_context,
        example_input_rows=config.example_input_rows,
        expected_rows=config.expected_rows,
    )
    draft_req = ExpertDraftRequest(goal=config.goal, context=ctx)
    draft_resp = config.expert.draft_program(draft_req)
    current = draft_resp.ax_source
    sort_key = config.score_sort_key
    max_it = max(1, int(config.max_iterations))

    iterations: List[CopilotIterationRecord] = []
    best_eval: Optional[ProgramEvaluationReport] = None
    best_source = current
    ingress_payload: ExpertRequestPayload = _draft_payload_dict(draft_req)

    final_report: Optional[ProgramEvaluationReport] = None
    converged = False

    for i in range(max_it):
        source_evaluated = current
        producing = ingress_payload

        report = evaluate_program(
            ProgramCandidate(source_evaluated),
            mode=config.mode,
            max_unroll=config.max_unroll,
            input_rows=config.example_input_rows,
            expected_rows=config.expected_rows,
            score_fn=config.score_fn,
            predictions_sample_limit=config.predictions_sample_limit,
            include_trace_snippet=config.include_trace_snippet,
        )
        final_report = report

        if _is_better(report, best_eval, sort_key):
            best_eval = report
            best_source = source_evaluated

        need_failure_repair = not report.success
        need_metric_repair = _needs_metric_repair(config, report)
        can_repair = i < max_it - 1
        will_repair = (need_failure_repair or need_metric_repair) and can_repair

        err_full: Optional[str] = None
        if will_repair:
            err_full = build_repair_error_report(
                goal=config.goal,
                domain_context=config.domain_context,
                current_ax=source_evaluated,
                evaluation=report,
            )
            repair_ctx = build_repair_context(
                example_input_rows=config.example_input_rows,
                expected_rows=config.expected_rows,
                evaluation_mode=config.mode,
            )
            repair_req = ExpertRepairRequest(
                goal=config.goal,
                current_program=source_evaluated,
                error_report=err_full,
                context=repair_ctx,
            )
            ingress_payload = _repair_payload_dict(repair_req)
            current = config.expert.repair_program(repair_req).ax_source
        else:
            if report.success:
                converged = True

        iterations.append(
            CopilotIterationRecord(
                index=i,
                source=source_evaluated,
                evaluation=report,
                producing_payload=producing,
                outgoing_repair_error_report=err_full,
            )
        )

        if not will_repair:
            break

    assert final_report is not None and best_eval is not None

    return CopilotSearchResult(
        best_source=best_source,
        best_evaluation=best_eval,
        final_report=final_report,
        converged=converged,
        iterations=iterations,
    )
