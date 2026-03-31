"""Draft → evaluate → repair loop over ``.ax`` programs (expert backend is injectable; no network here)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from axiom.copilot.evaluator import evaluate_program
from axiom.copilot.models import (
    EvaluationMode,
    ProgramCandidate,
    ProgramEvaluationReport,
    ProgramFailure,
    ProgramMetric,
    TrainTabularParams,
)
from axiom.copilot.summarize import safe_summarize_evaluation
from axiom.experts.base import ExpertDraftRequest, ExpertRepairRequest, SemanticExpert

ExpertRequestPayload = Dict[str, Any]


def build_draft_context(
    *,
    domain_context: Optional[str],
    example_input_rows: Optional[Sequence[Mapping[str, Any]]],
    expected_rows: Optional[Sequence[Mapping[str, Any]]],
    train_tabular_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Structured, JSON-serializable context for :class:`ExpertDraftRequest` (inspectable, deterministic)."""
    ctx: Dict[str, Any] = {
        "domain_context": domain_context or "",
        "example_input_rows": [dict(r) for r in example_input_rows] if example_input_rows else [],
        "expected_outputs": [dict(r) for r in expected_rows] if expected_rows else [],
    }
    if train_tabular_meta:
        ctx["train_tabular"] = dict(train_tabular_meta)
    return ctx


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
    train_tabular_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "example_input_rows": [dict(r) for r in example_input_rows] if example_input_rows else [],
        "expected_outputs": [dict(r) for r in expected_rows] if expected_rows else [],
        "evaluation_mode": evaluation_mode,
    }
    if train_tabular_meta:
        out["train_tabular"] = dict(train_tabular_meta)
    return out


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
    # If True, after each evaluation call expert.summarize_trace (extra latency; failures are ignored).
    summarize_traces: bool = False
    # If set, run_copilot_search writes best.ax, iterations.json, search_report.json under this path.
    artifact_dir: Optional[Path] = None
    # Merged into expert draft/repair JSON context (e.g. benchmark task ids); no effect on evaluation harness.
    draft_context_extras: Dict[str, Any] = field(default_factory=dict)
    repair_context_extras: Dict[str, Any] = field(default_factory=dict)
    # When mode == "train_tabular": merged row dicts (inputs ∪ expected) + target + params + expected for scoring.
    tabular_train_rows: Optional[Sequence[Mapping[str, Any]]] = None
    tabular_eval_rows: Optional[Sequence[Mapping[str, Any]]] = None
    tabular_target_var: Optional[str] = None
    tabular_train_params: Optional[TrainTabularParams] = None
    tabular_eval_expected_rows: Optional[Sequence[Mapping[str, Any]]] = None


@dataclass
class CopilotIterationRecord:
    index: int
    source: str
    evaluation: ProgramEvaluationReport
    producing_payload: ExpertRequestPayload
    outgoing_repair_error_report: Optional[str] = None
    producing_expert: Dict[str, Any] = field(default_factory=dict)
    """Expert response metadata for the call that produced ``source`` (draft or repair)."""
    semantic_trace_summary: Optional[str] = None
    """Natural-language trace/metrics narrative when :attr:`CopilotSearchConfig.summarize_traces` is on."""


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


def _train_tabular_meta(config: CopilotSearchConfig) -> Optional[Dict[str, Any]]:
    if config.mode != "train_tabular":
        return None
    ttp = config.tabular_train_params or TrainTabularParams()
    return {
        "target_var": config.tabular_target_var or "",
        "train_row_count": len(config.tabular_train_rows or ()),
        "eval_row_count": len(config.tabular_eval_rows or ()),
        "epochs": ttp.epochs,
        "learning_rate": ttp.learning_rate,
        "weight_decay": ttp.weight_decay,
        "batch_size": ttp.batch_size,
    }


def run_copilot_search(config: CopilotSearchConfig) -> CopilotSearchResult:
    from axiom.copilot.artifacts import expert_response_to_dict, persist_copilot_artifacts

    tt_meta = _train_tabular_meta(config)
    ctx: Dict[str, Any] = build_draft_context(
        domain_context=config.domain_context,
        example_input_rows=config.example_input_rows,
        expected_rows=config.expected_rows,
        train_tabular_meta=tt_meta,
    )
    if config.draft_context_extras:
        ctx = {**ctx, **dict(config.draft_context_extras)}
    draft_req = ExpertDraftRequest(goal=config.goal, context=ctx)
    draft_resp = config.expert.draft_program(draft_req)
    current = draft_resp.ax_source
    provenance_meta = expert_response_to_dict(draft_resp, "draft")
    sort_key = config.score_sort_key
    max_it = max(1, int(config.max_iterations))

    iterations: List[CopilotIterationRecord] = []
    best_eval: Optional[ProgramEvaluationReport] = None
    best_source = current
    ingress_payload: ExpertRequestPayload = _draft_payload_dict(draft_req)

    final_report: Optional[ProgramEvaluationReport] = None
    converged = False

    need_trace = config.include_trace_snippet or config.summarize_traces

    for i in range(max_it):
        source_evaluated = current
        producing = ingress_payload
        iter_expert_meta = provenance_meta

        if config.mode == "train_tabular":
            report = evaluate_program(
                ProgramCandidate(source_evaluated),
                mode="train_tabular",
                max_unroll=config.max_unroll,
                train_rows=config.tabular_train_rows,
                eval_rows=config.tabular_eval_rows,
                target_var=config.tabular_target_var,
                train_tabular_params=config.tabular_train_params,
                expected_rows=config.tabular_eval_expected_rows,
                score_fn=config.score_fn,
                predictions_sample_limit=config.predictions_sample_limit,
                include_trace_snippet=need_trace,
            )
        else:
            report = evaluate_program(
                ProgramCandidate(source_evaluated),
                mode=config.mode,
                max_unroll=config.max_unroll,
                input_rows=config.example_input_rows,
                expected_rows=config.expected_rows,
                score_fn=config.score_fn,
                predictions_sample_limit=config.predictions_sample_limit,
                include_trace_snippet=need_trace,
            )
        final_report = report

        sem_summary: Optional[str] = None
        if config.summarize_traces:
            sem_summary = safe_summarize_evaluation(
                config.expert,
                goal=config.goal,
                program=source_evaluated,
                report=report,
            )

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
            repair_ctx: Dict[str, Any] = build_repair_context(
                example_input_rows=config.example_input_rows,
                expected_rows=config.expected_rows,
                evaluation_mode=config.mode,
                train_tabular_meta=tt_meta,
            )
            if config.repair_context_extras:
                repair_ctx = {**repair_ctx, **dict(config.repair_context_extras)}
            repair_req = ExpertRepairRequest(
                goal=config.goal,
                current_program=source_evaluated,
                error_report=err_full,
                context=repair_ctx,
            )
            ingress_payload = _repair_payload_dict(repair_req)
            repair_resp = config.expert.repair_program(repair_req)
            current = repair_resp.ax_source
            provenance_meta = expert_response_to_dict(repair_resp, "repair")
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
                producing_expert=iter_expert_meta,
                semantic_trace_summary=sem_summary,
            )
        )

        if not will_repair:
            break

    assert final_report is not None and best_eval is not None

    result = CopilotSearchResult(
        best_source=best_source,
        best_evaluation=best_eval,
        final_report=final_report,
        converged=converged,
        iterations=iterations,
    )
    if config.artifact_dir is not None:
        persist_copilot_artifacts(config, result, config.artifact_dir)
    return result
