"""Draft → evaluate → repair loop over ``.ax`` programs (expert backend is injectable; no network here)."""

from __future__ import annotations

import json
import math
import re
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
from axiom.experts.base import ExpertDraftRequest, ExpertDraftResponse, ExpertRepairRequest, SemanticExpert
from axiom.experts.onyx_qwen import COMPLETION_OVERRIDES_CONTEXT_KEY, OnyxQwenHTTPError, ax_source_metadata_flags

ExpertRequestPayload = Dict[str, Any]

# Default stop threshold for built-in ``neg_mse`` (higher is better; 0 ≈ perfect). Repair while score < this.
DEFAULT_METRIC_REPAIR_THRESHOLD = -1e-9

_GOAL_SYMBOLIC_MATH_HINT = re.compile(
    r"(compute|formula|symbolic|arithmetic|algebra|multiply|coefficient|exact|"
    r"risk_score|linear|weighted|blend|double\b|mapping|polynomial)",
    re.I,
)
_GOAL_EXACT_SYMBOLIC_EXTRA = re.compile(
    r"(max\s*\(|min\s*\(|clamp|affine|weighted\s+(sum|blend)|risk_score)",
    re.I,
)

# Penalties subtracted from raw sort metric (higher-is-better, e.g. ``neg_mse``).
_PENALTY_NEURAL_EXACT = 2.0
_PENALTY_INDEXED = 0.25
_PENALTY_OUTPUT = 0.25
_PENALTY_SUSPICIOUS_NUM = 0.25


def _goal_suggests_symbolic_math(goal: str) -> bool:
    """Heuristic: user goal looks like an exact symbolic / numeric mapping (not a policy)."""
    g = (goal or "").strip()
    if not g:
        return False
    if _GOAL_SYMBOLIC_MATH_HINT.search(g):
        return True
    if len(g) <= 220 and re.search(r"[0-9]\s*[\*\+\-]\s*[0-9]|=\s*max|=\s*min|\*\s*x\b", g):
        return True
    return False


def is_exact_symbolic_examples_task(config: CopilotSearchConfig) -> bool:
    """predict_rows + expected rows + goal looks like affine/clamp/small math (not a policy)."""
    if config.mode != "predict_rows" or not config.expected_rows:
        return False
    g = config.goal or ""
    if _goal_suggests_symbolic_math(g):
        return True
    if len(g) <= 500 and _GOAL_EXACT_SYMBOLIC_EXTRA.search(g):
        return True
    return False


def _linear_xy_coeff_str(v: float) -> str:
    """Deterministic float formatting for emitted ``.ax`` literals."""
    if not math.isfinite(v):
        return repr(v)
    r = round(v)
    if abs(v - r) < 1e-9:
        return f"{float(r):.1f}"
    s = format(v, ".12g")
    s = s.rstrip("0").rstrip(".") if "." in s else s
    return s if s else "0.0"


def _linear_xy_canonical_source(a: float, b: float) -> str:
    ca, cb = _linear_xy_coeff_str(a), _linear_xy_coeff_str(b)
    if math.isclose(b, 0.0, abs_tol=1e-12, rel_tol=1e-12):
        return f"y = x * {ca};\n"
    return f"y = x * {ca} + {cb};\n"


def _try_linear_xy_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    """If ``exact_symbolic_examples_task`` and examples are exact ``y = a*x+b`` over ``x``/``y``, return draft; else None."""
    if not is_exact_symbolic_examples_task(config):
        return None
    if config.mode != "predict_rows":
        return None
    inp = config.example_input_rows
    exp = config.expected_rows
    if not inp or not exp or len(inp) != len(exp):
        return None
    n = len(inp)
    pts: List[tuple[float, float]] = []
    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping):
            return None
        if set(row_in.keys()) != {"x"} or set(row_ex.keys()) != {"y"}:
            return None
        try:
            x = float(row_in["x"])
            y = float(row_ex["y"])
        except (TypeError, ValueError, KeyError):
            return None
        pts.append((x, y))

    if n < 2:
        return None

    a: Optional[float] = None
    b: Optional[float] = None
    for i in range(n):
        for j in range(i + 1, n):
            x0, y0 = pts[i]
            x1, y1 = pts[j]
            if math.isclose(x0, x1, rel_tol=0.0, abs_tol=1e-12):
                continue
            a = (y1 - y0) / (x1 - x0)
            b = y0 - a * x0
            break
        if a is not None:
            break

    if a is None:
        return None

    for x, y in pts:
        pred = a * x + b
        if not math.isclose(y, pred, rel_tol=1e-12, abs_tol=1e-9):
            return None

    src = _linear_xy_canonical_source(a, b)
    return ExpertDraftResponse(
        ax_source=src,
        backend_name="linear_xy_fast_path",
        metadata={"fast_path": "linear_xy", "a": a, "b": b},
    )


def _compute_ranking_penalty(source: str, exact_symbolic_task: bool) -> tuple[float, Dict[str, float]]:
    flags = ax_source_metadata_flags(source)
    bd: Dict[str, float] = {}
    total = 0.0
    if exact_symbolic_task and flags.get("uses_neural"):
        bd["neural_on_exact_symbolic"] = _PENALTY_NEURAL_EXACT
        total += _PENALTY_NEURAL_EXACT
    if flags.get("indexed_variable_warning"):
        bd["indexed_variable_warning"] = _PENALTY_INDEXED
        total += _PENALTY_INDEXED
    if flags.get("output_call_warning"):
        bd["output_call_warning"] = _PENALTY_OUTPUT
        total += _PENALTY_OUTPUT
    if flags.get("suspicious_numeric_literal_warning"):
        bd["suspicious_numeric_literal_warning"] = _PENALTY_SUSPICIOUS_NUM
        total += _PENALTY_SUSPICIOUS_NUM
    return total, bd


def _enrich_report_ranking(report: ProgramEvaluationReport, source: str, config: CopilotSearchConfig) -> None:
    """Mutates ``report`` with penalty + adjusted score (candidate selection only)."""
    if config.mode != "predict_rows":
        report.ranking_penalty = 0.0
        report.ranking_penalty_breakdown = {}
        report.adjusted_sort_score = None
        return
    exact = is_exact_symbolic_examples_task(config)
    total, bd = _compute_ranking_penalty(source, exact)
    report.ranking_penalty = total
    report.ranking_penalty_breakdown = bd
    raw = _metric_value(report, config.score_sort_key)
    if raw is not None:
        report.adjusted_sort_score = raw - total
    else:
        report.adjusted_sort_score = None


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


def format_row_mismatches_for_repair(row_comparisons: Sequence[Mapping[str, Any]]) -> str:
    """Deterministic JSON block for repair prompts (worst rows first — see evaluator)."""
    if not row_comparisons:
        return ""
    body = json.dumps([dict(r) for r in row_comparisons], indent=2, sort_keys=True)
    return (
        "## Row-wise mismatches\n\n"
        "Ordered **worst-first** (by `row_max_abs_error`). "
        "Use these concrete input/output deltas to fix coefficients or structure.\n\n"
        "```json\n"
        + body
        + "\n```\n"
    )


def build_repair_error_report(
    *,
    goal: str,
    domain_context: Optional[str],
    current_ax: str,
    evaluation: ProgramEvaluationReport,
    symbolic_exact_hint: bool = False,
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
    if evaluation.row_comparisons:
        parts.append(format_row_mismatches_for_repair(evaluation.row_comparisons))
        parts.append("")
    if symbolic_exact_hint:
        parts.append(
            "## Symbolic mapping hint\n"
            "This task is defined by explicit input/output examples with numeric targets. "
            "Prefer **direct symbolic arithmetic** in `.ax` over `neural(...)` when the mapping can be "
            "written exactly. **Do NOT** use `neural(...)` unless the mapping truly cannot be expressed "
            "symbolically. For affine or clamp-style tasks, use `+`, `-`, `*`, `min`, `max` explicitly.\n\n"
        )
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


def merge_completion_overrides_into_context(
    ctx: Dict[str, Any],
    overrides: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Merge ``temperature`` / ``top_p`` (etc.) into :data:`~axiom.experts.onyx_qwen.COMPLETION_OVERRIDES_CONTEXT_KEY`.

    Stripped from the user JSON prompt by :class:`~axiom.experts.onyx_qwen.OnyxQwenBackend` before building prompts.
    """
    if not overrides:
        return ctx
    out = dict(ctx)
    merged = dict(out.get(COMPLETION_OVERRIDES_CONTEXT_KEY) or {})
    for k, v in overrides.items():
        if v is not None:
            merged[str(k)] = v
    out[COMPLETION_OVERRIDES_CONTEXT_KEY] = merged
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
    """When True, keep repairing successful programs whose metric is below :attr:`metric_repair_if_below` (effective)."""
    metric_repair_if_below: Optional[float] = None
    """If set, repair while the sort key is strictly below this. If unset and ``score_sort_key`` is ``neg_mse``, use
    :data:`DEFAULT_METRIC_REPAIR_THRESHOLD`."""
    predictions_sample_limit: int = 3
    include_trace_snippet: bool = True
    # If True, after each evaluation call expert.summarize_trace (extra latency; failures are ignored).
    summarize_traces: bool = False
    # If set, run_copilot_search writes best.ax, iterations.json, search_report.json under this path.
    artifact_dir: Optional[Path] = None
    # Merged into expert draft/repair JSON context (e.g. benchmark task ids); no effect on evaluation harness.
    draft_context_extras: Dict[str, Any] = field(default_factory=dict)
    repair_context_extras: Dict[str, Any] = field(default_factory=dict)
    #: predict_rows: max rows in :attr:`ProgramEvaluationReport.row_comparisons` (0 = disable).
    row_comparison_limit: int = 32
    #: OpenAI-style ``temperature`` / ``top_p`` for expert draft+repair (Onyx backend only; merged into context key).
    completion_overrides: Optional[Dict[str, Any]] = None
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
    metric_repair_enabled: bool = False
    metric_repair_threshold_effective: Optional[float] = None
    convergence_reason: str = ""
    """One of: ``metric_threshold_met``, ``metric_budget_exhausted``, ``compile_success``, ``failure``."""


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


def _backend_http_failure_report(
    config: CopilotSearchConfig,
    exc: OnyxQwenHTTPError,
    *,
    phase: str,
    source: str = "",
    prior_report: Optional[ProgramEvaluationReport] = None,
) -> ProgramEvaluationReport:
    body = exc.body_snippet or ""
    kind = "backend_oom" if "CUDA error: out of memory" in body else "backend_http"
    detail_obj: Dict[str, Any] = {
        "status_code": int(exc.status_code),
        "body_snippet": body,
        "phase": phase,
    }
    if prior_report is not None:
        detail_obj["prior_evaluation_success"] = bool(prior_report.success)
    return ProgramEvaluationReport(
        success=False,
        source=source,
        compile_stage_reached="expert",
        mode=config.mode,
        failures=[
            ProgramFailure(
                stage="expert",
                kind=kind,
                message=f"Expert backend HTTP {exc.status_code} during {phase}",
                detail=json.dumps(detail_obj, ensure_ascii=False),
            )
        ],
    )


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


def _sort_primary_value(
    report: ProgramEvaluationReport,
    sort_key: Optional[str],
) -> Optional[float]:
    """Prefer :attr:`ProgramEvaluationReport.adjusted_sort_score` when set (Phase 78)."""
    if report.adjusted_sort_score is not None:
        return report.adjusted_sort_score
    return _score_for_sort(report, sort_key)


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
    cs = _sort_primary_value(cand, sort_key)
    bs = _sort_primary_value(best, sort_key)
    if cs is not None and bs is not None:
        return cs > bs
    if cs is not None and bs is None:
        return True
    if cs is None and bs is not None:
        return False
    return False


def _effective_metric_threshold(config: CopilotSearchConfig) -> Optional[float]:
    """Threshold for ``v < thr`` ⇒ keep repairing (``neg_mse`` defaults to :data:`DEFAULT_METRIC_REPAIR_THRESHOLD`)."""
    if not config.repair_valid_with_metrics:
        return None
    if config.metric_repair_if_below is not None:
        return float(config.metric_repair_if_below)
    if config.score_sort_key == "neg_mse":
        return DEFAULT_METRIC_REPAIR_THRESHOLD
    return None


def _needs_metric_repair(config: CopilotSearchConfig, report: ProgramEvaluationReport) -> bool:
    if not report.success or not config.repair_valid_with_metrics:
        return False
    if not report.metrics and not report.program_metrics:
        return False
    thr = _effective_metric_threshold(config)
    if thr is None:
        return False
    v = _metric_value(report, config.score_sort_key)
    if v is None or v >= thr:
        return False
    return True


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
    if is_exact_symbolic_examples_task(config):
        ctx["exact_symbolic_examples_task"] = True
    ctx = merge_completion_overrides_into_context(ctx, config.completion_overrides)
    draft_req = ExpertDraftRequest(goal=config.goal, context=ctx)
    fast = _try_linear_xy_fast_path(config)
    if fast is not None:
        draft_resp = fast
    else:
        try:
            draft_resp = config.expert.draft_program(draft_req)
        except OnyxQwenHTTPError as e:
            fail_rep = _backend_http_failure_report(config, e, phase="draft")
            metric_thr_eff = _effective_metric_threshold(config)
            result = CopilotSearchResult(
                best_source="",
                best_evaluation=fail_rep,
                final_report=fail_rep,
                converged=False,
                iterations=[
                    CopilotIterationRecord(
                        index=0,
                        source="",
                        evaluation=fail_rep,
                        producing_payload=_draft_payload_dict(draft_req),
                        outgoing_repair_error_report=None,
                        producing_expert={},
                        semantic_trace_summary=None,
                    )
                ],
                metric_repair_enabled=bool(config.repair_valid_with_metrics),
                metric_repair_threshold_effective=metric_thr_eff,
                convergence_reason="failure",
            )
            if config.artifact_dir is not None:
                persist_copilot_artifacts(config, result, config.artifact_dir)
            return result
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
    convergence_reason = "failure"
    metric_thr_eff = _effective_metric_threshold(config)

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
                row_comparison_limit=config.row_comparison_limit,
            )
        _enrich_report_ranking(report, source_evaluated, config)
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
            sym_hint = (
                config.mode == "predict_rows"
                and bool(config.expected_rows)
                and is_exact_symbolic_examples_task(config)
            )
            err_full = build_repair_error_report(
                goal=config.goal,
                domain_context=config.domain_context,
                current_ax=source_evaluated,
                evaluation=report,
                symbolic_exact_hint=sym_hint,
            )
            repair_ctx: Dict[str, Any] = build_repair_context(
                example_input_rows=config.example_input_rows,
                expected_rows=config.expected_rows,
                evaluation_mode=config.mode,
                train_tabular_meta=tt_meta,
            )
            if config.repair_context_extras:
                repair_ctx = {**repair_ctx, **dict(config.repair_context_extras)}
            if is_exact_symbolic_examples_task(config):
                repair_ctx["exact_symbolic_examples_task"] = True
            repair_ctx = merge_completion_overrides_into_context(repair_ctx, config.completion_overrides)
            repair_req = ExpertRepairRequest(
                goal=config.goal,
                current_program=source_evaluated,
                error_report=err_full,
                context=repair_ctx,
            )
            ingress_payload = _repair_payload_dict(repair_req)
            try:
                repair_resp = config.expert.repair_program(repair_req)
            except OnyxQwenHTTPError as e:
                fail_rep = _backend_http_failure_report(
                    config, e, phase="repair", source=source_evaluated, prior_report=report
                )
                final_report = fail_rep
                converged = False
                convergence_reason = "failure"
                iterations.append(
                    CopilotIterationRecord(
                        index=i,
                        source=source_evaluated,
                        evaluation=fail_rep,
                        producing_payload=producing,
                        outgoing_repair_error_report=err_full,
                        producing_expert=iter_expert_meta,
                        semantic_trace_summary=sem_summary,
                    )
                )
                break
            current = repair_resp.ax_source
            provenance_meta = expert_response_to_dict(repair_resp, "repair")
        else:
            if report.success:
                converged = not need_metric_repair
                if need_metric_repair:
                    convergence_reason = "metric_budget_exhausted"
                elif (
                    config.repair_valid_with_metrics
                    and config.mode in ("predict_rows", "train_tabular")
                    and bool(report.metrics)
                ):
                    convergence_reason = "metric_threshold_met"
                else:
                    convergence_reason = "compile_success"
            else:
                converged = False
                convergence_reason = "failure"

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
        metric_repair_enabled=bool(config.repair_valid_with_metrics),
        metric_repair_threshold_effective=metric_thr_eff,
        convergence_reason=convergence_reason,
    )
    if config.artifact_dir is not None:
        persist_copilot_artifacts(config, result, config.artifact_dir)
    return result
