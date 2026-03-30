"""Compile / validate / lightweight predict harness for candidate ``.ax`` source (in-memory)."""

from __future__ import annotations

import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from axiom.api import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax
from axiom.copilot.models import (
    EvaluationMode,
    ProgramCandidate,
    ProgramEvaluationReport,
    ProgramFailure,
    ProgramMetric,
    ProgramValidationReport,
)
from axiom.engine.block_executor import InterpretedBlock

CompileStage = str  # "none" | "parse" | "ir" | "block" | "predict"

_MAX_ABI_VARS = 256


def _failure(stage: str, kind: str, exc: BaseException) -> ProgramFailure:
    return ProgramFailure(
        stage=stage,
        kind=kind,
        message=str(exc),
        detail=type(exc).__name__,
    )


def _try_compile(
    source: str, *, max_unroll: int
) -> Tuple[CompileStage, Optional[InterpretedBlock], List[ProgramFailure], List[str]]:
    """Return ``(stage_reached, block_or_none, failures, warnings)``."""
    failures: List[ProgramFailure] = []
    warnings: List[str] = []
    stage: CompileStage = "none"
    try:
        tree = parse_ax(source)
        stage = "parse"
    except Exception as e:
        failures.append(_failure("parse", "syntax", e))
        return "parse", None, failures, warnings

    try:
        ir = ast_to_ir(tree)
        stage = "ir"
    except Exception as e:
        failures.append(_failure("ir", "ir", e))
        return "ir", None, failures, warnings

    try:
        abi = extract_global_abi(ir, max_vars=_MAX_ABI_VARS)
        aw = extract_abi_widths(ir, max_vars=_MAX_ABI_VARS)
        block = InterpretedBlock(ir, abi, abi_widths=aw, max_unroll=max_unroll)
        block.eval()
        stage = "block"
    except Exception as e:
        failures.append(_failure("block", "compile", e))
        return "block", None, failures, warnings

    return stage, block, failures, warnings


def validate_program(
    candidate: ProgramCandidate,
    *,
    max_unroll: int = 8,
) -> ProgramValidationReport:
    """Parse and lower ``candidate.source`` to an ``InterpretedBlock``; never raises for user errors."""
    stage, _block, failures, warnings = _try_compile(candidate.source, max_unroll=max_unroll)
    ok = not failures and stage == "block"
    return ProgramValidationReport(
        success=ok,
        source=candidate.source,
        compile_stage_reached=stage,
        failures=failures,
        warnings=list(warnings),
    )


def evaluate_program(
    candidate: ProgramCandidate,
    *,
    mode: EvaluationMode = "compile_only",
    max_unroll: int = 8,
    input_rows: Optional[Sequence[Dict[str, Any]]] = None,
    expected_rows: Optional[Sequence[Dict[str, Any]]] = None,
    score_fn: Optional[Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, float]]] = None,
    predictions_sample_limit: int = 3,
    include_trace_snippet: bool = True,
) -> ProgramEvaluationReport:
    """Validate and optionally run batched ``predict`` with an optional ``score_fn``.

    * ``compile_only`` — same as :func:`validate_program` (no predict).
    * ``predict_rows`` — requires non-empty ``input_rows``; uses :class:`~axiom.api.AxiomModel`.
    * ``train_tabular`` — **not implemented**; returns ``success=False`` with a structured failure
      (no second training stack; use the normal ``axiom train`` / trainer APIs for real training).

    ``score_fn(predictions, expected) -> dict[str, float]`` runs only when both ``expected_rows`` and
    ``score_fn`` are provided; lengths must match ``input_rows``.
    """
    source = candidate.source
    failures: List[ProgramFailure] = []
    warnings: List[str] = []
    metrics: Dict[str, float] = {}
    program_metrics: List[ProgramMetric] = []
    predictions_sample: Optional[List[Dict[str, Any]]] = None
    trace_snippet: Optional[Dict[str, Any]] = None

    if mode == "train_tabular":
        return ProgramEvaluationReport(
            success=False,
            source=source,
            compile_stage_reached="none",
            mode=mode,
            failures=[
                ProgramFailure(
                    stage="train",
                    kind="unsupported",
                    message=(
                        "train_tabular is not implemented in the copilot harness; "
                        "use compile_only or predict_rows, or call EvolutionaryTrainer / axiom train directly."
                    ),
                    detail="NotImplemented",
                )
            ],
            warnings=warnings,
            metrics=metrics,
            program_metrics=program_metrics,
        )

    stage, block, compile_failures, compile_warnings = _try_compile(source, max_unroll=max_unroll)
    failures.extend(compile_failures)
    warnings.extend(compile_warnings)

    if failures or block is None:
        return ProgramEvaluationReport(
            success=False,
            source=source,
            compile_stage_reached=stage,
            mode=mode,
            failures=failures,
            warnings=warnings,
            metrics=metrics,
            program_metrics=program_metrics,
        )

    if mode == "compile_only":
        return ProgramEvaluationReport(
            success=True,
            source=source,
            compile_stage_reached=stage,
            mode=mode,
            failures=[],
            warnings=warnings,
            metrics=metrics,
            program_metrics=program_metrics,
        )

    # predict_rows
    if input_rows is None or len(input_rows) == 0:
        failures.append(
            ProgramFailure(
                stage="predict",
                kind="value",
                message="predict_rows requires a non-empty input_rows sequence.",
                detail="ValueError",
            )
        )
        return ProgramEvaluationReport(
            success=False,
            source=source,
            compile_stage_reached=stage,
            mode=mode,
            failures=failures,
            warnings=warnings,
            metrics=metrics,
            program_metrics=program_metrics,
        )

    rows_list = [dict(r) for r in input_rows]
    model = AxiomModel(block)

    try:
        preds_raw = model.predict(rows_list)
    except Exception as e:
        failures.append(_failure("predict", "runtime", e))
        tb = traceback.format_exc(limit=6)
        warnings.append(f"predict traceback (truncated):\n{tb}")
        return ProgramEvaluationReport(
            success=False,
            source=source,
            compile_stage_reached=stage,
            mode=mode,
            failures=failures,
            warnings=warnings,
            metrics=metrics,
            program_metrics=program_metrics,
        )

    preds_list_out: List[Dict[str, Any]] = preds_raw if isinstance(preds_raw, list) else [preds_raw]

    stage = "predict"
    lim = max(0, int(predictions_sample_limit))
    if lim:
        predictions_sample = preds_list_out[:lim]

    if include_trace_snippet and rows_list:
        try:
            trace_snippet = model.explain(rows_list[0])
        except Exception as e:
            warnings.append(f"explain(first_row) skipped: {e}")

    if score_fn is not None:
        if expected_rows is None:
            warnings.append("score_fn provided but expected_rows is missing; skipping metrics.")
        else:
            exp_list = [dict(r) for r in expected_rows]
            if len(exp_list) != len(rows_list):
                failures.append(
                    ProgramFailure(
                        stage="predict",
                        kind="value",
                        message=(
                            f"expected_rows length {len(exp_list)} != input_rows length {len(rows_list)}."
                        ),
                        detail="ValueError",
                    )
                )
                return ProgramEvaluationReport(
                    success=False,
                    source=source,
                    compile_stage_reached=stage,
                    mode=mode,
                    failures=failures,
                    warnings=warnings,
                    metrics=metrics,
                    program_metrics=program_metrics,
                    predictions_sample=predictions_sample,
                    trace_snippet=trace_snippet,
                )
            try:
                raw_scores = score_fn(
                    preds_list_out,
                    exp_list,
                )
                metrics = {str(k): float(v) for k, v in raw_scores.items()}
                program_metrics = [ProgramMetric(name=k, value=v) for k, v in metrics.items()]
            except Exception as e:
                failures.append(_failure("predict", "metric", e))

    success = not failures
    return ProgramEvaluationReport(
        success=success,
        source=source,
        compile_stage_reached=stage,
        mode=mode,
        failures=failures,
        warnings=warnings,
        metrics=metrics,
        program_metrics=program_metrics,
        predictions_sample=predictions_sample,
        trace_snippet=trace_snippet,
    )
