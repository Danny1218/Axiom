"""Draft–evaluate–repair search loop (``axiom.copilot.search``)."""

from __future__ import annotations

import json

import pytest

from axiom.compiler.parser import reset_parser
from axiom.copilot import (
    CopilotSearchConfig,
    ProgramCandidate,
    ProgramEvaluationReport,
    TrainTabularParams,
    build_draft_context,
    build_repair_error_report,
    evaluate_program,
    format_failures_for_repair,
    format_metrics_for_repair,
    run_copilot_search,
)
from axiom.copilot.artifacts import evaluation_report_to_dict
from axiom.copilot.benchmarks import default_neg_mse_score_fn
from axiom.copilot.search import DEFAULT_METRIC_REPAIR_THRESHOLD, _is_better
from axiom.experts import (
    ExpertDraftRequest,
    ExpertDraftResponse,
    ExpertRepairRequest,
    ExpertTraceSummaryRequest,
    SemanticExpert,
)

BROKEN_AX = "y = ++++ ;"
GOOD_AX = "y = neural([1.0, 2.0]);"
LOW_Q_AX = "y = 1.0;"
HIGH_Q_AX = "y = 0.5;"
BAD_DOUBLE_AX = "y = x * 1.0;"
GOOD_DOUBLE_AX = "y = x * 2.0;"
CLOSE_DOUBLE_AX = "y = x * 1.0;\n"
EX_IN = [{"x": 1.0}]
EX_EXP = [{"y": 2.0}]


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


class ScriptedExpert:
    """Deterministic expert: fixed draft plus a queue of repair outputs."""

    def __init__(self, draft_source: str, repair_sources: list[str]) -> None:
        self.draft_source = draft_source
        self._repairs = list(repair_sources)
        self.draft_calls: list[ExpertDraftRequest] = []
        self.repair_calls: list[ExpertRepairRequest] = []
        self.summarize_calls: list[ExpertTraceSummaryRequest] = []

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        self.draft_calls.append(request)
        return ExpertDraftResponse(
            ax_source=self.draft_source,
            backend_name="scripted",
            metadata={"call": "draft", "seq": 0},
        )

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        self.repair_calls.append(request)
        if not self._repairs:
            raise AssertionError("unexpected repair_program call")
        nxt = self._repairs.pop(0)
        return ExpertDraftResponse(
            ax_source=nxt,
            backend_name="scripted",
            metadata={"call": "repair", "seq": len(self.repair_calls)},
        )

    def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
        self.summarize_calls.append(request)
        return "ok"


def test_scripted_expert_is_semantic_expert():
    e: SemanticExpert = ScriptedExpert(GOOD_AX, [])
    assert isinstance(e, SemanticExpert)


def test_compile_failure_then_repair_then_success():
    ex = ScriptedExpert(BROKEN_AX, [GOOD_AX])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="emit neural y",
        domain_context="tabular",
        max_iterations=3,
        mode="compile_only",
    )
    out = run_copilot_search(cfg)
    assert len(out.iterations) == 2
    assert out.iterations[0].evaluation.success is False
    assert out.iterations[1].evaluation.success is True
    assert out.best_source.strip() == GOOD_AX.strip()
    assert out.best_evaluation.success is True
    assert out.final_report.success is True
    assert out.converged is True
    assert out.convergence_reason == "compile_success"
    assert len(ex.draft_calls) == 1
    assert ex.draft_calls[0].context["domain_context"] == "tabular"
    assert len(ex.repair_calls) == 1
    rep = ex.repair_calls[0]
    assert "emit neural y" in rep.error_report
    assert BROKEN_AX.strip() in rep.error_report
    assert "syntax" in rep.error_report
    assert rep.context.get("evaluation_mode") == "compile_only"
    assert out.iterations[0].outgoing_repair_error_report is not None
    assert out.iterations[0].producing_payload["type"] == "draft"
    assert out.iterations[1].producing_payload["type"] == "repair"
    assert out.iterations[0].producing_expert["expert_call"] == "draft"
    assert out.iterations[0].producing_expert["metadata"]["call"] == "draft"
    assert out.iterations[1].producing_expert["expert_call"] == "repair"


def test_best_candidate_prefers_valid_over_invalid():
    ex = ScriptedExpert(BROKEN_AX, [LOW_Q_AX])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="constant y",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=[{}],
        expected_rows=[{"y": 0.5}],
        score_fn=lambda p, e: {"quality": 1.0 / (1.0 + abs(float(p[0]["y"]) - float(e[0]["y"])))},
        score_sort_key="quality",
        repair_valid_with_metrics=False,
    )
    out = run_copilot_search(cfg)
    assert out.best_evaluation.success is True
    assert "1.0" in out.best_source
    assert out.iterations[0].evaluation.success is False


def test_best_candidate_higher_score_wins():
    def score_fn(preds, exp):
        err = abs(float(preds[0]["y"]) - float(exp[0]["y"]))
        return {"quality": 1.0 / (1.0 + err)}

    ex = ScriptedExpert(LOW_Q_AX, [HIGH_Q_AX])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="match y",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=[{}],
        expected_rows=[{"y": 0.5}],
        score_fn=score_fn,
        score_sort_key="quality",
        repair_valid_with_metrics=True,
        metric_repair_if_below=0.99,
    )
    out = run_copilot_search(cfg)
    assert len(out.iterations) == 2
    assert out.best_source.strip() == HIGH_Q_AX.strip()
    q0 = out.iterations[0].evaluation.metrics["quality"]
    q1 = out.iterations[1].evaluation.metrics["quality"]
    assert q1 > q0
    assert out.best_evaluation.metrics["quality"] == q1
    assert out.convergence_reason == "metric_threshold_met"


def test_predict_rows_repairs_poor_neg_mse_until_symbolic_fix():
    ex = ScriptedExpert(BAD_DOUBLE_AX, [GOOD_DOUBLE_AX])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="double x",
        max_iterations=4,
        mode="predict_rows",
        example_input_rows=EX_IN,
        expected_rows=EX_EXP,
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=True,
    )
    out = run_copilot_search(cfg)
    assert len(out.iterations) == 2
    assert out.converged is True
    assert out.convergence_reason == "metric_threshold_met"
    assert out.metric_repair_threshold_effective == DEFAULT_METRIC_REPAIR_THRESHOLD
    assert out.best_source.strip() == GOOD_DOUBLE_AX.strip()
    assert len(ex.repair_calls) == 1


def test_repair_prompt_includes_row_mismatches_when_valid_but_wrong_metric():
    """Valid program with wrong coefficients: repair prompt lists per-row predicted vs expected."""
    ex = ScriptedExpert(CLOSE_DOUBLE_AX, [GOOD_DOUBLE_AX])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="Compute y = 2*x from examples",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=EX_IN,
        expected_rows=EX_EXP,
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=True,
    )
    out = run_copilot_search(cfg)
    assert len(ex.repair_calls) == 1
    rep = ex.repair_calls[0].error_report
    assert "## Row-wise mismatches" in rep
    assert '"predicted"' in rep and '"expected"' in rep
    assert '"abs_error"' in rep
    assert "## Symbolic mapping hint" in rep
    assert "symbolic arithmetic" in rep.lower()
    ev0 = out.iterations[0].evaluation
    assert ev0.row_comparisons
    d = evaluation_report_to_dict(ev0)
    json.dumps(d)
    assert "row_comparisons" in d


def test_predict_rows_metric_budget_exhausted_when_still_poor():
    ex = ScriptedExpert(BAD_DOUBLE_AX, [BAD_DOUBLE_AX])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="double x",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=EX_IN,
        expected_rows=EX_EXP,
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=True,
    )
    out = run_copilot_search(cfg)
    assert not out.converged
    assert out.convergence_reason == "metric_budget_exhausted"


def test_summarize_traces_calls_expert_and_sets_iteration_field():
    ex = ScriptedExpert(GOOD_AX, [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="emit neural y",
        max_iterations=1,
        mode="compile_only",
        summarize_traces=True,
    )
    out = run_copilot_search(cfg)
    assert len(ex.summarize_calls) == 1
    assert ex.summarize_calls[0].goal == "emit neural y"
    assert out.iterations[0].semantic_trace_summary == "ok"
    assert out.convergence_reason == "compile_success"


def test_summarize_traces_expert_failure_still_completes_search():
    class _Flaky(ScriptedExpert):
        def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
            raise RuntimeError("no summary")

    ex = _Flaky(GOOD_AX, [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="g",
        max_iterations=1,
        mode="compile_only",
        summarize_traces=True,
    )
    out = run_copilot_search(cfg)
    assert out.iterations[0].evaluation.success is True
    assert out.iterations[0].semantic_trace_summary is None


def test_converged_false_when_still_failing_at_budget():
    ex = ScriptedExpert(BROKEN_AX, [BROKEN_AX])
    cfg = CopilotSearchConfig(expert=ex, goal="x", max_iterations=2, mode="compile_only")
    out = run_copilot_search(cfg)
    assert out.converged is False
    assert out.convergence_reason == "failure"
    assert out.final_report.success is False
    assert len(out.iterations) == 2


def test_build_draft_context_serializable():
    ctx = build_draft_context(
        domain_context="ctx",
        example_input_rows=[{"a": 1}],
        expected_rows=[{"y": 0}],
        train_tabular_meta={"target_var": "y", "train_row_count": 3, "eval_row_count": 1},
    )
    assert ctx["domain_context"] == "ctx"
    assert ctx["example_input_rows"] == [{"a": 1}]
    assert ctx["expected_outputs"] == [{"y": 0}]
    assert ctx["train_tabular"]["target_var"] == "y"
    assert ctx["train_tabular"]["train_row_count"] == 3


def test_train_tabular_search_success_and_draft_context():
    ex = ScriptedExpert("y = neural([x]);\n", [])
    train = [{"x": float(i) * 0.2, "y": float(i) * 0.4} for i in range(10)]
    ev = [{"x": 1.0, "y": 2.0}]
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="fit y from x",
        max_iterations=1,
        mode="train_tabular",
        tabular_train_rows=train,
        tabular_eval_rows=ev,
        tabular_target_var="y",
        tabular_train_params=TrainTabularParams(epochs=60, learning_rate=0.08, batch_size=5),
        tabular_eval_expected_rows=[{"y": 2.0}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    out = run_copilot_search(cfg)
    assert out.best_evaluation.success
    assert out.best_evaluation.mode == "train_tabular"
    assert "eval_mse" in out.best_evaluation.metrics
    assert "neg_mse" in out.best_evaluation.metrics
    assert "train_tabular" in ex.draft_calls[0].context
    assert ex.draft_calls[0].context["train_tabular"]["target_var"] == "y"


def test_format_failures_and_full_repair_report():
    reset_parser()
    bad = evaluate_program(ProgramCandidate(BROKEN_AX), mode="compile_only")
    assert not bad.success
    ff = format_failures_for_repair(bad.failures)
    assert "parse" in ff and "syntax" in ff
    full = build_repair_error_report(
        goal="g",
        domain_context="d",
        current_ax=BROKEN_AX,
        evaluation=bad,
    )
    assert "## Goal" in full and "g" in full
    assert "## Current .ax program" in full
    assert BROKEN_AX.strip() in full


def test_format_metrics_for_repair():
    r = ProgramEvaluationReport(
        success=True,
        source="s",
        compile_stage_reached="predict",
        mode="predict_rows",
        metrics={"mse": 0.25},
        program_metrics=[],
    )
    txt = format_metrics_for_repair(r.metrics, r.program_metrics)
    assert "mse" in txt and "0.25" in txt


def test_is_better_ordering():
    ok = ProgramEvaluationReport(
        success=True, source="a", compile_stage_reached="block", mode="compile_only", metrics={"q": 2.0}
    )
    ok_low = ProgramEvaluationReport(
        success=True, source="b", compile_stage_reached="block", mode="compile_only", metrics={"q": 1.0}
    )
    bad = ProgramEvaluationReport(
        success=False, source="c", compile_stage_reached="parse", mode="compile_only"
    )
    assert _is_better(ok, None, "q") is True
    assert _is_better(ok, ok_low, "q") is True
    assert _is_better(ok_low, ok, "q") is False
    assert _is_better(ok, bad, "q") is True
    assert _is_better(bad, ok, "q") is False
