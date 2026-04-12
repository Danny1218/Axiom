"""Draft–evaluate–repair search loop (``axiom.copilot.search``)."""

from __future__ import annotations

import json
from pathlib import Path

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
from axiom.copilot.search import (
    DEFAULT_METRIC_REPAIR_THRESHOLD,
    _is_better,
    _try_affine_multi_input_fast_path,
    _try_bounded_affine2_fast_path,
    _try_linear_xy_fast_path,
    _try_minmax_blend_fast_path,
    _try_nested_piecewise_identity_cap_fast_path,
    _try_piecewise_threshold_identity_fast_path,
    _try_three_way_maxmin_fast_path,
    _try_two_input_interaction_fast_path,
    is_exact_symbolic_examples_task,
    merge_completion_overrides_into_context,
)
from axiom.experts import (
    ExpertDraftRequest,
    ExpertDraftResponse,
    ExpertRepairRequest,
    ExpertTraceSummaryRequest,
    SemanticExpert,
)
from axiom.experts.onyx_qwen import OnyxQwenHTTPError

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


def _assert_no_forbidden_fast_path_syntax(ax_source: str) -> None:
    for token in ("&&", "||", "else if"):
        assert token not in ax_source


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


def test_repair_prompt_adds_missing_bias_hint_for_constant_offset_rows():
    rep = ProgramEvaluationReport(
        success=True,
        source="y = a + a * b;",
        compile_stage_reached="predict",
        mode="predict_rows",
        row_comparisons=[
            {"predicted": {"y": 0.0}, "expected": {"y": 1.0}},
            {"predicted": {"y": 3.0}, "expected": {"y": 4.0}},
            {"predicted": {"y": -4.0}, "expected": {"y": -3.0}},
        ],
    )
    txt = build_repair_error_report(
        goal="Write .ax so y = a * b + a + 1.0.",
        domain_context="",
        current_ax="y = a + a * b;",
        evaluation=rep,
        symbolic_exact_hint=True,
    )
    assert "near-constant offset" in txt
    assert "missing constant/bias" in txt
    assert "**preserve additive bias exactly**" in txt.lower()


def test_repair_prompt_adds_cross_term_preservation_hint():
    rep = ProgramEvaluationReport(
        success=True,
        source="y = a * 2 + a + 1.0;",
        compile_stage_reached="predict",
        mode="predict_rows",
        row_comparisons=[
            {"inputs": {"a": 0.0, "b": 0.0}, "predicted": {"y": 1.0}, "expected": {"y": 1.0}},
            {"inputs": {"a": 1.0, "b": 0.0}, "predicted": {"y": 2.0}, "expected": {"y": 2.0}},
            {"inputs": {"a": 0.0, "b": 1.0}, "predicted": {"y": 1.0}, "expected": {"y": 1.0}},
            {"inputs": {"a": 1.0, "b": 1.0}, "predicted": {"y": 3.0}, "expected": {"y": 4.0}},
            {"inputs": {"a": 2.0, "b": 2.0}, "predicted": {"y": 5.0}, "expected": {"y": 9.0}},
        ],
    )
    txt = build_repair_error_report(
        goal="Write a valid Axiom .ax program in this repo's DSL that computes y = a * b + a + 1.0;",
        domain_context="",
        current_ax="y = a * 2 + a + 1.0;",
        evaluation=rep,
        symbolic_exact_hint=True,
    )
    assert "**Preserve interaction terms exactly**" in txt
    assert "**Preserve additive bias exactly**" in txt
    assert "**Do not replace interaction terms with boolean guards or branch logic**" in txt
    assert "**Do not replace interaction terms with scaled unary terms**" in txt
    assert "**Missing or wrong interaction term**" in txt
    assert "`a * b`" in txt
    assert "**missing interaction term** (`a * b`)" in txt
    assert "a + a * b" in txt and "a * 2 + a + 1.0" in txt


def test_repair_prompt_adds_distorted_unary_coefficient_hint():
    rep = ProgramEvaluationReport(
        success=True,
        source="y = 2.0 * a + b + 1.0;",
        compile_stage_reached="predict",
        mode="predict_rows",
        row_comparisons=[
            {"inputs": {"a": 0.0, "b": 0.0}, "predicted": {"y": 1.0}, "expected": {"y": 1.0}},
            {"inputs": {"a": 1.0, "b": 0.0}, "predicted": {"y": 3.0}, "expected": {"y": 2.0}},
            {"inputs": {"a": 2.0, "b": 0.0}, "predicted": {"y": 5.0}, "expected": {"y": 3.0}},
            {"inputs": {"a": 3.0, "b": 0.0}, "predicted": {"y": 7.0}, "expected": {"y": 4.0}},
        ],
    )
    txt = build_repair_error_report(
        goal="Write .ax so y = a + b + 1.0.",
        domain_context="",
        current_ax="y = 2.0 * a + b + 1.0;",
        evaluation=rep,
        symbolic_exact_hint=True,
    )
    assert "distorted unary coefficient" in txt.lower()
    assert "missing constant/bias" in txt
    assert "do not replace interaction terms with scaled unary terms" in txt.lower()


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


def test_is_better_uses_adjusted_sort_score_when_set():
    """Neural can look better on raw metric but lose after anti-neural penalty."""
    neural_like = ProgramEvaluationReport(
        success=True,
        source="n",
        compile_stage_reached="predict",
        mode="predict_rows",
        metrics={"neg_mse": -0.001},
        adjusted_sort_score=-2.001,
    )
    symbolic = ProgramEvaluationReport(
        success=True,
        source="s",
        compile_stage_reached="predict",
        mode="predict_rows",
        metrics={"neg_mse": -0.5},
        adjusted_sort_score=-0.5,
    )
    assert _is_better(symbolic, neural_like, "neg_mse") is True
    assert _is_better(neural_like, symbolic, "neg_mse") is False


def test_is_exact_symbolic_examples_task_detects_clamp_goal():
    cfg = CopilotSearchConfig(
        expert=ScriptedExpert(GOOD_AX, []),
        goal="Write .ax that computes risk_score = max(0.0, min(1.0, 0.7 * risk_a + 0.3 * risk_b));",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=[{"risk_a": 1.0}],
        expected_rows=[{"risk_score": 0.7}],
    )
    assert is_exact_symbolic_examples_task(cfg) is True


def test_merge_completion_overrides_into_context():
    from axiom.experts.onyx_qwen import COMPLETION_OVERRIDES_CONTEXT_KEY

    base = {"domain_context": "x"}
    m = merge_completion_overrides_into_context(base, {"temperature": 0.3})
    assert m[COMPLETION_OVERRIDES_CONTEXT_KEY] == {"temperature": 0.3}
    assert base.get(COMPLETION_OVERRIDES_CONTEXT_KEY) is None


def test_run_copilot_search_threads_completion_overrides_to_draft_and_repair():
    from axiom.experts.onyx_qwen import COMPLETION_OVERRIDES_CONTEXT_KEY

    ex = ScriptedExpert(BROKEN_AX, [GOOD_DOUBLE_AX])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="double x",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=EX_IN,
        expected_rows=EX_EXP,
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
        completion_overrides={"temperature": 0.2, "top_p": 0.95},
    )
    run_copilot_search(cfg)
    assert len(ex.draft_calls) == 1 and len(ex.repair_calls) == 1
    want = {"temperature": 0.2, "top_p": 0.95}
    assert ex.draft_calls[0].context.get(COMPLETION_OVERRIDES_CONTEXT_KEY) == want
    assert ex.repair_calls[0].context.get(COMPLETION_OVERRIDES_CONTEXT_KEY) == want


def test_run_copilot_search_omitted_overrides_leaves_context_without_key():
    from axiom.experts.onyx_qwen import COMPLETION_OVERRIDES_CONTEXT_KEY

    ex = ScriptedExpert(GOOD_DOUBLE_AX, [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="double x",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=EX_IN,
        expected_rows=EX_EXP,
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
        completion_overrides=None,
    )
    run_copilot_search(cfg)
    assert COMPLETION_OVERRIDES_CONTEXT_KEY not in ex.draft_calls[0].context


def test_run_copilot_search_draft_http_error_returns_failure_report():
    class _E:
        def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
            raise OnyxQwenHTTPError(500, '{"detail":"internal"}')

        def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
            raise AssertionError("no repair")

        def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
            return ""

    ex: SemanticExpert = _E()  # type: ignore[assignment]
    r = run_copilot_search(
        CopilotSearchConfig(expert=ex, goal="g", max_iterations=2, mode="compile_only")
    )
    assert not r.converged and r.convergence_reason == "failure"
    assert not r.best_evaluation.success
    assert len(r.iterations) == 1
    f = r.best_evaluation.failures[0]
    assert f.kind == "backend_http"
    assert f.detail and '"status_code": 500' in f.detail and "internal" in f.detail


def test_run_copilot_search_repair_http_oom_kind():
    class _E:
        def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
            return ExpertDraftResponse(ax_source=BROKEN_AX, backend_name="t", metadata={})

        def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
            raise OnyxQwenHTTPError(500, "CUDA error: out of memory")

        def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
            return ""

    ex: SemanticExpert = _E()  # type: ignore[assignment]
    r = run_copilot_search(
        CopilotSearchConfig(expert=ex, goal="g", max_iterations=2, mode="compile_only")
    )
    assert len(r.iterations) == 1
    f = r.final_report.failures[0]
    assert f.kind == "backend_oom"
    assert "CUDA error: out of memory" in f.detail


def test_linear_xy_fast_path_skips_expert_and_emits_canonical_source():
    ex = ScriptedExpert("SHOULD_NOT_USE", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="double x",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=[{"x": 0.0}, {"x": 1.0}],
        expected_rows=[{"y": 0.0}, {"y": 2.0}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    out = run_copilot_search(cfg)
    assert len(ex.draft_calls) == 0
    assert out.best_source.strip() == "y = x * 2.0;"
    assert out.iterations[0].producing_expert["backend_name"] == "linear_xy_fast_path"
    assert out.iterations[0].producing_expert["metadata"].get("fast_path") == "linear_xy"
    assert out.converged and out.best_evaluation.success


def test_linear_xy_fast_path_not_used_single_example_row():
    ex = ScriptedExpert(GOOD_DOUBLE_AX, [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="double x",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=EX_IN,
        expected_rows=EX_EXP,
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    run_copilot_search(cfg)
    assert len(ex.draft_calls) == 1


def test_linear_xy_fast_path_not_used_non_collinear():
    ex = ScriptedExpert(GOOD_DOUBLE_AX, [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="double x",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=[{"x": 0.0}, {"x": 1.0}, {"x": 2.0}],
        expected_rows=[{"y": 0.0}, {"y": 1.0}, {"y": 5.0}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    run_copilot_search(cfg)
    assert len(ex.draft_calls) == 1


def test_piecewise_threshold_identity_fast_path_success():
    ex = ScriptedExpert("SHOULD_NOT_DRAFT", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="compute y = x when x > 0 else y = 0.0",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=[{"x": -2.0}, {"x": -0.1}, {"x": 0.0}, {"x": 0.4}, {"x": 1.5}],
        expected_rows=[{"y": 0.0}, {"y": 0.0}, {"y": 0.0}, {"y": 0.4}, {"y": 1.5}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    out = run_copilot_search(cfg)
    assert len(ex.draft_calls) == 0
    assert out.iterations[0].producing_expert["backend_name"] == "piecewise_threshold_identity_fast_path"
    assert out.best_source.strip() == "if (x < 0.0) {\n    y = 0.0;\n} else {\n    y = x;\n}"
    assert out.converged and out.best_evaluation.success


def test_piecewise_threshold_identity_fast_path_ambiguous_returns_none():
    cfg = CopilotSearchConfig(
        expert=ScriptedExpert(GOOD_DOUBLE_AX, []),
        goal="compute thresholded y from x",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=[{"x": 0.0}, {"x": 0.5}, {"x": 2.0}],
        expected_rows=[{"y": 0.0}, {"y": 0.5}, {"y": 2.0}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
    )
    assert _try_piecewise_threshold_identity_fast_path(cfg) is None


def test_piecewise_threshold_identity_fast_path_falls_back_noisy():
    ex = ScriptedExpert(GOOD_DOUBLE_AX, [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="compute thresholded y from x",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=[{"x": -1.0}, {"x": 0.0}, {"x": 1.0}],
        expected_rows=[{"y": 0.0}, {"y": 0.0}, {"y": 1.01}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
    )
    assert _try_piecewise_threshold_identity_fast_path(cfg) is None
    run_copilot_search(cfg)
    assert len(ex.draft_calls) == 1


def test_nested_piecewise_identity_cap_fast_path_success():
    ex = ScriptedExpert("SHOULD_NOT_DRAFT", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="compute nested piecewise clamp so y is capped between 0.0 and 1.0",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=[{"x": -2.0}, {"x": -0.1}, {"x": 0.0}, {"x": 0.4}, {"x": 1.0}, {"x": 1.5}],
        expected_rows=[{"y": 0.0}, {"y": 0.0}, {"y": 0.0}, {"y": 0.4}, {"y": 1.0}, {"y": 1.0}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    out = run_copilot_search(cfg)
    assert len(ex.draft_calls) == 0
    assert out.iterations[0].producing_expert["backend_name"] == "nested_piecewise_identity_cap_fast_path"
    assert out.iterations[0].producing_expert["metadata"].get("fast_path") == "nested_piecewise_identity_cap"
    source = out.best_source.strip()
    assert source == (
        "if (x < 0.0) {\n"
        "    y = 0.0;\n"
        "} else {\n"
        "    if (x < 1.0) {\n"
        "        y = x;\n"
        "    } else {\n"
        "        y = 1.0;\n"
        "    }\n"
        "}"
    )
    _assert_no_forbidden_fast_path_syntax(source)
    assert out.converged and out.best_evaluation.success


def test_nested_piecewise_identity_cap_fast_path_ambiguous_returns_none():
    cfg = CopilotSearchConfig(
        expert=ScriptedExpert(GOOD_DOUBLE_AX, []),
        goal="compute nested piecewise clamp from x",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=[{"x": 0.0}, {"x": 0.4}, {"x": 1.0}],
        expected_rows=[{"y": 0.0}, {"y": 0.4}, {"y": 1.0}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
    )
    assert _try_nested_piecewise_identity_cap_fast_path(cfg) is None


def test_nested_piecewise_identity_cap_fast_path_falls_back_noisy():
    ex = ScriptedExpert(GOOD_DOUBLE_AX, [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="compute nested piecewise clamp so y is capped between 0.0 and 1.0",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=[{"x": -1.0}, {"x": 0.0}, {"x": 0.4}, {"x": 1.0}, {"x": 1.5}],
        expected_rows=[{"y": 0.0}, {"y": 0.0}, {"y": 0.41}, {"y": 1.0}, {"y": 1.0}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
    )
    assert _try_nested_piecewise_identity_cap_fast_path(cfg) is None
    run_copilot_search(cfg)
    assert len(ex.draft_calls) == 1


def test_linear_xy_fast_path_affine_with_intercept():
    ex = ScriptedExpert("noop", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="linear formula",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=[{"x": 0.0}, {"x": 1.0}],
        expected_rows=[{"y": 1.0}, {"y": 3.0}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    out = run_copilot_search(cfg)
    assert len(ex.draft_calls) == 0
    assert out.best_source.strip() == "y = x * 2.0 + 1.0;"


def test_try_linear_xy_fast_path_none_when_keys_not_only_x_y():
    cfg = CopilotSearchConfig(
        expert=ScriptedExpert(GOOD_AX, []),
        goal="double x",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=[{"x": 1.0, "z": 0.0}, {"x": 2.0, "z": 0.0}],
        expected_rows=[{"y": 2.0}, {"y": 4.0}],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
    )
    assert _try_linear_xy_fast_path(cfg) is None


def _load_risk_score_v3_rows():
    p = Path(__file__).resolve().parent.parent / "examples" / "risk_score_v3.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    ex_in = [dict(x["inputs"]) for x in data]
    ex_out = [dict(x["expected"]) for x in data]
    return ex_in, ex_out


def _load_minmax_blend_rows():
    p = Path(__file__).resolve().parent.parent / "examples" / "minmax_blend.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    ex_in = [dict(x["inputs"]) for x in data]
    ex_out = [dict(x["expected"]) for x in data]
    return ex_in, ex_out


def _load_three_way_maxmin_rows():
    p = Path(__file__).resolve().parent.parent / "examples" / "three_way_maxmin.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    ex_in = [dict(x["inputs"]) for x in data]
    ex_out = [dict(x["expected"]) for x in data]
    return ex_in, ex_out


def test_three_way_maxmin_fast_path_exact_success():
    ex_in, ex_out = _load_three_way_maxmin_rows()
    ex = ScriptedExpert("SHOULD_NOT_DRAFT", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="Write .ax so score = max(min(a, b), c).",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=ex_in,
        expected_rows=ex_out,
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    out = run_copilot_search(cfg)
    assert len(ex.draft_calls) == 0
    assert out.iterations[0].producing_expert["backend_name"] == "three_way_maxmin_fast_path"
    assert out.iterations[0].producing_expert["metadata"].get("fast_path") == "three_way_maxmin"
    source = out.best_source.strip()
    assert source == "score = max(min(a, b), c);"
    _assert_no_forbidden_fast_path_syntax(source)
    assert out.converged and out.best_evaluation.success


def test_three_way_maxmin_fast_path_falls_back_when_row_noisy():
    ex_in, ex_out = _load_three_way_maxmin_rows()
    ex_out = [dict(r) for r in ex_out]
    ex_out[-1] = {"score": 5.001}
    ex = ScriptedExpert("score = 0.0;\n", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="Write .ax so score = max(min(a, b), c).",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=ex_in,
        expected_rows=ex_out,
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
    )
    assert _try_three_way_maxmin_fast_path(cfg) is None
    run_copilot_search(cfg)
    assert len(ex.draft_calls) == 1


def test_minmax_blend_fast_path_exact_success():
    ex_in, ex_out = _load_minmax_blend_rows()
    ex = ScriptedExpert("SHOULD_NOT_DRAFT", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="Write .ax so score = max(0.0, min(a + b, 1.0)).",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=ex_in,
        expected_rows=ex_out,
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    out = run_copilot_search(cfg)
    assert len(ex.draft_calls) == 0
    assert out.iterations[0].producing_expert["backend_name"] == "minmax_blend_fast_path"
    assert out.iterations[0].producing_expert["metadata"].get("fast_path") == "minmax_blend"
    source = out.best_source.strip()
    assert source == "score = max(0.0, min(a + b, 1.0));"
    _assert_no_forbidden_fast_path_syntax(source)
    assert out.converged and out.best_evaluation.success


def test_minmax_blend_fast_path_falls_back_when_row_noisy():
    ex_in, ex_out = _load_minmax_blend_rows()
    ex_out = [dict(r) for r in ex_out]
    ex_out[1] = {"score": 0.7005}
    ex = ScriptedExpert("score = 0.0;\n", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="Write .ax so score = max(0.0, min(a + b, 1.0)).",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=ex_in,
        expected_rows=ex_out,
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
    )
    assert _try_minmax_blend_fast_path(cfg) is None
    run_copilot_search(cfg)
    assert len(ex.draft_calls) == 1


def test_bounded_affine2_fast_path_risk_score_v3_exact():
    ex_in, ex_out = _load_risk_score_v3_rows()
    ex = ScriptedExpert("SHOULD_NOT_DRAFT", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="risk_score weighted blend formula",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=ex_in,
        expected_rows=ex_out,
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    out = run_copilot_search(cfg)
    assert len(ex.draft_calls) == 0
    assert out.iterations[0].producing_expert["backend_name"] == "bounded_affine2_fast_path"
    want = "risk_score = max(0.0, min(1.0, 0.7 * risk_a + 0.3 * risk_b));"
    assert out.best_source.strip() == want
    assert out.converged and out.best_evaluation.success


def test_bounded_affine2_fast_path_falls_back_when_row_noisy():
    ex_in, ex_out = _load_risk_score_v3_rows()
    ex_out = [dict(r) for r in ex_out]
    ex_out[-1] = {"risk_score": 0.999}  # last row should be 1.0 for consistent blend
    ex = ScriptedExpert(GOOD_DOUBLE_AX, [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="risk_score weighted blend formula",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=ex_in,
        expected_rows=ex_out,
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    run_copilot_search(cfg)
    assert len(ex.draft_calls) >= 1


def test_bounded_affine2_fast_path_falls_back_insufficient_strict_interior():
    ex = ScriptedExpert(GOOD_DOUBLE_AX, [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="risk_score clamp",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=[
            {"risk_a": 0.0, "risk_b": 0.0},
            {"risk_a": 1.0, "risk_b": 0.0},
            {"risk_a": 0.0, "risk_b": 1.0},
        ],
        expected_rows=[
            {"risk_score": 0.0},
            {"risk_score": 0.7},
            {"risk_score": 0.3},
        ],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    assert _try_bounded_affine2_fast_path(cfg) is None
    run_copilot_search(cfg)
    assert len(ex.draft_calls) == 1


def test_bounded_affine2_clamp_edges_validated_on_v3_subset():
    """Rows at 0, 1, and interior — clamp must match after inference."""
    ex_in, ex_out = _load_risk_score_v3_rows()
    # First 12 rows include clamp-to-0, interior, and clamp-to-1 cases
    ex_in, ex_out = ex_in[:12], ex_out[:12]
    r = _try_bounded_affine2_fast_path(
        CopilotSearchConfig(
            expert=ScriptedExpert(GOOD_AX, []),
            goal="risk_score",
            max_iterations=1,
            mode="predict_rows",
            example_input_rows=ex_in,
            expected_rows=ex_out,
            score_fn=default_neg_mse_score_fn(),
            score_sort_key="neg_mse",
        )
    )
    assert r is not None
    assert "max(0.0, min(1.0" in r.ax_source
    md = r.metadata
    assert md.get("a") == pytest.approx(0.7) and md.get("b") == pytest.approx(0.3)


def test_two_input_interaction_fast_path_exact_cross_term_success():
    ex = ScriptedExpert("SHOULD_NOT_DRAFT", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="Write .ax so y = a * b + a.",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=[
            {"a": 0.0, "b": 0.0},
            {"a": 1.0, "b": 0.0},
            {"a": 0.0, "b": 1.0},
            {"a": 1.0, "b": 1.0},
            {"a": 2.0, "b": 3.0},
        ],
        expected_rows=[
            {"y": 0.0},
            {"y": 1.0},
            {"y": 0.0},
            {"y": 2.0},
            {"y": 8.0},
        ],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    out = run_copilot_search(cfg)
    assert len(ex.draft_calls) == 0
    assert out.iterations[0].producing_expert["backend_name"] == "two_input_interaction_fast_path"
    assert out.iterations[0].producing_expert["metadata"].get("fast_path") == "two_input_interaction"
    source = out.best_source.strip()
    assert source == "y = a * b + a;"
    _assert_no_forbidden_fast_path_syntax(source)
    assert out.converged and out.best_evaluation.success


def test_two_input_interaction_fast_path_exact_cross_term_with_bias_success():
    r = _try_two_input_interaction_fast_path(
        CopilotSearchConfig(
            expert=ScriptedExpert(GOOD_AX, []),
            goal="Write .ax so y = a * b + a + 1.0.",
            max_iterations=1,
            mode="predict_rows",
            example_input_rows=[
                {"a": 0.0, "b": 0.0},
                {"a": 1.0, "b": 0.0},
                {"a": 0.0, "b": 1.0},
                {"a": 1.0, "b": 1.0},
                {"a": 2.0, "b": 2.0},
            ],
            expected_rows=[
                {"y": 1.0},
                {"y": 2.0},
                {"y": 1.0},
                {"y": 3.0},
                {"y": 7.0},
            ],
            score_fn=default_neg_mse_score_fn(),
            score_sort_key="neg_mse",
        )
    )
    assert r is not None
    source = r.ax_source.strip()
    assert source == "y = a * b + a + 1.0;"
    _assert_no_forbidden_fast_path_syntax(source)
    assert r.metadata["w_ab"] == pytest.approx(1.0)
    assert r.metadata["w_a"] == pytest.approx(1.0)
    assert r.metadata["w_b"] == pytest.approx(0.0)
    assert r.metadata["bias"] == pytest.approx(1.0)


def test_two_input_interaction_fast_path_falls_back_when_noisy():
    ex = ScriptedExpert("y = 0.0;\n", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="Write .ax so y = a * b + a + 1.0.",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=[
            {"a": 0.0, "b": 0.0},
            {"a": 1.0, "b": 0.0},
            {"a": 0.0, "b": 1.0},
            {"a": 1.0, "b": 1.0},
            {"a": 2.0, "b": 2.0},
        ],
        expected_rows=[
            {"y": 1.0},
            {"y": 2.0},
            {"y": 1.0},
            {"y": 3.0},
            {"y": 7.001},
        ],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
        repair_valid_with_metrics=False,
    )
    assert _try_two_input_interaction_fast_path(cfg) is None
    run_copilot_search(cfg)
    assert len(ex.draft_calls) == 1


def test_two_input_interaction_fast_path_returns_none_when_ambiguous():
    cfg = CopilotSearchConfig(
        expert=ScriptedExpert(GOOD_AX, []),
        goal="Write .ax so y = a * b + a + 1.0.",
        max_iterations=1,
        mode="predict_rows",
        example_input_rows=[
            {"a": 0.0, "b": 0.0},
            {"a": 1.0, "b": 0.0},
            {"a": 2.0, "b": 0.0},
            {"a": 3.0, "b": 0.0},
        ],
        expected_rows=[
            {"y": 1.0},
            {"y": 2.0},
            {"y": 3.0},
            {"y": 4.0},
        ],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
    )
    assert _try_two_input_interaction_fast_path(cfg) is None


def test_affine_multi_input_fast_path_exact_three_input_success():
    ex = ScriptedExpert("SHOULD_NOT_DRAFT", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="compute weighted blend score",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=[
            {"a": 1.0, "b": 0.0, "c": 0.0},
            {"a": 0.0, "b": 1.0, "c": 0.0},
            {"a": 0.0, "b": 0.0, "c": 1.0},
            {"a": 1.0, "b": 1.0, "c": 1.0},
        ],
        expected_rows=[
            {"score": 0.5},
            {"score": 0.3},
            {"score": 0.2},
            {"score": 1.0},
        ],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
    )
    out = run_copilot_search(cfg)
    assert len(ex.draft_calls) == 0
    assert out.iterations[0].producing_expert["backend_name"] == "affine_multi_input_fast_path"
    assert out.best_source.strip() == "score = 0.5 * a + 0.3 * b + 0.2 * c;"


def test_affine_multi_input_fast_path_affine_with_bias_success():
    r = _try_affine_multi_input_fast_path(
        CopilotSearchConfig(
            expert=ScriptedExpert(GOOD_AX, []),
            goal="linear formula with bias",
            max_iterations=1,
            mode="predict_rows",
            example_input_rows=[
                {"a": 0.0, "b": 0.0, "c": 0.0},
                {"a": 1.0, "b": 0.0, "c": 0.0},
                {"a": 0.0, "b": 1.0, "c": 0.0},
                {"a": 0.0, "b": 0.0, "c": 1.0},
                {"a": 1.0, "b": 1.0, "c": 1.0},
            ],
            expected_rows=[
                {"score": 0.1},
                {"score": 0.6},
                {"score": 0.4},
                {"score": 0.3},
                {"score": 1.1},
            ],
            score_fn=default_neg_mse_score_fn(),
            score_sort_key="neg_mse",
        )
    )
    assert r is not None
    assert r.ax_source.strip() == "score = 0.5 * a + 0.3 * b + 0.2 * c + 0.1;"


def test_affine_multi_input_fast_path_falls_back_when_noisy():
    ex = ScriptedExpert("score = 0.0;\n", [])
    cfg = CopilotSearchConfig(
        expert=ex,
        goal="compute weighted blend score",
        max_iterations=2,
        mode="predict_rows",
        example_input_rows=[
            {"a": 1.0, "b": 0.0, "c": 0.0},
            {"a": 0.0, "b": 1.0, "c": 0.0},
            {"a": 0.0, "b": 0.0, "c": 1.0},
            {"a": 1.0, "b": 1.0, "c": 1.0},
            {"a": 2.0, "b": 2.0, "c": 2.0},
        ],
        expected_rows=[
            {"score": 0.5},
            {"score": 0.3},
            {"score": 0.2},
            {"score": 1.001},
            {"score": 2.0},
        ],
        score_fn=default_neg_mse_score_fn(),
        score_sort_key="neg_mse",
    )
    run_copilot_search(cfg)
    assert len(ex.draft_calls) == 1


def test_affine_multi_input_fast_path_fixture_emits_canonical_without_indexed_access():
    p = Path(__file__).resolve().parent.parent / "examples" / "three_input_affine_fast_path.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    ex_in = [dict(x["inputs"]) for x in data]
    ex_out = [dict(x["expected"]) for x in data]
    out = run_copilot_search(
        CopilotSearchConfig(
            expert=ScriptedExpert("SHOULD_NOT_DRAFT", []),
            goal="score affine blend",
            max_iterations=1,
            mode="predict_rows",
            example_input_rows=ex_in,
            expected_rows=ex_out,
            score_fn=default_neg_mse_score_fn(),
            score_sort_key="neg_mse",
        )
    )
    src = out.best_source.strip()
    assert src == "score = 0.5 * a + 0.3 * b + 0.2 * c;"
    assert "[0]" not in src and "[1]" not in src and "[2]" not in src
