"""``axiom.copilot.benchmarks`` — NL→``.ax`` internal benchmark harness (Phase 65)."""

from __future__ import annotations

import json

import pytest

from axiom.compiler.parser import reset_parser
from axiom.copilot.benchmarks import (
    BENCHMARK_SUITE_SCHEMA_VERSION,
    DEFAULT_BENCHMARK_TASKS,
    BenchmarkDispatchExpert,
    BenchmarkTask,
    benchmark_suite_to_dict,
    benchmark_tasks_from_json_dict,
    compile_success,
    default_benchmark_tasks_json_path,
    default_neg_mse_score_fn,
    load_benchmark_tasks_json_path,
    metric_success,
    run_benchmark_draft_only,
    run_benchmark_search,
    run_benchmark_suite,
    summarize_rates,
)
from axiom.copilot.models import ProgramEvaluationReport
from axiom.copilot.search import CopilotSearchConfig, run_copilot_search
from axiom.experts import ExpertDraftRequest, ExpertDraftResponse, ExpertRepairRequest, SemanticExpert


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


def test_default_benchmark_tasks_cover_finance_risk_loop():
    ids = {t.id for t in DEFAULT_BENCHMARK_TASKS}
    assert ids == {"finance_threshold_policy", "simple_risk_score", "looped_numeric_counter"}
    assert DEFAULT_BENCHMARK_TASKS[0].evaluation_mode == "predict_rows"
    assert DEFAULT_BENCHMARK_TASKS[2].evaluation_mode == "compile_only"


def test_benchmark_task_predict_rows_requires_rows():
    with pytest.raises(ValueError, match="predict_rows requires"):
        BenchmarkTask(
            id="bad",
            title="bad",
            goal="g",
            evaluation_mode="predict_rows",
            example_input_rows=(),
            expected_rows=(),
        )


def test_benchmark_task_row_count_mismatch():
    with pytest.raises(ValueError, match="count mismatch"):
        BenchmarkTask(
            id="bad",
            title="bad",
            goal="g",
            evaluation_mode="predict_rows",
            example_input_rows=({"a": 1},),
            expected_rows=({"b": 1}, {"b": 2}),
        )


def test_default_neg_mse_score_fn():
    fn = default_neg_mse_score_fn()
    s = fn([{"y": 1.0}], [{"y": 1.0}])
    assert s["neg_mse"] == 0.0


def test_compile_and_metric_success_helpers():
    ok = ProgramEvaluationReport(
        success=True,
        source="s",
        compile_stage_reached="predict",
        mode="predict_rows",
        metrics={"neg_mse": 0.0},
    )
    t2 = BenchmarkTask(
        id="t2",
        title="t2",
        goal="g",
        evaluation_mode="predict_rows",
        example_input_rows=({"x": 1.0},),
        expected_rows=({"y": 1.0},),
        metric_pass_min=("neg_mse", -0.1),
    )
    assert compile_success(ok) is True
    assert metric_success(t2, ok) is True
    bad_m = ProgramEvaluationReport(
        success=True,
        source="s",
        compile_stage_reached="predict",
        mode="predict_rows",
        metrics={"neg_mse": -99.0},
    )
    assert metric_success(t2, bad_m) is False


def test_summarize_rates_empty_and_full():
    assert summarize_rates([]).task_count == 0
    ex = BenchmarkDispatchExpert()
    rec = run_benchmark_draft_only(ex, DEFAULT_BENCHMARK_TASKS[2])
    s = summarize_rates([rec])
    assert s.task_count == 1 and s.compile_success_rate == 1.0 and s.metric_success_rate == 1.0


def test_dispatch_expert_requires_benchmark_task_id():
    ex: SemanticExpert = BenchmarkDispatchExpert()
    with pytest.raises(ValueError, match="benchmark_task_id"):
        ex.draft_program(ExpertDraftRequest(goal="g", context={}))


def test_run_benchmark_draft_only_and_search_with_dispatch():
    ex = BenchmarkDispatchExpert()
    t = DEFAULT_BENCHMARK_TASKS[1]
    dr = run_benchmark_draft_only(ex, t)
    assert dr.compile_ok and dr.metric_ok
    sr = run_benchmark_search(ex, t, max_iterations=2)
    assert sr.compile_ok and sr.iterations_run == 1


def test_run_benchmark_suite_dict_roundtrip():
    ex = BenchmarkDispatchExpert()
    suite = run_benchmark_suite(ex, tasks=DEFAULT_BENCHMARK_TASKS, max_iterations=2)
    assert suite.draft_summary and suite.search_summary
    assert suite.draft_summary.task_count == 3
    assert suite.run_draft and suite.run_search
    d = benchmark_suite_to_dict(suite)
    assert d["schema_version"] == BENCHMARK_SUITE_SCHEMA_VERSION
    assert d["kind"] == "axiom.copilot.benchmark_suite"
    assert d["run_options"] == {"draft": True, "search": True}
    json.dumps(d)
    assert d["draft_summary"]["compile_success_rate"] == 1.0
    assert d["search_summary"]["metric_success_rate"] == 1.0


def test_run_benchmark_suite_requires_draft_or_search():
    ex = BenchmarkDispatchExpert()
    with pytest.raises(ValueError, match="run_draft and/or"):
        run_benchmark_suite(ex, tasks=DEFAULT_BENCHMARK_TASKS[:1], run_draft=False, run_search=False)


def test_run_benchmark_suite_draft_only():
    ex = BenchmarkDispatchExpert()
    suite = run_benchmark_suite(
        ex, tasks=DEFAULT_BENCHMARK_TASKS[:1], max_iterations=2, run_draft=True, run_search=False
    )
    assert suite.draft_summary is not None
    assert suite.search_summary is None
    assert suite.tasks[0].draft_only is not None
    assert suite.tasks[0].search is None
    d = benchmark_suite_to_dict(suite)
    assert d["search_summary"] is None
    assert d["tasks"][0]["search"] is None
    assert d["run_options"] == {"draft": True, "search": False}


def test_run_benchmark_suite_search_only():
    ex = BenchmarkDispatchExpert()
    suite = run_benchmark_suite(
        ex, tasks=(DEFAULT_BENCHMARK_TASKS[2],), run_draft=False, run_search=True, max_iterations=2
    )
    assert suite.draft_summary is None
    assert suite.search_summary is not None
    d = benchmark_suite_to_dict(suite)
    assert d["draft_summary"] is None
    assert d["tasks"][0]["draft_only"] is None


def test_search_beats_draft_when_first_draft_broken():
    ex = BenchmarkDispatchExpert(broken_draft_by_task={"finance_threshold_policy": "y = ++++ ;\n"})
    fin = DEFAULT_BENCHMARK_TASKS[0]
    suite = run_benchmark_suite(ex, tasks=(fin,), max_iterations=3)
    cmp = suite.tasks[0]
    assert cmp.draft_only.compile_ok is False
    assert cmp.search.compile_ok is True
    assert cmp.search.iterations_run >= 2


def test_benchmark_tasks_from_json_dict_and_bundled_path():
    path = default_benchmark_tasks_json_path()
    assert path.is_file()
    tasks = load_benchmark_tasks_json_path(path)
    assert len(tasks) == 1 and tasks[0].id == "risk_from_json_fixture"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    same = benchmark_tasks_from_json_dict(raw)
    assert same[0].goal == tasks[0].goal


def test_benchmark_tasks_from_json_invalid():
    with pytest.raises(ValueError, match="root must"):
        benchmark_tasks_from_json_dict([])
    with pytest.raises(ValueError, match="'tasks' array"):
        benchmark_tasks_from_json_dict({})


def test_draft_context_extras_reach_expert():
    """Regression: benchmark_task_id must flow through search repair context."""
    captured: list[dict] = []

    class Spy(BenchmarkDispatchExpert):
        def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
            captured.append(dict(request.context))
            return super().repair_program(request)

    ex = Spy(broken_draft_by_task={"simple_risk_score": "y = ++++ ;\n"})
    t = DEFAULT_BENCHMARK_TASKS[1]
    run_benchmark_search(ex, t, max_iterations=3)
    assert captured, "repair should run"
    assert captured[0].get("benchmark_task_id") == "simple_risk_score"


def test_copilot_search_config_context_extras_merged():
    from axiom.experts import ExpertTraceSummaryRequest

    class E:
        def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
            assert request.context.get("extra_k") == 1
            assert request.context.get("domain_context") == ""
            return ExpertDraftResponse(ax_source="y = ++++;\n", backend_name="e")

        def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
            assert request.context.get("extra_k") == 1
            return ExpertDraftResponse(ax_source="y = 1.0;\n", backend_name="e")

        def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
            return ""

    cfg = CopilotSearchConfig(
        expert=E(),
        goal="g",
        max_iterations=2,
        mode="compile_only",
        draft_context_extras={"extra_k": 1},
        repair_context_extras={"extra_k": 1},
    )
    run_copilot_search(cfg)
