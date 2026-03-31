"""``axiom.copilot`` compile / validate / predict harness."""

from __future__ import annotations

import json

import pytest

from axiom.compiler.parser import reset_parser
from axiom.copilot import (
    ProgramCandidate,
    ProgramEvaluationReport,
    ProgramFailure,
    ProgramMetric,
    ProgramValidationReport,
    TrainTabularParams,
    evaluate_program,
    validate_program,
)
from axiom.copilot.artifacts import evaluation_report_to_dict
from axiom.copilot.benchmarks import default_neg_mse_score_fn

GOOD_AX = "y = neural([1.0, 2.0]);"


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


def test_program_candidate_frozen_fields():
    c = ProgramCandidate(source="x=1;", id="a")
    assert c.source == "x=1;"
    assert c.id == "a"


def test_validate_success_reaches_block():
    r = validate_program(ProgramCandidate(GOOD_AX))
    assert isinstance(r, ProgramValidationReport)
    assert r.success is True
    assert r.compile_stage_reached == "block"
    assert r.failures == []


def test_validate_syntax_error_structured():
    r = validate_program(ProgramCandidate("y = ++++ ;"))
    assert r.success is False
    assert r.compile_stage_reached == "parse"
    assert len(r.failures) == 1
    f = r.failures[0]
    assert isinstance(f, ProgramFailure)
    assert f.stage == "parse"
    assert f.kind == "syntax"


def test_validate_ir_error_structured():
    r = validate_program(ProgramCandidate("return 1;"))
    assert r.success is False
    assert r.compile_stage_reached == "ir"
    assert r.failures[0].stage == "ir"
    assert r.failures[0].kind == "ir"


def test_evaluate_compile_only_matches_validate():
    c = ProgramCandidate(GOOD_AX)
    v = validate_program(c)
    e = evaluate_program(c, mode="compile_only")
    assert e.success == v.success
    assert e.compile_stage_reached == v.compile_stage_reached
    assert len(e.failures) == len(v.failures)


def test_train_tabular_requires_target_var():
    r = evaluate_program(ProgramCandidate(GOOD_AX), mode="train_tabular")
    assert r.success is False
    assert r.mode == "train_tabular"
    assert any("target_var" in f.message.lower() for f in r.failures)


def test_train_tabular_smoke_with_params():
    rows = [{"x": 0.5, "y": 1.0}, {"x": 1.0, "y": 2.0}]
    r = evaluate_program(
        ProgramCandidate("y = neural([x]);"),
        mode="train_tabular",
        target_var="y",
        train_rows=rows,
        eval_rows=rows,
        train_tabular_params=TrainTabularParams(epochs=5, batch_size=2),
        include_trace_snippet=False,
    )
    assert r.success
    assert r.compile_stage_reached == "train"
    assert "eval_mse" in r.metrics


def test_predict_rows_batch_score_fn():
    def score_fn(preds, exp):
        err = sum((float(p["y"]) - float(e["y"])) ** 2 for p, e in zip(preds, exp))
        return {"batch_sse": err}

    r = evaluate_program(
        ProgramCandidate(GOOD_AX),
        mode="predict_rows",
        input_rows=[{}, {}],
        expected_rows=[{"y": 0.0}, {"y": 1.0}],
        score_fn=score_fn,
        include_trace_snippet=False,
    )
    assert r.success is True
    assert "batch_sse" in r.metrics
    assert len(r.predictions_sample) == 2


def test_predict_rows_success_metrics_and_trace():
    def score_fn(preds, exp):
        assert len(preds) == len(exp)
        return {"mse": (preds[0]["y"] - exp[0]["y"]) ** 2}

    r = evaluate_program(
        ProgramCandidate(GOOD_AX),
        mode="predict_rows",
        input_rows=[{}],
        expected_rows=[{"y": 0.5}],
        score_fn=score_fn,
        include_trace_snippet=True,
    )
    assert r.success is True
    assert r.compile_stage_reached == "predict"
    assert "mse" in r.metrics
    assert any(m.name == "mse" for m in r.program_metrics)
    assert isinstance(r.predictions_sample, list)
    assert "y" in r.predictions_sample[0]
    assert r.trace_snippet is not None
    assert isinstance(r.trace_snippet, dict)


def test_predict_rows_empty_input_fails():
    r = evaluate_program(ProgramCandidate(GOOD_AX), mode="predict_rows", input_rows=[])
    assert r.success is False
    assert any(f.stage == "predict" for f in r.failures)


def test_score_fn_without_expected_warns():
    r = evaluate_program(
        ProgramCandidate(GOOD_AX),
        mode="predict_rows",
        input_rows=[{}],
        score_fn=lambda p, e: {"x": 1.0},
    )
    assert r.success is True
    assert r.metrics == {}
    assert any("expected_rows" in w for w in r.warnings)


def test_expected_length_mismatch():
    r = evaluate_program(
        ProgramCandidate(GOOD_AX),
        mode="predict_rows",
        input_rows=[{}, {}],
        expected_rows=[{"y": 1.0}],
        score_fn=lambda p, e: {},
    )
    assert r.success is False
    assert any("length" in f.message for f in r.failures)


def test_score_fn_exception_becomes_metric_failure():
    def bad_score(p, e):
        raise RuntimeError("metric boom")

    r = evaluate_program(
        ProgramCandidate(GOOD_AX),
        mode="predict_rows",
        input_rows=[{}],
        expected_rows=[{"y": 0.0}],
        score_fn=bad_score,
    )
    assert r.success is False
    assert any(f.stage == "predict" and f.kind == "metric" for f in r.failures)


def test_no_trace_snippet_when_disabled():
    r = evaluate_program(
        ProgramCandidate(GOOD_AX),
        mode="predict_rows",
        input_rows=[{}],
        include_trace_snippet=False,
    )
    assert r.success is True
    assert r.trace_snippet is None


def test_program_metric_dataclass():
    m = ProgramMetric(name="a", value=1.5)
    assert m.name == "a" and m.value == 1.5


def test_evaluation_report_typing():
    r = ProgramEvaluationReport(
        success=True,
        source="s",
        compile_stage_reached="block",
        mode="compile_only",
    )
    assert r.metrics == {}
    assert r.program_metrics == []
    assert r.row_comparisons is None


def test_predict_rows_row_comparisons_worst_first_and_json():
    """y = x * 1.0 under-predicts doubling task; larger error row sorts first."""
    src = "y = x * 1.0;\n"
    inp = [{"x": 1.0}, {"x": 2.0}]
    exp = [{"y": 2.0}, {"y": 4.0}]
    r = evaluate_program(
        ProgramCandidate(src),
        mode="predict_rows",
        input_rows=inp,
        expected_rows=exp,
        score_fn=default_neg_mse_score_fn(),
        include_trace_snippet=False,
    )
    assert r.success and r.row_comparisons
    assert len(r.row_comparisons) == 2
    assert r.row_comparisons[0]["row_max_abs_error"] >= r.row_comparisons[1]["row_max_abs_error"]
    assert r.row_comparisons[0]["expected"]["y"] == 4.0
    d = evaluation_report_to_dict(r)
    json.dumps(d)
    assert "row_comparisons" in d and len(d["row_comparisons"]) == 2


def test_predict_rows_row_comparison_limit_slices():
    src = "y = x * 1.0;\n"
    inp = [{"x": float(i)} for i in range(5)]
    exp = [{"y": float(i * 2)} for i in range(5)]
    r = evaluate_program(
        ProgramCandidate(src),
        mode="predict_rows",
        input_rows=inp,
        expected_rows=exp,
        score_fn=default_neg_mse_score_fn(),
        include_trace_snippet=False,
        row_comparison_limit=2,
    )
    assert r.success and r.row_comparisons is not None
    assert len(r.row_comparisons) == 2


def test_predict_rows_row_comparison_limit_zero():
    r = evaluate_program(
        ProgramCandidate("y = x * 1.0;\n"),
        mode="predict_rows",
        input_rows=[{"x": 1.0}],
        expected_rows=[{"y": 2.0}],
        score_fn=default_neg_mse_score_fn(),
        include_trace_snippet=False,
        row_comparison_limit=0,
    )
    assert r.success
    assert r.row_comparisons is None
