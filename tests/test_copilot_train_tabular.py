"""Copilot ``evaluate_program(..., mode=\"train_tabular\")`` — in-memory Adam on ``InterpretedBlock``."""

from __future__ import annotations

import pytest

from axiom.compiler.parser import reset_parser
from axiom.copilot import (
    ProgramCandidate,
    TrainTabularParams,
    evaluate_program,
)


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


NEURAL_AX = "y = neural([x]);"
LINEAR_AX = "y = x * 2.0;"


def _synth_regression(n: int, *, seed: float = 0.0) -> list[dict]:
    rows = []
    for i in range(n):
        x = float(i) * 0.3 + seed
        rows.append({"x": x, "y": 2.0 * x + 0.1})
    return rows


def test_train_tabular_learns_regression_mse_drops():
    train = _synth_regression(24, seed=0.02)
    eval_rows = _synth_regression(8, seed=1.0)
    r0 = evaluate_program(
        ProgramCandidate(NEURAL_AX),
        mode="train_tabular",
        target_var="y",
        train_rows=train,
        eval_rows=eval_rows,
        train_tabular_params=TrainTabularParams(epochs=1, learning_rate=0.0, batch_size=8),
        include_trace_snippet=False,
    )
    r1 = evaluate_program(
        ProgramCandidate(NEURAL_AX),
        mode="train_tabular",
        target_var="y",
        train_rows=train,
        eval_rows=eval_rows,
        train_tabular_params=TrainTabularParams(epochs=120, learning_rate=0.08, batch_size=8),
        include_trace_snippet=False,
    )
    assert r0.success and r1.success
    assert r0.metrics["eval_mse"] > r1.metrics["eval_mse"]


def test_train_tabular_success_metrics_trace_score_fn():
    train = _synth_regression(16)
    eval_rows = _synth_regression(4, seed=0.5)
    exp = [{"y": r["y"]} for r in eval_rows]

    def score_fn(preds, exp_):
        return {"neg_mse": -sum((float(p["y"]) - float(e["y"])) ** 2 for p, e in zip(preds, exp_))}

    r = evaluate_program(
        ProgramCandidate(NEURAL_AX),
        mode="train_tabular",
        target_var="y",
        train_rows=train,
        eval_rows=eval_rows,
        expected_rows=exp,
        score_fn=score_fn,
        train_tabular_params=TrainTabularParams(epochs=80, learning_rate=0.06, batch_size=8),
        predictions_sample_limit=2,
        include_trace_snippet=True,
    )
    assert r.success
    assert r.compile_stage_reached == "train"
    assert "train_mse" in r.metrics and "eval_mse" in r.metrics
    assert "neg_mse" in r.metrics
    assert r.predictions_sample is not None and len(r.predictions_sample) == 2
    assert r.trace_snippet is not None and isinstance(r.trace_snippet, dict)


def test_train_tabular_no_trainable_params_warning_and_exact_symbolic():
    train = [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}]
    eval_rows = [{"x": 3.0, "y": 6.0}]
    r = evaluate_program(
        ProgramCandidate(LINEAR_AX),
        mode="train_tabular",
        target_var="y",
        train_rows=train,
        eval_rows=eval_rows,
        include_trace_snippet=False,
    )
    assert r.success
    assert any("no_trainable_parameters" in w for w in r.warnings)
    assert r.metrics["eval_mse"] < 1e-5


def test_train_tabular_missing_target_var():
    r = evaluate_program(
        ProgramCandidate(NEURAL_AX),
        mode="train_tabular",
        train_rows=[{"x": 1.0, "y": 1.0}],
        eval_rows=[{"x": 2.0, "y": 2.0}],
    )
    assert not r.success
    assert any(f.stage == "train" and "target_var" in f.message.lower() for f in r.failures)


def test_train_tabular_empty_train():
    r = evaluate_program(
        ProgramCandidate(NEURAL_AX),
        mode="train_tabular",
        target_var="y",
        train_rows=[],
        eval_rows=[{"x": 1.0, "y": 1.0}],
    )
    assert not r.success


def test_train_tabular_empty_eval():
    r = evaluate_program(
        ProgramCandidate(NEURAL_AX),
        mode="train_tabular",
        target_var="y",
        train_rows=[{"x": 1.0, "y": 1.0}],
        eval_rows=[],
    )
    assert not r.success


def test_train_tabular_target_not_in_abi():
    r = evaluate_program(
        ProgramCandidate(NEURAL_AX),
        mode="train_tabular",
        target_var="z_missing",
        train_rows=[{"x": 1.0, "z_missing": 1.0}],
        eval_rows=[{"x": 2.0, "z_missing": 2.0}],
    )
    assert not r.success
    assert any("ABI" in f.message for f in r.failures)


def test_train_tabular_row_missing_target_key():
    r = evaluate_program(
        ProgramCandidate(NEURAL_AX),
        mode="train_tabular",
        target_var="y",
        train_rows=[{"x": 1.0}],
        eval_rows=[{"x": 2.0, "y": 2.0}],
    )
    assert not r.success


def test_train_tabular_non_numeric_schema():
    r = evaluate_program(
        ProgramCandidate(NEURAL_AX),
        mode="train_tabular",
        target_var="y",
        train_rows=[{"x": "not_a_number", "y": 1.0}],
        eval_rows=[{"x": 0.0, "y": 0.0}],
    )
    assert not r.success
    assert any(f.kind == "schema" for f in r.failures)


def test_train_tabular_score_fn_length_mismatch():
    r = evaluate_program(
        ProgramCandidate(NEURAL_AX),
        mode="train_tabular",
        target_var="y",
        train_rows=_synth_regression(8),
        eval_rows=_synth_regression(3),
        expected_rows=[{"y": 1.0}],
        score_fn=lambda p, e: {"x": 1.0},
    )
    assert not r.success
    assert any("length" in f.message for f in r.failures)


def test_train_tabular_score_fn_exception():
    train = _synth_regression(8)
    eval_rows = _synth_regression(3)
    exp = [{"y": r["y"]} for r in eval_rows]

    def bad_score(p, e):
        raise RuntimeError("boom")

    r = evaluate_program(
        ProgramCandidate(NEURAL_AX),
        mode="train_tabular",
        target_var="y",
        train_rows=train,
        eval_rows=eval_rows,
        expected_rows=exp,
        score_fn=bad_score,
        train_tabular_params=TrainTabularParams(epochs=5, batch_size=4),
        include_trace_snippet=False,
    )
    assert not r.success
    assert any(f.stage == "train" and f.kind == "metric" for f in r.failures)


def test_train_tabular_score_fn_without_expected_warns():
    r = evaluate_program(
        ProgramCandidate(NEURAL_AX),
        mode="train_tabular",
        target_var="y",
        train_rows=_synth_regression(8),
        eval_rows=_synth_regression(3),
        score_fn=lambda p, e: {"custom": 1.0},
        include_trace_snippet=False,
    )
    assert r.success
    assert any("expected_rows" in w for w in r.warnings)
    assert "custom" not in r.metrics
