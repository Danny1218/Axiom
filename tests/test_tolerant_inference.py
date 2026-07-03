"""Tolerant symbolic inference — noisy fits, rejection, and robustness tasks."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from axiom.copilot.search import CopilotSearchConfig, run_copilot_draft
from axiom.copilot.tolerant_inference import (
    DEFAULT_RMSE_TOLERANCE,
    try_tolerant_symbolic_inference,
)
from axiom.experts.base import ExpertDraftRequest, SemanticExpert


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


class _FailExpert(SemanticExpert):
    def draft_program(self, request: ExpertDraftRequest):
        raise AssertionError("LLM should not be called")

    def repair_program(self, request):
        raise AssertionError("repair should not run")

    def summarize_trace(self, request):
        return ""


def _config(
    inp: list[dict],
    exp: list[dict],
    *,
    goal: str = "compute symbolic mapping",
    tol: float = DEFAULT_RMSE_TOLERANCE,
) -> CopilotSearchConfig:
    return CopilotSearchConfig(
        goal=goal,
        expert=_FailExpert(),
        mode="predict_rows",
        example_input_rows=inp,
        expected_rows=exp,
        max_iterations=1,
    )


def test_tolerant_recovers_noisy_affine():
    rng = random.Random(42)
    inp = [{"x": float(v)} for v in (-1.0, 0.0, 1.0, 2.0)]
    exp = [{"y": 1.25 * row["x"] - 0.2 + rng.uniform(-0.02, 0.02)} for row in inp]
    resp = try_tolerant_symbolic_inference(_config(inp, exp))
    assert resp is not None
    assert resp.metadata.get("inference_kind") == "tolerant"
    assert "1.25" in resp.ax_source or "1.2" in resp.ax_source
    assert resp.metadata["relative_rmse"] <= DEFAULT_RMSE_TOLERANCE


def test_tolerant_scale_relative_row_gate():
    """Large-magnitude targets allow proportionally larger per-row error."""
    inp = [{"x": float(v)} for v in (100.0, 200.0, 300.0, 400.0)]
    exp = [{"y": 2.0 * row["x"] + rng_offset} for row, rng_offset in zip(inp, (1.0, -2.0, 3.0, -4.0))]
    resp = try_tolerant_symbolic_inference(_config(inp, exp))
    assert resp is not None
    assert resp.metadata["relative_rmse"] <= DEFAULT_RMSE_TOLERANCE


def test_tolerant_rejects_pure_noise():
    rng = random.Random(0)
    inp = [{"x": float(i)} for i in range(6)]
    exp = [{"y": rng.uniform(-100.0, 100.0)} for _ in inp]
    assert try_tolerant_symbolic_inference(_config(inp, exp)) is None


def test_tolerant_prefers_simpler_affine_over_quadratic():
    inp = [{"x": float(v)} for v in (0.0, 1.0, 2.0, 3.0)]
    exp = [{"y": 2.0 * row["x"] + 1.0} for row in inp]
    resp = try_tolerant_symbolic_inference(_config(inp, exp))
    assert resp is not None
    assert resp.metadata["fast_path"] == "single_input_affine"
    assert "x * x" not in resp.ax_source


@pytest.mark.parametrize(
    "task_id,needle",
    [
        ("noisy_affine_thermometer", "thermometer_reading"),
        ("signed_cross_term_noisy", "exposure * hedge"),
        ("near_abs_with_bias", "delta +"),
    ],
)
def test_robustness_tasks_solved_without_llm(task_id: str, needle: str):
    raw = json.loads(
        (_root() / "benchmarks" / "copilot_symbolic_robustness_ambiguity_stress_tasks.json").read_text(
            encoding="utf-8"
        )
    )
    task = next(t for t in raw["tasks"] if t["id"] == task_id)
    cfg = CopilotSearchConfig(
        goal=task["goal"],
        domain_context=task.get("domain_context"),
        expert=_FailExpert(),
        mode="predict_rows",
        example_input_rows=task["example_input_rows"],
        expected_rows=task["expected_rows"],
        max_iterations=1,
    )
    _, resp = run_copilot_draft(cfg)
    assert resp.metadata.get("inference_kind") == "tolerant"
    assert needle in resp.ax_source
