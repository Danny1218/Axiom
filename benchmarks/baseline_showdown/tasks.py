"""Synthetic ground-truth tasks for symbolic extrapolation benchmark."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from benchmarks.baseline_showdown.harness import (
    EXTRAP_N,
    GLOBAL_SEED,
    INTERP_N,
    NOISE_FRAC,
    TRAIN_N,
    TaskSplit,
    coeff_within_pct,
)


def _task_seed(task_id: str, base: int = GLOBAL_SEED) -> int:
    return base + sum(ord(c) for c in task_id)


@dataclass(frozen=True)
class SynthTask:
    task_id: str
    family: str
    in_family: bool
    input_names: Tuple[str, ...]
    output_name: str
    goal: str
    ground_truth: str
    formula: Callable[[Dict[str, float]], float]
    expected_coeffs: Dict[str, float]

    def generate_split(self, seed: int = GLOBAL_SEED) -> TaskSplit:
        rng = random.Random(_task_seed(self.task_id))
        train_range = (-1.0, 1.0) if self.family == "clamped_affine_multi_input" else (-2.0, 2.0)
        train_rows, train_y = _sample_rows(
            rng, self.input_names, self.formula, TRAIN_N, train_range, noisy=True
        )
        interp_rows, interp_y = _sample_rows(
            rng, self.input_names, self.formula, INTERP_N, (-2.0, 2.0), noisy=False
        )
        extrap_lo, extrap_y_lo = _sample_rows(
            rng, self.input_names, self.formula, EXTRAP_N // 2, (-6.0, -2.0), noisy=False
        )
        extrap_hi, extrap_y_hi = _sample_rows(
            rng, self.input_names, self.formula, EXTRAP_N // 2, (2.0, 6.0), noisy=False
        )
        return TaskSplit(
            train_rows=train_rows,
            train_y=train_y,
            interp_rows=interp_rows,
            interp_y=interp_y,
            extrap_rows=extrap_lo + extrap_hi,
            extrap_y=extrap_y_lo + extrap_y_hi,
        )

    def training_examples(self, split: TaskSplit) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        inp = split.train_rows
        exp = [{self.output_name: y} for y in split.train_y]
        return inp, exp

    def coeffs_match(self, ax_source: str) -> bool:
        """Heuristic: recovered literals within 5% of ground-truth coefficients."""
        for key, expected in self.expected_coeffs.items():
            token = _format_coeff(expected)
            if token not in ax_source and not _any_near_literal(ax_source, expected):
                return False
        return True


def _format_coeff(v: float) -> str:
    r = round(v)
    if abs(v - r) < 1e-9:
        return f"{float(r):.1f}"
    s = format(v, ".6g")
    return s.rstrip("0").rstrip(".") if "." in s else s


def _any_near_literal(source: str, expected: float) -> bool:
    import re

    for m in re.finditer(r"-?\d+\.?\d*", source):
        try:
            val = float(m.group())
        except ValueError:
            continue
        if coeff_within_pct(expected, val):
            return True
    return False


def _sample_rows(
    rng: random.Random,
    keys: Sequence[str],
    formula: Callable[[Dict[str, float]], float],
    n: int,
    x_range: Tuple[float, float],
    *,
    noisy: bool,
) -> Tuple[List[Dict[str, float]], List[float]]:
    lo, hi = x_range
    rows: List[Dict[str, float]] = []
    ys: List[float] = []
    for _ in range(n):
        row = {k: rng.uniform(lo, hi) for k in keys}
        y = formula(row)
        if noisy:
            clean = y
            sigma = NOISE_FRAC * max(abs(clean), 1.0)
            y = clean + rng.gauss(0.0, sigma)
        rows.append(row)
        ys.append(y)
    return rows, ys


def all_tasks() -> List[SynthTask]:
    return [
        SynthTask(
            task_id="affine_slope_2",
            family="single_input_affine",
            in_family=True,
            input_names=("x",),
            output_name="y",
            goal="Compute y = 2.0 * x + 1.5 from noisy examples.",
            ground_truth="y = 2.0*x + 1.5",
            formula=lambda r: 2.0 * r["x"] + 1.5,
            expected_coeffs={"slope": 2.0, "bias": 1.5},
        ),
        SynthTask(
            task_id="affine_neg_slope",
            family="single_input_affine",
            in_family=True,
            input_names=("x",),
            output_name="y",
            goal="Compute y = -1.25 * x + 0.5 from noisy examples.",
            ground_truth="y = -1.25*x + 0.5",
            formula=lambda r: -1.25 * r["x"] + 0.5,
            expected_coeffs={"slope": -1.25, "bias": 0.5},
        ),
        SynthTask(
            task_id="quadratic_standard",
            family="quadratic_single_input",
            in_family=True,
            input_names=("x",),
            output_name="y",
            goal="Compute quadratic y = 0.3 * x * x - 0.5 * x + 1.0 from noisy examples.",
            ground_truth="y = 0.3*x^2 - 0.5*x + 1.0",
            formula=lambda r: 0.3 * r["x"] ** 2 - 0.5 * r["x"] + 1.0,
            expected_coeffs={"quad": 0.3, "lin": -0.5, "bias": 1.0},
        ),
        SynthTask(
            task_id="quadratic_inverted",
            family="quadratic_single_input",
            in_family=True,
            input_names=("x",),
            output_name="y",
            goal="Compute quadratic y = -0.2 * x * x + x + 0.25 from noisy examples.",
            ground_truth="y = -0.2*x^2 + x + 0.25",
            formula=lambda r: -0.2 * r["x"] ** 2 + r["x"] + 0.25,
            expected_coeffs={"quad": -0.2, "lin": 1.0, "bias": 0.25},
        ),
        SynthTask(
            task_id="interaction_full",
            family="two_input_interaction",
            in_family=True,
            input_names=("a", "b"),
            output_name="out",
            goal="Compute out = 0.5 * a * b + 0.3 * a + 0.2 * b + 1.0 from noisy examples.",
            ground_truth="out = 0.5*a*b + 0.3*a + 0.2*b + 1.0",
            formula=lambda r: 0.5 * r["a"] * r["b"] + 0.3 * r["a"] + 0.2 * r["b"] + 1.0,
            expected_coeffs={"w_ab": 0.5, "w_a": 0.3, "w_b": 0.2, "bias": 1.0},
        ),
        SynthTask(
            task_id="interaction_ab_only",
            family="two_input_ab_b",
            in_family=True,
            input_names=("a", "b"),
            output_name="out",
            goal="Compute out = 2.0 * a * b + 0.5 from noisy cross-term examples.",
            ground_truth="out = 2.0*a*b + 0.5",
            formula=lambda r: 2.0 * r["a"] * r["b"] + 0.5,
            expected_coeffs={"w_ab": 2.0, "bias": 0.5},
        ),
        SynthTask(
            task_id="abs_with_bias",
            family="absolute_with_bias",
            in_family=True,
            input_names=("x",),
            output_name="y",
            goal="Compute y = abs(x) + 0.75 from noisy examples (mirror / absolute value).",
            ground_truth="y = |x| + 0.75",
            formula=lambda r: abs(r["x"]) + 0.75,
            expected_coeffs={"bias": 0.75},
        ),
        SynthTask(
            task_id="affine_three_input",
            family="affine_multi_input",
            in_family=True,
            input_names=("u", "v", "w"),
            output_name="score",
            goal="Compute score = 0.4 * u + 0.35 * v + 0.25 * w + 0.5 weighted sum from examples.",
            ground_truth="score = 0.4*u + 0.35*v + 0.25*w + 0.5",
            formula=lambda r: 0.4 * r["u"] + 0.35 * r["v"] + 0.25 * r["w"] + 0.5,
            expected_coeffs={"w_u": 0.4, "w_v": 0.35, "w_w": 0.25, "bias": 0.5},
        ),
        SynthTask(
            task_id="clamped_affine_three",
            family="clamped_affine_multi_input",
            in_family=True,
            input_names=("x1", "x2", "x3"),
            output_name="out",
            goal="Compute out = max(0.0, min(1.0, 0.6 * x1 + 0.3 * x2 + 0.1 * x3 + 0.05)) clamped unit interval.",
            ground_truth="out = clamp(0.6*x1 + 0.3*x2 + 0.1*x3 + 0.05, 0, 1)",
            formula=lambda r: max(
                0.0, min(1.0, 0.6 * r["x1"] + 0.3 * r["x2"] + 0.1 * r["x3"] + 0.05)
            ),
            expected_coeffs={"w1": 0.6, "w2": 0.3, "w3": 0.1, "bias": 0.05},
        ),
        SynthTask(
            task_id="interaction_mixed",
            family="two_input_interaction",
            in_family=True,
            input_names=("a", "b"),
            output_name="out",
            goal="Compute out = -0.4 * a * b + 1.2 * a - 0.3 * b + 2.0 from noisy examples.",
            ground_truth="out = -0.4*a*b + 1.2*a - 0.3*b + 2.0",
            formula=lambda r: -0.4 * r["a"] * r["b"] + 1.2 * r["a"] - 0.3 * r["b"] + 2.0,
            expected_coeffs={"w_ab": -0.4, "w_a": 1.2, "w_b": -0.3, "bias": 2.0},
        ),
        SynthTask(
            task_id="sabotage_sin",
            family="sinusoid",
            in_family=False,
            input_names=("x",),
            output_name="y",
            goal="Discover symbolic formula y = sin(x) from noisy examples.",
            ground_truth="y = sin(x)",
            formula=lambda r: math.sin(r["x"]),
            expected_coeffs={},
        ),
        SynthTask(
            task_id="sabotage_exp_decay",
            family="exponential_decay",
            in_family=False,
            input_names=("x",),
            output_name="y",
            goal="Discover symbolic formula y = exp(-x) exponential decay from noisy examples.",
            ground_truth="y = exp(-x)",
            formula=lambda r: math.exp(-r["x"]),
            expected_coeffs={},
        ),
    ]
