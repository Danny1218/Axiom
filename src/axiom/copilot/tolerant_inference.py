"""Noise-tolerant symbolic regression (least-squares families before LLM draft)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from axiom.copilot.search import (
    CopilotSearchConfig,
    _affine_multi_input_source,
    _clamped_affine_multi_input_source,
    _linear_xy_coeff_str,
    _quadratic_single_input_source,
    _single_input_affine_source,
    _two_input_interaction_source,
    is_exact_symbolic_examples_task,
)
from axiom.experts.base import ExpertDraftResponse

DEFAULT_RMSE_TOLERANCE = 0.05
MAX_ROW_ABS_ERROR = 0.025
_NONLINEAR_GOAL_HINTS = (
    "clamp",
    "bounded",
    "min(",
    "max(",
    "prefer",
    "cap",
    "clip",
    "piecewise",
    "threshold",
    "mirror",
)
_AFFINE_ONLY_FAMILIES = frozenset(
    {
        "single_input_affine",
        "two_input_interaction",
        "two_input_ab_b",
        "two_input_ab_a",
        "two_input_ab_bias",
        "affine_multi_input",
    }
)


def _round_coeff(v: float) -> float:
    if not math.isfinite(v):
        return v
    for step in (1.0, 0.5, 0.25, 0.125, 0.1):
        snapped = round(v / step) * step
        if abs(v - snapped) <= max(0.03 * max(abs(v), 1.0), 0.005):
            return float(snapped)
    return float(f"{v:.6g}")


def _relative_rmse(pred: Sequence[float], actual: Sequence[float]) -> float:
    p = torch.tensor(list(pred), dtype=torch.float64)
    a = torch.tensor(list(actual), dtype=torch.float64)
    mse = torch.mean((p - a) ** 2).item()
    scale = max(torch.mean(a ** 2).item(), 1e-12)
    return math.sqrt(mse / scale)


def _lstsq(design: torch.Tensor, targets: torch.Tensor) -> Optional[List[float]]:
    if design.shape[0] < 2:
        return None
    try:
        sol = torch.linalg.lstsq(design, targets.unsqueeze(1)).solution.squeeze(1)
    except RuntimeError:
        return None
    if sol.numel() != design.shape[1] or not torch.isfinite(sol).all():
        return None
    return [float(x) for x in sol.tolist()]


@dataclass(frozen=True)
class _RowBatch:
    in_keys: Tuple[str, ...]
    out_key: str
    xs: List[List[float]]
    ys: List[float]


@dataclass(frozen=True)
class _Cand:
    source: str
    backend: str
    family: str
    rmse: float
    complexity: int
    max_abs_error: float


def _load_row_batch(config: CopilotSearchConfig) -> Optional[_RowBatch]:
    if config.mode != "predict_rows" or not config.expected_rows:
        return None
    inp, exp = config.example_input_rows, config.expected_rows
    if not inp or not exp or len(inp) != len(exp) or len(inp) < 3:
        return None
    in_keys: Optional[Tuple[str, ...]] = None
    out_key: Optional[str] = None
    xs: List[List[float]] = []
    ys: List[float] = []
    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping) or len(row_ex) != 1:
            return None
        keys = tuple(sorted(str(k) for k in row_in.keys()))
        if in_keys is None:
            in_keys = keys
        elif keys != in_keys:
            return None
        ok = str(next(iter(row_ex.keys())))
        if out_key is None:
            out_key = ok
        elif ok != out_key:
            return None
        try:
            xs.append([float(row_in[k]) for k in in_keys])
            ys.append(float(row_ex[ok]))
        except (TypeError, ValueError, KeyError):
            return None
        if not all(math.isfinite(v) for v in xs[-1]) or not math.isfinite(ys[-1]):
            return None
    assert in_keys is not None and out_key is not None
    return _RowBatch(in_keys, out_key, xs, ys)


def _abs_bias_source(in_key: str, out_key: str, bias: float) -> str:
    b = _linear_xy_coeff_str(bias)
    return (
        f"if ({in_key} < 0.0) {{\n"
        f"    {out_key} = -{in_key} + {b};\n"
        "} else {\n"
        f"    {out_key} = {in_key} + {b};\n"
        "}\n"
    )


def _max_abs_error(pred: Sequence[float], actual: Sequence[float]) -> float:
    return max(abs(p - a) for p, a in zip(pred, actual))


def _pick(candidates: List[_Cand], tol: float) -> Optional[_Cand]:
    ok = [
        c
        for c in candidates
        if c.rmse <= tol and c.max_abs_error <= MAX_ROW_ABS_ERROR
    ]
    if not ok:
        return None
    ok.sort(key=lambda c: (c.rmse, c.complexity))
    return ok[0]


def _append_two_input_candidate(
    cands: List[_Cand],
    *,
    out_key: str,
    k1: str,
    k2: str,
    w_ab: float,
    w_a: float,
    w_b: float,
    bias: float,
    pred: Sequence[float],
    ys: Sequence[float],
    backend: str,
    family: str,
    complexity: int,
) -> None:
    cands.append(
        _Cand(
            _two_input_interaction_source(out_key, k1, k2, w_ab, w_a, w_b, bias),
            backend,
            family,
            _relative_rmse(pred, ys),
            complexity,
            _max_abs_error(pred, ys),
        )
    )


def _goal_hints_nonlinear_structure(config: CopilotSearchConfig) -> bool:
    text = f"{config.goal} {config.domain_context or ''}".lower()
    return any(hint in text for hint in _NONLINEAR_GOAL_HINTS)


def try_tolerant_symbolic_inference(
    config: CopilotSearchConfig,
    *,
    rmse_tolerance: float = DEFAULT_RMSE_TOLERANCE,
) -> Optional[ExpertDraftResponse]:
    if not is_exact_symbolic_examples_task(config):
        return None
    batch = _load_row_batch(config)
    if batch is None:
        return None

    cands: List[_Cand] = []
    y_t = torch.tensor(batch.ys, dtype=torch.float64)
    n_in = len(batch.in_keys)

    if n_in == 1:
        x = [r[0] for r in batch.xs]
        x_t = torch.tensor(x, dtype=torch.float64)
        coef = _lstsq(torch.stack([x_t, torch.ones_like(x_t)], dim=1), y_t)
        if coef:
            slope, bias = map(_round_coeff, coef)
            pred = [slope * xi + bias for xi in x]
            cands.append(
                _Cand(
                    _single_input_affine_source(batch.in_keys[0], batch.out_key, slope, bias),
                    "tolerant_single_input_affine",
                    "single_input_affine",
                    _relative_rmse(pred, batch.ys),
                    2,
                    _max_abs_error(pred, batch.ys),
                )
            )
        coef_q = _lstsq(torch.stack([x_t * x_t, x_t, torch.ones_like(x_t)], dim=1), y_t)
        if coef_q:
            q, lin, bias = map(_round_coeff, coef_q)
            pred = [q * xi * xi + lin * xi + bias for xi in x]
            cands.append(
                _Cand(
                    _quadratic_single_input_source(batch.in_keys[0], batch.out_key, q, lin, bias),
                    "tolerant_quadratic",
                    "quadratic_single_input",
                    _relative_rmse(pred, batch.ys),
                    3,
                    _max_abs_error(pred, batch.ys),
                )
            )
        abs_t = torch.tensor([abs(v) for v in x], dtype=torch.float64)
        coef_a = _lstsq(torch.stack([abs_t, torch.ones_like(abs_t)], dim=1), y_t)
        if coef_a and math.isclose(coef_a[0], 1.0, rel_tol=0.0, abs_tol=0.08):
            bias = _round_coeff(coef_a[1])
            pred = [abs(v) + bias for v in x]
            cands.append(
                _Cand(
                    _abs_bias_source(batch.in_keys[0], batch.out_key, bias),
                    "tolerant_abs_bias",
                    "absolute_with_bias",
                    _relative_rmse(pred, batch.ys),
                    2,
                    _max_abs_error(pred, batch.ys),
                )
            )

    if n_in == 2:
        a_vals = [r[0] for r in batch.xs]
        b_vals = [r[1] for r in batch.xs]
        k1, k2 = batch.in_keys[0], batch.in_keys[1]
        sparse_variants: List[Tuple[List[int], str, str, int]] = [
            ([0, 1, 2, 3], "tolerant_two_input_interaction", "two_input_interaction", 4),
            ([0, 2, 3], "tolerant_two_input_ab_b", "two_input_ab_b", 3),
            ([0, 1, 3], "tolerant_two_input_ab_a", "two_input_ab_a", 3),
            ([0, 3], "tolerant_two_input_ab_bias", "two_input_ab_bias", 2),
        ]
        full_design = torch.tensor(
            [[ai * bi, ai, bi, 1.0] for ai, bi in zip(a_vals, b_vals)],
            dtype=torch.float64,
        )
        for cols, backend, family, complexity in sparse_variants:
            design = full_design[:, cols]
            coef = _lstsq(design, y_t)
            if not coef:
                continue
            full_coef = [0.0, 0.0, 0.0, 0.0]
            for col_idx, val in zip(cols, coef):
                full_coef[col_idx] = val
            w_ab, w_a, w_b, bias = map(_round_coeff, full_coef)
            pred = [
                w_ab * ai * bi + w_a * ai + w_b * bi + bias for ai, bi in zip(a_vals, b_vals)
            ]
            _append_two_input_candidate(
                cands,
                out_key=batch.out_key,
                k1=k1,
                k2=k2,
                w_ab=w_ab,
                w_a=w_a,
                w_b=w_b,
                bias=bias,
                pred=pred,
                ys=batch.ys,
                backend=backend,
                family=family,
                complexity=complexity,
            )

    if n_in >= 3:
        mat = torch.tensor(batch.xs, dtype=torch.float64)
        coef = _lstsq(torch.cat([mat, torch.ones((len(batch.ys), 1))], dim=1), y_t)
        if coef:
            weights = [_round_coeff(v) for v in coef[:-1]]
            bias = _round_coeff(coef[-1])
            pred = [sum(w * xi for w, xi in zip(weights, row)) + bias for row in batch.xs]
            rmse = _relative_rmse(pred, batch.ys)
            cands.append(
                _Cand(
                    _affine_multi_input_source(batch.out_key, batch.in_keys, weights, bias),
                    "tolerant_affine_multi_input",
                    "affine_multi_input",
                    rmse,
                    len(weights) + 1,
                    _max_abs_error(pred, batch.ys),
                )
            )
            clamped = [max(0.0, min(1.0, p)) for p in pred]
            cands.append(
                _Cand(
                    _clamped_affine_multi_input_source(batch.out_key, batch.in_keys, weights, bias),
                    "tolerant_clamped_affine_multi_input",
                    "clamped_affine_multi_input",
                    _relative_rmse(clamped, batch.ys),
                    len(weights) + 2,
                    _max_abs_error(clamped, batch.ys),
                )
            )

    pick = _pick(cands, rmse_tolerance)
    if pick is None:
        return None
    if _goal_hints_nonlinear_structure(config) and pick.family in _AFFINE_ONLY_FAMILIES:
        return None
    return ExpertDraftResponse(
        ax_source=pick.source,
        backend_name=pick.backend,
        metadata={
            "inference_kind": "tolerant",
            "fast_path": pick.family,
            "relative_rmse": pick.rmse,
        },
    )


__all__ = ["DEFAULT_RMSE_TOLERANCE", "try_tolerant_symbolic_inference"]
