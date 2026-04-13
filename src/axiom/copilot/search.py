"""Draft → evaluate → repair loop over ``.ax`` programs (expert backend is injectable; no network here)."""

from __future__ import annotations

import json
import itertools
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
    r"risk_score|linear|weighted|blend|double\b|mapping|polynomial|abs\b|absolute)",
    re.I,
)
_GOAL_EXACT_SYMBOLIC_EXTRA = re.compile(
    r"(max\s*\(|min\s*\(|clamp|affine|weighted\s+(sum|blend)|risk_score)",
    re.I,
)
_GOAL_CLAMP_HINT = re.compile(r"(max\s*\(|min\s*\(|clamp|bounded|risk_score)", re.I)
_GOAL_EXACT_PIECEWISE_CONTROL = re.compile(r"\bif\b.+(?:<|>).+\bthen\b.+\belse\b", re.I)

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


def _goal_suggests_piecewise_control(goal: str) -> bool:
    g = (goal or "").strip()
    if not g or len(g) > 500:
        return False
    return bool(_GOAL_EXACT_PIECEWISE_CONTROL.search(g))


def _exact_symbolic_hint_text(config: CopilotSearchConfig) -> str:
    parts = [(config.goal or "").strip()]
    domain_context = (config.domain_context or "").strip()
    if domain_context:
        parts.append(domain_context)
    return " ".join(p for p in parts if p)


def _hint_suggests_clamp_family(config: CopilotSearchConfig) -> bool:
    return bool(_GOAL_CLAMP_HINT.search(_exact_symbolic_hint_text(config)))


def _has_single_numeric_input_output_rows(config: CopilotSearchConfig) -> bool:
    inp = config.example_input_rows
    exp = config.expected_rows
    if not inp or not exp or len(inp) != len(exp):
        return False
    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping):
            return False
        if len(row_in) != 1 or len(row_ex) != 1:
            return False
        try:
            x = float(next(iter(row_in.values())))
            y = float(next(iter(row_ex.values())))
        except (TypeError, ValueError, StopIteration):
            return False
        if not math.isfinite(x) or not math.isfinite(y):
            return False
    return True


def is_exact_symbolic_examples_task(config: CopilotSearchConfig) -> bool:
    """predict_rows + expected rows + goal/context look like an exact symbolic arithmetic or piecewise task."""
    if config.mode != "predict_rows" or not config.expected_rows:
        return False
    hint_text = _exact_symbolic_hint_text(config)
    if _goal_suggests_symbolic_math(hint_text):
        return True
    if _goal_suggests_piecewise_control(hint_text) and _has_single_numeric_input_output_rows(config):
        return True
    if len(hint_text) <= 800 and _GOAL_EXACT_SYMBOLIC_EXTRA.search(hint_text):
        return True
    if len(hint_text) <= 800 and _goal_suggests_cross_term_with_additive(hint_text):
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


def _quadratic_single_input_source(in_key: str, out_key: str, bias: float) -> str:
    expr = f"{in_key} * {in_key}"
    if math.isclose(bias, 0.0, abs_tol=1e-12, rel_tol=1e-12):
        return f"{out_key} = {expr};\n"
    if bias < 0.0:
        return f"{out_key} = {expr} - {_linear_xy_coeff_str(abs(bias))};\n"
    return f"{out_key} = {expr} + {_linear_xy_coeff_str(bias)};\n"


def _piecewise_threshold_identity_source(in_key: str, out_key: str) -> str:
    return (
        f"if ({in_key} < 0.0) {{\n"
        f"    {out_key} = 0.0;\n"
        "} else {\n"
        f"    {out_key} = {in_key};\n"
        "}\n"
    )


def _absolute_value_piecewise_source(in_key: str, out_key: str) -> str:
    return (
        f"if ({in_key} < 0.0) {{\n"
        f"    {out_key} = -{in_key};\n"
        "} else {\n"
        f"    {out_key} = {in_key};\n"
        "}\n"
    )


def _single_input_affine_expr(in_key: str, slope: float, bias: float) -> str:
    parts: List[str] = []

    def _append_signed(expr: str, negative: bool) -> None:
        if not parts:
            parts.append(f"-{expr}" if negative else expr)
        else:
            parts.append(f"- {expr}" if negative else f"+ {expr}")

    if not math.isclose(slope, 0.0, abs_tol=1e-12, rel_tol=1e-12):
        if math.isclose(abs(slope), 1.0, abs_tol=1e-12, rel_tol=1e-12):
            term = in_key
        else:
            term = f"{_linear_xy_coeff_str(abs(slope))} * {in_key}"
        _append_signed(term, slope < 0.0)
    if not math.isclose(bias, 0.0, abs_tol=1e-12, rel_tol=1e-12):
        _append_signed(_linear_xy_coeff_str(abs(bias)), bias < 0.0)
    return " ".join(parts) if parts else "0.0"


def _nested_piecewise_affine_source(
    in_key: str,
    out_key: str,
    low_threshold: float,
    high_threshold: float,
    low_value: float,
    mid_slope: float,
    mid_bias: float,
    high_value: float,
) -> str:
    return (
        f"if ({in_key} < {_linear_xy_coeff_str(low_threshold)}) {{\n"
        f"    {out_key} = {_linear_xy_coeff_str(low_value)};\n"
        "} else {\n"
        f"    if ({in_key} < {_linear_xy_coeff_str(high_threshold)}) {{\n"
        f"        {out_key} = {_single_input_affine_expr(in_key, mid_slope, mid_bias)};\n"
        "    } else {\n"
        f"        {out_key} = {_linear_xy_coeff_str(high_value)};\n"
        "    }\n"
        "}\n"
    )


def _max_of_two_source(k1: str, k2: str, out_var: str) -> str:
    return (
        f"if ({k1} > {k2}) {{\n"
        f"    {out_var} = {k1};\n"
        "} else {\n"
        f"    {out_var} = {k2};\n"
        "}\n"
    )


def _try_max_of_two_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    """Exact two-input max family: ``out = max(a, b)`` with evidence for both strict branches."""
    if not is_exact_symbolic_examples_task(config):
        return None
    if config.mode != "predict_rows":
        return None
    inp = config.example_input_rows
    exp = config.expected_rows
    if not inp or not exp or len(inp) != len(exp):
        return None
    if len(inp) < 2:
        return None

    in_keys: Optional[tuple[str, str]] = None
    out_key: Optional[str] = None
    saw_left = False
    saw_right = False

    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping):
            return None
        if len(row_in) != 2 or len(row_ex) != 1:
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
            a = float(row_in[in_keys[0]])
            b = float(row_in[in_keys[1]])
            y = float(row_ex[out_key])
        except (TypeError, ValueError, KeyError):
            return None
        if not math.isfinite(a) or not math.isfinite(b) or not math.isfinite(y):
            return None
        pred = max(a, b)
        if not math.isclose(pred, y, rel_tol=1e-12, abs_tol=1e-9):
            return None
        if a > b and math.isclose(y, a, rel_tol=1e-12, abs_tol=1e-9):
            saw_left = True
        elif b > a and math.isclose(y, b, rel_tol=1e-12, abs_tol=1e-9):
            saw_right = True

    if not saw_left or not saw_right:
        return None

    assert in_keys is not None and out_key is not None
    return ExpertDraftResponse(
        ax_source=_max_of_two_source(in_keys[0], in_keys[1], out_key),
        backend_name="max_of_two_fast_path",
        metadata={"fast_path": "max_of_two", "in_keys": [in_keys[0], in_keys[1]], "out_key": out_key},
    )


def _try_absolute_value_piecewise_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    """Exact single-input absolute value family: ``if x < 0 then -x else x``."""
    if not is_exact_symbolic_examples_task(config):
        return None
    if config.mode != "predict_rows":
        return None
    inp = config.example_input_rows
    exp = config.expected_rows
    if not inp or not exp or len(inp) != len(exp):
        return None

    in_key: Optional[str] = None
    out_key: Optional[str] = None
    saw_neg = False
    saw_pos = False

    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping):
            return None
        if len(row_in) != 1 or len(row_ex) != 1:
            return None
        ik = str(next(iter(row_in.keys())))
        ok = str(next(iter(row_ex.keys())))
        if in_key is None:
            in_key = ik
        elif ik != in_key:
            return None
        if out_key is None:
            out_key = ok
        elif ok != out_key:
            return None
        try:
            x = float(row_in[in_key])
            y = float(row_ex[out_key])
        except (TypeError, ValueError, KeyError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        pred = -x if x < 0.0 else x
        if not math.isclose(pred, y, rel_tol=1e-12, abs_tol=1e-9):
            return None
        if x < 0.0:
            saw_neg = True
        elif x > 0.0:
            saw_pos = True

    if not saw_neg or not saw_pos:
        return None
    assert in_key is not None and out_key is not None
    return ExpertDraftResponse(
        ax_source=_absolute_value_piecewise_source(in_key, out_key),
        backend_name="absolute_value_piecewise_fast_path",
        metadata={"fast_path": "absolute_value_piecewise", "in_key": in_key, "out_key": out_key},
    )


def _fit_single_input_affine_rows(rows: Sequence[tuple[float, float]]) -> Optional[tuple[float, float]]:
    if len(rows) < 2:
        return None
    slope: Optional[float] = None
    bias: Optional[float] = None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            x0, y0 = rows[i]
            x1, y1 = rows[j]
            if math.isclose(x0, x1, rel_tol=0.0, abs_tol=1e-12):
                continue
            slope = (y1 - y0) / (x1 - x0)
            bias = y0 - slope * x0
            break
        if slope is not None:
            break
    if slope is None or bias is None or math.isclose(slope, 0.0, abs_tol=1e-12, rel_tol=1e-12):
        return None
    for x, y in rows:
        if not math.isclose(slope * x + bias, y, rel_tol=1e-12, abs_tol=1e-9):
            return None
    return slope, bias


def _constant_region_value(rows: Sequence[tuple[float, float]]) -> Optional[float]:
    if not rows:
        return None
    value = rows[0][1]
    for _, y in rows[1:]:
        if not math.isclose(y, value, rel_tol=1e-12, abs_tol=1e-9):
            return None
    return value


def _nested_piecewise_affine_predict(
    x: float,
    low_threshold: float,
    high_threshold: float,
    low_value: float,
    mid_slope: float,
    mid_bias: float,
    high_value: float,
) -> float:
    if x < low_threshold:
        return low_value
    if x < high_threshold:
        return mid_slope * x + mid_bias
    return high_value


def _nested_piecewise_identity_cap_source(in_key: str, out_key: str, low_value: float, high_value: float) -> str:
    return _nested_piecewise_affine_source(in_key, out_key, low_value, high_value, low_value, 1.0, 0.0, high_value)


def _try_nested_piecewise_identity_cap_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    """Exact one-input nested piecewise family: constant / affine / constant with nested control flow."""
    if not is_exact_symbolic_examples_task(config):
        return None
    if config.mode != "predict_rows":
        return None
    inp = config.example_input_rows
    exp = config.expected_rows
    if not inp or not exp or len(inp) != len(exp):
        return None
    if len(inp) < 3:
        return None

    in_key: Optional[str] = None
    out_key: Optional[str] = None
    rows: List[tuple[float, float]] = []

    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping):
            return None
        if len(row_in) != 1 or len(row_ex) != 1:
            return None

        ik = str(next(iter(row_in.keys())))
        ok = str(next(iter(row_ex.keys())))
        if in_key is None:
            in_key = ik
        elif ik != in_key:
            return None
        if out_key is None:
            out_key = ok
        elif ok != out_key:
            return None

        try:
            x = float(row_in[in_key])
            y = float(row_ex[out_key])
        except (TypeError, ValueError, KeyError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        rows.append((x, y))

    rows.sort(key=lambda pair: pair[0])
    valid_model: Optional[tuple[float, float, float, float, float, float]] = None
    if len(rows) >= 4:
        for low_end in range(1, len(rows) - 2):
            for high_start in range(low_end + 2, len(rows)):
                low_rows = rows[:low_end]
                mid_rows = rows[low_end:high_start]
                high_rows = rows[high_start:]
                low_value = _constant_region_value(low_rows)
                high_value = _constant_region_value(high_rows)
                if low_value is None or high_value is None:
                    continue
                affine = _fit_single_input_affine_rows(mid_rows)
                if affine is None:
                    continue
                mid_slope, mid_bias = affine
                if not any(
                    not math.isclose(mid_slope * x + mid_bias, y, rel_tol=1e-12, abs_tol=1e-9) for x, y in low_rows
                ):
                    continue
                if not any(
                    not math.isclose(mid_slope * x + mid_bias, y, rel_tol=1e-12, abs_tol=1e-9) for x, y in high_rows
                ):
                    continue
                low_threshold = (low_value - mid_bias) / mid_slope
                high_threshold = (high_value - mid_bias) / mid_slope
                if not math.isfinite(low_threshold) or not math.isfinite(high_threshold):
                    continue
                if not low_threshold < high_threshold:
                    continue
                exact = True
                for x, y in rows:
                    pred = _nested_piecewise_affine_predict(
                        x, low_threshold, high_threshold, low_value, mid_slope, mid_bias, high_value
                    )
                    if not math.isclose(pred, y, rel_tol=1e-12, abs_tol=1e-9):
                        exact = False
                        break
                if not exact:
                    continue
                cand = (low_threshold, high_threshold, low_value, mid_slope, mid_bias, high_value)
                if valid_model is None:
                    valid_model = cand
                elif any(not math.isclose(valid_model[i], cand[i], rel_tol=1e-10, abs_tol=1e-8) for i in range(6)):
                    return None

    if valid_model is None:
        saw_low = False
        saw_mid = False
        saw_high = False
        for x, y in rows:
            pred = _nested_piecewise_affine_predict(x, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
            if not math.isclose(pred, y, rel_tol=1e-12, abs_tol=1e-9):
                return None
            if x < 0.0:
                saw_low = True
            elif x < 1.0:
                saw_mid = True
            else:
                saw_high = True
        if not saw_low or not saw_mid or not saw_high:
            return None
        valid_model = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    assert in_key is not None and out_key is not None
    low_threshold, high_threshold, low_value, mid_slope, mid_bias, high_value = valid_model
    return ExpertDraftResponse(
        ax_source=_nested_piecewise_affine_source(
            in_key, out_key, low_threshold, high_threshold, low_value, mid_slope, mid_bias, high_value
        ),
        backend_name="nested_piecewise_identity_cap_fast_path",
        metadata={
            "fast_path": "nested_piecewise_identity_cap",
            "in_key": in_key,
            "out_key": out_key,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "low_value": low_value,
            "mid_slope": mid_slope,
            "mid_bias": mid_bias,
            "high_value": high_value,
        },
    )


def _try_piecewise_threshold_identity_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    """Exact one-input/one-output zero-floor identity: ``y = x`` for non-negative ``x``, else ``0.0``."""
    if not is_exact_symbolic_examples_task(config):
        return None
    if config.mode != "predict_rows":
        return None
    inp = config.example_input_rows
    exp = config.expected_rows
    if not inp or not exp or len(inp) != len(exp):
        return None

    in_key: Optional[str] = None
    out_key: Optional[str] = None
    saw_neg = False
    saw_pos = False

    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping):
            return None
        if len(row_in) != 1 or len(row_ex) != 1:
            return None

        ik = str(next(iter(row_in.keys())))
        ok = str(next(iter(row_ex.keys())))
        if in_key is None:
            in_key = ik
        elif ik != in_key:
            return None
        if out_key is None:
            out_key = ok
        elif ok != out_key:
            return None

        try:
            x = float(row_in[in_key])
            y = float(row_ex[out_key])
        except (TypeError, ValueError, KeyError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None

        if x < 0.0:
            saw_neg = True
            if not math.isclose(y, 0.0, rel_tol=0.0, abs_tol=1e-9):
                return None
        else:
            if x > 0.0:
                saw_pos = True
            if not math.isclose(y, x, rel_tol=1e-12, abs_tol=1e-9):
                return None

    # Require both sides of the threshold to avoid ambiguous/extrapolated inference.
    if not saw_neg or not saw_pos:
        return None
    assert in_key is not None and out_key is not None
    return ExpertDraftResponse(
        ax_source=_piecewise_threshold_identity_source(in_key, out_key),
        backend_name="piecewise_threshold_identity_fast_path",
        metadata={"fast_path": "piecewise_threshold_identity", "in_key": in_key, "out_key": out_key},
    )


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


def _try_quadratic_single_input_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    """Exact single-input square-plus-bias family: ``y = x * x + c``."""
    if not is_exact_symbolic_examples_task(config):
        return None
    if config.mode != "predict_rows":
        return None
    inp = config.example_input_rows
    exp = config.expected_rows
    if not inp or not exp or len(inp) != len(exp):
        return None
    if len(inp) < 3:
        return None

    in_key: Optional[str] = None
    out_key: Optional[str] = None
    bias: Optional[float] = None
    rows: List[tuple[float, float]] = []
    distinct_xs: List[float] = []

    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping):
            return None
        if len(row_in) != 1 or len(row_ex) != 1:
            return None
        ik = str(next(iter(row_in.keys())))
        ok = str(next(iter(row_ex.keys())))
        if in_key is None:
            in_key = ik
        elif ik != in_key:
            return None
        if out_key is None:
            out_key = ok
        elif ok != out_key:
            return None
        try:
            x = float(row_in[in_key])
            y = float(row_ex[out_key])
        except (TypeError, ValueError, KeyError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        row_bias = y - (x * x)
        if bias is None:
            bias = row_bias
        elif not math.isclose(row_bias, bias, rel_tol=1e-12, abs_tol=1e-9):
            return None
        rows.append((x, y))
        if not any(math.isclose(x, seen, rel_tol=0.0, abs_tol=1e-12) for seen in distinct_xs):
            distinct_xs.append(x)

    if bias is None or len(distinct_xs) < 3:
        return None
    for x, y in rows:
        pred = x * x + bias
        if not math.isclose(pred, y, rel_tol=1e-12, abs_tol=1e-9):
            return None

    assert in_key is not None and out_key is not None
    return ExpertDraftResponse(
        ax_source=_quadratic_single_input_source(in_key, out_key, bias),
        backend_name="quadratic_single_input_fast_path",
        metadata={
            "fast_path": "quadratic_single_input",
            "in_key": in_key,
            "out_key": out_key,
            "bias": bias,
        },
    )


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _is_strict01_interior(y: float) -> bool:
    return y > 0.0 and y < 1.0


def _minmax_blend_source(k1: str, k2: str, out_var: str) -> str:
    return f"{out_var} = max(0.0, min({k1} + {k2}, 1.0));\n"


def _three_way_maxmin_source(k1: str, k2: str, k3: str, out_var: str) -> str:
    return f"{out_var} = max(min({k1}, {k2}), {k3});\n"


def _try_three_way_maxmin_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    """Exact three-input symbolic min/max family: ``out = max(min(a, b), c)``."""
    if not is_exact_symbolic_examples_task(config):
        return None
    if config.mode != "predict_rows":
        return None
    inp = config.example_input_rows
    exp = config.expected_rows
    if not inp or not exp or len(inp) != len(exp):
        return None
    if len(inp) < 3:
        return None

    in_keys: Optional[tuple[str, str, str]] = None
    out_key: Optional[str] = None
    rows: List[tuple[float, float, float, float]] = []

    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping):
            return None
        if len(row_in) != 3 or len(row_ex) != 1:
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
            a = float(row_in[in_keys[0]])
            b = float(row_in[in_keys[1]])
            c = float(row_in[in_keys[2]])
            y = float(row_ex[out_key])
        except (TypeError, ValueError, KeyError):
            return None
        if not math.isfinite(a) or not math.isfinite(b) or not math.isfinite(c) or not math.isfinite(y):
            return None
        rows.append((a, b, c, y))

    assert in_keys is not None and out_key is not None
    candidates = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
    matches: List[tuple[int, int, int]] = []
    for i, j, k in candidates:
        exact = True
        for a, b, c, y in rows:
            vals = (a, b, c)
            pred = max(min(vals[i], vals[j]), vals[k])
            if not math.isclose(pred, y, rel_tol=1e-12, abs_tol=1e-9):
                exact = False
                break
        if exact:
            matches.append((i, j, k))
    if len(matches) != 1:
        return None

    i, j, k = matches[0]
    return ExpertDraftResponse(
        ax_source=_three_way_maxmin_source(in_keys[i], in_keys[j], in_keys[k], out_key),
        backend_name="three_way_maxmin_fast_path",
        metadata={
            "fast_path": "three_way_maxmin",
            "in_keys": [in_keys[i], in_keys[j], in_keys[k]],
            "out_key": out_key,
        },
    )


def _try_minmax_blend_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    """Exact two-input clamp blend: ``out = max(0.0, min(x1 + x2, 1.0))``."""
    if not is_exact_symbolic_examples_task(config):
        return None
    if config.mode != "predict_rows":
        return None
    inp = config.example_input_rows
    exp = config.expected_rows
    if not inp or not exp or len(inp) != len(exp):
        return None
    if len(inp) < 3:
        return None

    in_keys: Optional[tuple[str, str]] = None
    out_key: Optional[str] = None
    saw_low = False
    saw_mid = False
    saw_high = False

    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping):
            return None
        if len(row_in) != 2 or len(row_ex) != 1:
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
            x1 = float(row_in[in_keys[0]])
            x2 = float(row_in[in_keys[1]])
            y = float(row_ex[out_key])
        except (TypeError, ValueError, KeyError):
            return None
        if not math.isfinite(x1) or not math.isfinite(x2) or not math.isfinite(y):
            return None

        raw = x1 + x2
        pred = _clamp01(raw)
        if not math.isclose(pred, y, rel_tol=1e-12, abs_tol=1e-9):
            return None
        if raw < 0.0 and math.isclose(y, 0.0, rel_tol=0.0, abs_tol=1e-9):
            saw_low = True
        elif 0.0 < raw < 1.0 and math.isclose(y, raw, rel_tol=1e-12, abs_tol=1e-9):
            saw_mid = True
        elif raw > 1.0 and math.isclose(y, 1.0, rel_tol=0.0, abs_tol=1e-9):
            saw_high = True

    if not saw_low or not saw_mid or not saw_high:
        return None

    assert in_keys is not None and out_key is not None
    return ExpertDraftResponse(
        ax_source=_minmax_blend_source(in_keys[0], in_keys[1], out_key),
        backend_name="minmax_blend_fast_path",
        metadata={"fast_path": "minmax_blend", "in_keys": [in_keys[0], in_keys[1]], "out_key": out_key},
    )


def _solve_linear_system(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> Optional[List[float]]:
    """Solve square system by Gaussian elimination with partial pivoting; None when singular."""
    n = len(matrix)
    if n == 0 or len(rhs) != n:
        return None
    a: List[List[float]] = []
    for i in range(n):
        row = list(matrix[i])
        if len(row) != n:
            return None
        a.append(row + [float(rhs[i])])

    for col in range(n):
        pivot = col
        best = abs(a[col][col])
        for r in range(col + 1, n):
            v = abs(a[r][col])
            if v > best:
                best = v
                pivot = r
        if best < 1e-15:
            return None
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        pv = a[col][col]
        for j in range(col, n + 1):
            a[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            f = a[r][col]
            if abs(f) < 1e-18:
                continue
            for j in range(col, n + 1):
                a[r][j] -= f * a[col][j]
    return [a[i][n] for i in range(n)]


def _signed_weighted_sum_source_expr(in_keys: Sequence[str], weights: Sequence[float], bias: float) -> str:
    parts: List[str] = []
    for key, weight in zip(in_keys, weights):
        if math.isclose(weight, 0.0, abs_tol=1e-12, rel_tol=1e-12):
            continue
        term = f"{_linear_xy_coeff_str(abs(weight))} * {key}"
        if not parts:
            parts.append(f"-{term}" if weight < 0.0 else term)
        else:
            parts.append(f"- {term}" if weight < 0.0 else f"+ {term}")
    if not math.isclose(bias, 0.0, abs_tol=1e-12, rel_tol=1e-12):
        term = _linear_xy_coeff_str(abs(bias))
        if not parts:
            parts.append(f"-{term}" if bias < 0.0 else term)
        else:
            parts.append(f"- {term}" if bias < 0.0 else f"+ {term}")
    return " ".join(parts) if parts else "0.0"


def _clamped_affine_multi_input_source(
    out_key: str,
    in_keys: Sequence[str],
    weights: Sequence[float],
    bias: float,
    *,
    low: float = 0.0,
    high: float = 1.0,
) -> str:
    inner = _signed_weighted_sum_source_expr(in_keys, weights, bias)
    return (
        f"{out_key} = max({_linear_xy_coeff_str(low)}, "
        f"min({_linear_xy_coeff_str(high)}, {inner}));\n"
    )


def _affine_multi_input_source(out_key: str, in_keys: Sequence[str], weights: Sequence[float], bias: float) -> str:
    expr = _signed_weighted_sum_source_expr(in_keys, weights, bias)
    return f"{out_key} = {expr};\n"


def _try_affine_multi_input_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    """Exact one-output affine fit for N>=3 numeric inputs: out = sum(w_i * x_i) + b."""
    if not is_exact_symbolic_examples_task(config):
        return None
    if config.mode != "predict_rows":
        return None
    inp = config.example_input_rows
    exp = config.expected_rows
    if not inp or not exp or len(inp) != len(exp):
        return None
    if len(inp) < 4:
        return None

    in_keys: Optional[List[str]] = None
    out_key: Optional[str] = None
    xs: List[List[float]] = []
    ys: List[float] = []

    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping):
            return None
        if len(row_ex) != 1:
            return None
        keys = sorted(str(k) for k in row_in.keys())
        if in_keys is None:
            if len(keys) < 3:
                return None
            in_keys = keys
        elif keys != in_keys:
            return None
        ok = str(next(iter(row_ex.keys())))
        if out_key is None:
            out_key = ok
        elif ok != out_key:
            return None
        try:
            xrow = [float(row_in[k]) for k in in_keys]
            yv = float(row_ex[out_key])
        except (TypeError, ValueError, KeyError):
            return None
        xs.append(xrow)
        ys.append(yv)

    assert in_keys is not None and out_key is not None
    n_in = len(in_keys)
    unknowns = n_in + 1
    if len(xs) < unknowns:
        return None

    valid_solution: Optional[List[float]] = None
    indices = range(len(xs))
    for combo in itertools.combinations(indices, unknowns):
        mat: List[List[float]] = []
        rhs: List[float] = []
        for idx in combo:
            mat.append(xs[idx] + [1.0])
            rhs.append(ys[idx])
        sol = _solve_linear_system(mat, rhs)
        if sol is None:
            continue
        ok = True
        for xrow, yv in zip(xs, ys):
            pred = sum(sol[i] * xrow[i] for i in range(n_in)) + sol[-1]
            if not math.isclose(pred, yv, rel_tol=1e-11, abs_tol=1e-8):
                ok = False
                break
        if not ok:
            continue
        if valid_solution is None:
            valid_solution = sol
        else:
            # Multiple materially different exact fits => ambiguous.
            if any(not math.isclose(valid_solution[i], sol[i], rel_tol=1e-10, abs_tol=1e-8) for i in range(unknowns)):
                return None

    if valid_solution is None:
        return None

    src = _affine_multi_input_source(out_key, in_keys, valid_solution[:-1], valid_solution[-1])
    return ExpertDraftResponse(
        ax_source=src,
        backend_name="affine_multi_input_fast_path",
        metadata={
            "fast_path": "affine_multi_input",
            "in_keys": list(in_keys),
            "out_key": out_key,
            "weights": list(valid_solution[:-1]),
            "bias": valid_solution[-1],
        },
    )


def _try_clamped_affine_multi_input_fast_path(
    config: CopilotSearchConfig,
    *,
    min_inputs: int,
    max_inputs: Optional[int],
    backend_name: str,
    fast_path_name: str,
) -> Optional[ExpertDraftResponse]:
    """Exact ``[0.0, 1.0]`` clamped affine fit using interior rows to recover weights and bias."""
    if not is_exact_symbolic_examples_task(config):
        return None
    if not _hint_suggests_clamp_family(config):
        return None
    if config.mode != "predict_rows":
        return None
    inp = config.example_input_rows
    exp = config.expected_rows
    if not inp or not exp or len(inp) != len(exp):
        return None

    in_keys: Optional[List[str]] = None
    out_key: Optional[str] = None
    xs: List[List[float]] = []
    ys: List[float] = []
    interior_indices: List[int] = []

    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping):
            return None
        if len(row_ex) != 1:
            return None
        keys = sorted(str(k) for k in row_in.keys())
        if in_keys is None:
            if len(keys) < min_inputs:
                return None
            if max_inputs is not None and len(keys) > max_inputs:
                return None
            in_keys = keys
        elif keys != in_keys:
            return None
        ok = str(next(iter(row_ex.keys())))
        if out_key is None:
            out_key = ok
        elif ok != out_key:
            return None
        try:
            xrow = [float(row_in[k]) for k in in_keys]
            yv = float(row_ex[out_key])
        except (TypeError, ValueError, KeyError):
            return None
        if not all(math.isfinite(v) for v in xrow) or not math.isfinite(yv):
            return None
        xs.append(xrow)
        ys.append(yv)
        if _is_strict01_interior(yv):
            interior_indices.append(len(xs) - 1)

    assert in_keys is not None and out_key is not None
    unknowns = len(in_keys) + 1
    if len(interior_indices) < unknowns:
        return None

    valid_solution: Optional[List[float]] = None
    for combo in itertools.combinations(interior_indices, unknowns):
        mat = [xs[idx] + [1.0] for idx in combo]
        rhs = [ys[idx] for idx in combo]
        sol = _solve_linear_system(mat, rhs)
        if sol is None:
            continue
        ok = True
        for idx in interior_indices:
            pred = sum(sol[i] * xs[idx][i] for i in range(len(in_keys))) + sol[-1]
            if not math.isclose(pred, ys[idx], rel_tol=1e-11, abs_tol=1e-8):
                ok = False
                break
        if not ok:
            continue
        for xrow, yv in zip(xs, ys):
            raw = sum(sol[i] * xrow[i] for i in range(len(in_keys))) + sol[-1]
            pred = _clamp01(raw)
            if not math.isclose(pred, yv, rel_tol=1e-11, abs_tol=1e-8):
                ok = False
                break
        if not ok:
            continue
        if valid_solution is None:
            valid_solution = sol
        elif any(
            not math.isclose(valid_solution[i], sol[i], rel_tol=1e-10, abs_tol=1e-8) for i in range(unknowns)
        ):
            return None

    if valid_solution is None:
        return None

    metadata: Dict[str, Any] = {
        "fast_path": fast_path_name,
        "in_keys": list(in_keys),
        "out_key": out_key,
        "weights": list(valid_solution[:-1]),
        "bias": valid_solution[-1],
        "low": 0.0,
        "high": 1.0,
    }
    if len(in_keys) == 2:
        metadata["a"] = valid_solution[0]
        metadata["b"] = valid_solution[1]
        metadata["c"] = valid_solution[2]

    return ExpertDraftResponse(
        ax_source=_clamped_affine_multi_input_source(out_key, in_keys, valid_solution[:-1], valid_solution[-1]),
        backend_name=backend_name,
        metadata=metadata,
    )


def _try_bounded_affine_multi_input_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    """Exact multi-input clamp: ``out = max(0.0, min(1.0, sum(w_i * x_i) + c))`` for N>=3 inputs."""
    return _try_clamped_affine_multi_input_fast_path(
        config,
        min_inputs=3,
        max_inputs=None,
        backend_name="bounded_affine_multi_input_fast_path",
        fast_path_name="bounded_affine_multi_input",
    )


def _try_bounded_affine2_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    """``out = max(0, min(1, a*x1 + b*x2 + c))`` with two numeric inputs and one output (exact fit only)."""
    return _try_clamped_affine_multi_input_fast_path(
        config,
        min_inputs=2,
        max_inputs=2,
        backend_name="bounded_affine2_fast_path",
        fast_path_name="bounded_affine2",
    )


def _two_input_interaction_source(
    out_key: str,
    k1: str,
    k2: str,
    w_ab: float,
    w_a: float,
    w_b: float,
    bias: float,
) -> str:
    parts: List[str] = []

    def _append_term(coeff: float, expr: str) -> None:
        if math.isclose(coeff, 0.0, abs_tol=1e-12, rel_tol=1e-12):
            return
        ac = abs(coeff)
        if math.isclose(ac, 1.0, abs_tol=1e-12, rel_tol=1e-12):
            rendered = expr
        else:
            rendered = f"{_linear_xy_coeff_str(ac)} * {expr}"
        if not parts:
            parts.append(f"-{rendered}" if coeff < 0 else rendered)
        else:
            parts.append(f"- {rendered}" if coeff < 0 else f"+ {rendered}")

    def _append_bias(coeff: float) -> None:
        if math.isclose(coeff, 0.0, abs_tol=1e-12, rel_tol=1e-12):
            return
        rendered = _linear_xy_coeff_str(abs(coeff))
        if not parts:
            parts.append(f"-{rendered}" if coeff < 0 else rendered)
        else:
            parts.append(f"- {rendered}" if coeff < 0 else f"+ {rendered}")

    _append_term(w_ab, f"{k1} * {k2}")
    _append_term(w_a, k1)
    _append_term(w_b, k2)
    _append_bias(bias)
    expr = " ".join(parts) if parts else "0.0"
    return f"{out_key} = {expr};\n"


def _try_two_input_interaction_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    """Exact two-input interaction fit: ``out = w_ab*a*b + w_a*a + w_b*b + c``."""
    if not is_exact_symbolic_examples_task(config):
        return None
    if config.mode != "predict_rows":
        return None
    inp = config.example_input_rows
    exp = config.expected_rows
    if not inp or not exp or len(inp) != len(exp):
        return None
    if len(inp) < 4:
        return None

    in_keys: Optional[tuple[str, str]] = None
    out_key: Optional[str] = None
    basis_rows: List[List[float]] = []
    ys: List[float] = []

    for row_in, row_ex in zip(inp, exp):
        if not isinstance(row_in, Mapping) or not isinstance(row_ex, Mapping):
            return None
        if len(row_in) != 2 or len(row_ex) != 1:
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
            a = float(row_in[in_keys[0]])
            b = float(row_in[in_keys[1]])
            y = float(row_ex[out_key])
        except (TypeError, ValueError, KeyError):
            return None
        if not math.isfinite(a) or not math.isfinite(b) or not math.isfinite(y):
            return None
        basis_rows.append([a * b, a, b, 1.0])
        ys.append(y)

    assert in_keys is not None and out_key is not None
    valid_solution: Optional[List[float]] = None
    for combo in itertools.combinations(range(len(basis_rows)), 4):
        mat = [basis_rows[idx] for idx in combo]
        rhs = [ys[idx] for idx in combo]
        sol = _solve_linear_system(mat, rhs)
        if sol is None:
            continue
        ok = True
        for row, y in zip(basis_rows, ys):
            pred = sum(sol[i] * row[i] for i in range(4))
            if not math.isclose(pred, y, rel_tol=1e-12, abs_tol=1e-9):
                ok = False
                break
        if not ok:
            continue
        if valid_solution is None:
            valid_solution = sol
        elif any(
            not math.isclose(valid_solution[i], sol[i], rel_tol=1e-10, abs_tol=1e-8) for i in range(4)
        ):
            return None

    if valid_solution is None:
        return None

    src = _two_input_interaction_source(
        out_key,
        in_keys[0],
        in_keys[1],
        valid_solution[0],
        valid_solution[1],
        valid_solution[2],
        valid_solution[3],
    )
    return ExpertDraftResponse(
        ax_source=src,
        backend_name="two_input_interaction_fast_path",
        metadata={
            "fast_path": "two_input_interaction",
            "in_keys": [in_keys[0], in_keys[1]],
            "out_key": out_key,
            "w_ab": valid_solution[0],
            "w_a": valid_solution[1],
            "w_b": valid_solution[2],
            "bias": valid_solution[3],
        },
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


def _constant_offset_from_row_comparisons(
    row_comparisons: Sequence[Mapping[str, Any]],
) -> Optional[float]:
    """If one output error is approximately constant across rows, return that signed offset."""
    deltas: List[float] = []
    for row in row_comparisons:
        pred = row.get("predicted")
        exp = row.get("expected")
        if not isinstance(pred, Mapping) or not isinstance(exp, Mapping):
            continue
        shared = [k for k in exp.keys() if k in pred]
        if len(shared) != 1:
            continue
        k = shared[0]
        try:
            d = float(pred[k]) - float(exp[k])
        except (TypeError, ValueError):
            continue
        if math.isfinite(d):
            deltas.append(d)
    if len(deltas) < 3:
        return None
    mean = sum(deltas) / len(deltas)
    if abs(mean) < 1e-6:
        return None
    spread = max(abs(d - mean) for d in deltas)
    if spread > 1e-4:
        return None
    return mean


_GOAL_HAS_INTERACTION_TERM = re.compile(r"\b[a-zA-Z_]\w*\s*\*\s*[a-zA-Z_]\w*\b")


def _goal_suggests_cross_term_with_additive(goal: str) -> bool:
    g = (goal or "").lower()
    return bool(_GOAL_HAS_INTERACTION_TERM.search(g) and "+" in g)


def _extract_numeric_row_errors(
    row_comparisons: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, float]], List[float], Optional[str]]:
    """Return numeric inputs per row, signed errors, and target output key when unambiguous."""
    xs: List[Dict[str, float]] = []
    errs: List[float] = []
    out_key: Optional[str] = None
    for row in row_comparisons:
        pred = row.get("predicted")
        exp = row.get("expected")
        inp = row.get("inputs")
        if not isinstance(pred, Mapping) or not isinstance(exp, Mapping) or not isinstance(inp, Mapping):
            continue
        shared = [k for k in exp.keys() if k in pred]
        if len(shared) != 1:
            continue
        k = str(shared[0])
        try:
            y_pred = float(pred[k])
            y_exp = float(exp[k])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(y_pred) and math.isfinite(y_exp)):
            continue
        num_inp: Dict[str, float] = {}
        ok_inputs = True
        for ik, iv in inp.items():
            try:
                fv = float(iv)
            except (TypeError, ValueError):
                ok_inputs = False
                break
            if not math.isfinite(fv):
                ok_inputs = False
                break
            num_inp[str(ik)] = fv
        if not ok_inputs:
            continue
        if out_key is None:
            out_key = k
        elif out_key != k:
            return [], [], None
        xs.append(num_inp)
        errs.append(y_pred - y_exp)
    if len(xs) != len(errs) or not xs:
        return [], [], None
    return xs, errs, out_key


def _corr_abs(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = 0.0
    denx = 0.0
    deny = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        denx += dx * dx
        deny += dy * dy
    if denx <= 1e-18 or deny <= 1e-18:
        return 0.0
    return abs(num / math.sqrt(denx * deny))


def _symbolic_row_error_hints(
    goal: str,
    row_comparisons: Sequence[Mapping[str, Any]],
) -> List[str]:
    """Heuristic, deterministic symbolic repair hints inferred from row error structure."""
    hints: List[str] = []

    offset = _constant_offset_from_row_comparisons(row_comparisons)
    if offset is not None:
        hints.append(
            "Row errors show a **near-constant offset** across rows "
            f"(`predicted - expected ~= {offset:.6g}`): **missing constant/bias** or altered additive bias is likely; "
            "**Preserve additive bias exactly** (intercept / constant term such as `+ 1.0`)."
        )

    xs, errs, _ = _extract_numeric_row_errors(row_comparisons)
    if len(xs) < 3:
        return hints

    input_keys = sorted(xs[0].keys())
    for r in xs[1:]:
        if sorted(r.keys()) != input_keys:
            return hints

    # Distorted coefficient on one variable: error tracks one input (unary / linear in one ABI name).
    best_key = ""
    best_corr = 0.0
    for k in input_keys:
        corr = _corr_abs([r[k] for r in xs], errs)
        if corr > best_corr:
            best_corr = corr
            best_key = k
    if best_key and best_corr >= 0.85:
        hints.append(
            f"Error varies strongly with `{best_key}` (corr~{best_corr:.2f}); "
            f"a **distorted unary coefficient** on `{best_key}` is likely. Preserve variable coefficients exactly; "
            "do not replace interaction terms with scaled unary terms."
        )
    elif best_key and 0.7 <= best_corr < 0.85:
        hints.append(
            f"Error correlates with `{best_key}` (corr~{best_corr:.2f}); "
            f"a **distorted unary coefficient** on `{best_key}` may be present — recheck unary/additive terms vs the goal."
        )

    # Interaction term (e.g. a * b): error aligns with product more than unary — or moderate product signal.
    if _goal_suggests_cross_term_with_additive(goal) and len(input_keys) >= 2:
        best_pair = ""
        best_pair_corr = 0.0
        for i in range(len(input_keys)):
            for j in range(i + 1, len(input_keys)):
                a = input_keys[i]
                b = input_keys[j]
                corr = _corr_abs([r[a] * r[b] for r in xs], errs)
                if corr > best_pair_corr:
                    best_pair_corr = corr
                    best_pair = f"{a} * {b}"
        strong = bool(
            best_pair and best_pair_corr >= max(0.8, best_corr + 0.05)
        )
        moderate = bool(
            best_pair
            and 0.5 <= best_pair_corr < max(0.8, best_corr + 0.05)
            and best_pair_corr + 1e-9 >= best_corr
        )
        if strong:
            hints.append(
                "**Missing or wrong interaction term** (e.g. `a * b`): row errors track the product "
                f"`{best_pair}` (corr~{best_pair_corr:.2f}). "
                f"**Preserve interaction terms exactly**; keep an explicit `{best_pair}` (or equivalent). "
                "**Do not replace interaction terms with boolean guards or branch logic.** "
                "**Do not replace interaction terms with scaled unary terms.**"
            )
        elif moderate:
            hints.append(
                f"Row errors partially align with `{best_pair}` (corr~{best_pair_corr:.2f}); "
                f"check for a **missing interaction term** like `{best_pair}` — **preserve interaction terms exactly**; "
                "**do not replace interaction terms with boolean guards or branch logic**; "
                "**do not replace interaction terms with scaled unary terms.**"
            )
        elif max(abs(e) for e in errs) > 1e-9:
            hints.append(
                "Goal appears to combine a **product** (`a * b`), unary terms, and a **bias**; row errors remain. "
                "**Preserve interaction terms exactly** and **Preserve additive bias exactly**; "
                "**do not replace interaction terms with boolean guards or branch logic**; "
                "**do not replace interaction terms with scaled unary terms** (e.g. `a * 2` instead of `a * b`)."
            )
    return hints


def _exact_symbolic_row_repair_preamble() -> str:
    """Fixed bullets for exact-symbolic tasks with row data (repair prompts only)."""
    return (
        "- **Preserve interaction terms exactly** (e.g. `a * b` when the goal requires a product of inputs).\n"
        "- **Preserve additive bias exactly** (intercepts, constant offsets, explicit bias like `+ 1.0`).\n"
        "- **Do not replace interaction terms with boolean guards or branch logic** (`if`, `else`, `&&`, `||`).\n"
        "- **Do not replace interaction terms with scaled unary terms** (e.g. `a * 2` or `2.0 * a` standing in for `a * b`).\n"
        "- Check row errors for: **missing interaction term** (`a * b`), **distorted unary coefficient** on an additive term, "
        "or **missing constant/bias** like `+ 1.0`.\n"
        "- Do not accept common wrong shapes when the goal specifies `a * b + …` — e.g. `y = a + a * b` (wrong bias) "
        "or `y = a * 2 + a + 1.0` (scaled unary instead of `a * b`) unless they match every row.\n"
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
        if evaluation.row_comparisons:
            parts.append("### Exact symbolic repair cues")
            parts.append(
                "The program compiles but mismatches examples — fix semantics, not syntax only:\n"
            )
            parts.append(_exact_symbolic_row_repair_preamble().rstrip())
            parts.append("")
            for h in _symbolic_row_error_hints(goal, evaluation.row_comparisons):
                parts.append(f"- {h}")
            parts.append("")
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


def _build_copilot_draft_request(config: CopilotSearchConfig) -> ExpertDraftRequest:
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
    return ExpertDraftRequest(goal=config.goal, context=ctx)


def _try_exact_symbolic_fast_path(config: CopilotSearchConfig) -> Optional[ExpertDraftResponse]:
    fast = _try_nested_piecewise_identity_cap_fast_path(config)
    if fast is None:
        fast = _try_piecewise_threshold_identity_fast_path(config)
    if fast is None:
        fast = _try_absolute_value_piecewise_fast_path(config)
    if fast is None:
        fast = _try_linear_xy_fast_path(config)
    if fast is None:
        fast = _try_quadratic_single_input_fast_path(config)
    if fast is None:
        fast = _try_max_of_two_fast_path(config)
    if fast is None:
        fast = _try_three_way_maxmin_fast_path(config)
    if fast is None:
        fast = _try_minmax_blend_fast_path(config)
    if fast is None:
        fast = _try_bounded_affine_multi_input_fast_path(config)
    if fast is None:
        fast = _try_bounded_affine2_fast_path(config)
    if fast is None:
        fast = _try_two_input_interaction_fast_path(config)
    if fast is None:
        fast = _try_affine_multi_input_fast_path(config)
    return fast


def run_copilot_draft(
    config: CopilotSearchConfig, draft_req: Optional[ExpertDraftRequest] = None
) -> tuple[ExpertDraftRequest, ExpertDraftResponse]:
    """One draft step with shared fast-path inference before expert fallback."""
    if draft_req is None:
        draft_req = _build_copilot_draft_request(config)
    draft_resp = _try_exact_symbolic_fast_path(config)
    if draft_resp is None:
        draft_resp = config.expert.draft_program(draft_req)
    return draft_req, draft_resp


def run_copilot_search(config: CopilotSearchConfig) -> CopilotSearchResult:
    from axiom.copilot.artifacts import expert_response_to_dict, persist_copilot_artifacts

    tt_meta = _train_tabular_meta(config)
    draft_req = _build_copilot_draft_request(config)
    try:
        draft_req, draft_resp = run_copilot_draft(config, draft_req)
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
