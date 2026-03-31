"""Parse copilot **tabular JSON** for ``train_tabular`` search / CLI / HTTP / Studio.

Schema (object, UTF-8 JSON file or textarea)::

    {
      "target_var": "y",
      "train_rows": [{"inputs": {"x": 0.1}, "expected": {"y": 0.2}}, ...],
      "eval_rows": [{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}, ...],
      "epochs": 50,
      "learning_rate": 0.01,
      "weight_decay": 0.0,
      "batch_size": 32
    }

Each row **merges** ``inputs`` and ``expected`` into one dict for :func:`~axiom.copilot.evaluator.evaluate_program`
(ABI keys + scalar ``target_var`` label). ``expected`` objects are also kept separately for ``neg_mse`` scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from axiom.copilot.models import TrainTabularParams


@dataclass(frozen=True)
class TabularJsonPayload:
    target_var: str
    train_rows: Tuple[Dict[str, Any], ...]
    eval_rows: Tuple[Dict[str, Any], ...]
    eval_expected_rows: Tuple[Dict[str, Any], ...]
    params: TrainTabularParams


def _one_row(obj: Any, i: int, *, field_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(obj, dict):
        raise ValueError(f'{field_name}[{i}] must be an object, not {type(obj).__name__}.')
    if "inputs" not in obj or "expected" not in obj:
        raise ValueError(f'{field_name}[{i}] must have "inputs" and "expected".')
    ins, exp = obj["inputs"], obj["expected"]
    if not isinstance(ins, dict) or not isinstance(exp, dict):
        raise ValueError(f'{field_name}[{i}]: inputs and expected must be objects.')
    merged: Dict[str, Any] = {**dict(ins), **dict(exp)}
    return merged, dict(exp)


def parse_tabular_json_dict(data: Dict[str, Any]) -> TabularJsonPayload:
    if not isinstance(data, dict):
        raise ValueError("Root JSON value must be an object.")
    tv = data.get("target_var")
    if not isinstance(tv, str) or not tv.strip():
        raise ValueError('Missing or invalid string "target_var".')
    tv = tv.strip()
    for key in ("train_rows", "eval_rows"):
        rows = data.get(key)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'"{key}" must be a non-empty array.')
    train_raw = data["train_rows"]
    eval_raw = data["eval_rows"]
    train_merged: List[Dict[str, Any]] = []
    eval_merged: List[Dict[str, Any]] = []
    eval_exp: List[Dict[str, Any]] = []
    for i, row in enumerate(train_raw):
        m, _e = _one_row(row, i, field_name="train_rows")
        train_merged.append(m)
    for i, row in enumerate(eval_raw):
        m, e = _one_row(row, i, field_name="eval_rows")
        eval_merged.append(m)
        eval_exp.append(e)

    def _opt_int(name: str, default: int) -> int:
        v = data.get(name, default)
        if v is None:
            return default
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f'"{name}" must be a number.')
        return int(v)

    def _opt_float(name: str, default: float) -> float:
        v = data.get(name, default)
        if v is None:
            return float(default)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f'"{name}" must be a number.')
        return float(v)

    params = TrainTabularParams(
        epochs=max(0, _opt_int("epochs", 30)),
        learning_rate=_opt_float("learning_rate", 0.01),
        weight_decay=_opt_float("weight_decay", 0.0),
        batch_size=max(1, _opt_int("batch_size", 32)),
    )
    return TabularJsonPayload(
        target_var=tv,
        train_rows=tuple(train_merged),
        eval_rows=tuple(eval_merged),
        eval_expected_rows=tuple(eval_exp),
        params=params,
    )


def parse_tabular_json_text(text: str) -> TabularJsonPayload:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError("Root must be a JSON object.")
    return parse_tabular_json_dict(raw)
