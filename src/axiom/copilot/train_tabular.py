"""In-memory tabular train + eval for copilot ``evaluate_program(..., mode=\"train_tabular\")``."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from axiom.copilot.models import ProgramFailure, ProgramMetric
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _abi_outputs_from_trunk_row


def _env_tensor_to_python(t: torch.Tensor) -> Any:
    x = t.detach().cpu()
    if x.dim() >= 1:
        x = x[0]
    if x.dim() == 0:
        return float(x.item())
    flat = x.flatten().tolist()
    if len(flat) == 1:
        return float(flat[0])
    return [float(v) for v in flat]


def _trunk_dim(abi: Dict[str, int], abi_widths: Dict[str, int]) -> int:
    return max((abi[n] + max(1, int(abi_widths.get(n, 1))) for n in abi), default=16)


def _scalar_numeric(val: Any, *, ctx: str) -> float:
    if isinstance(val, bool):
        raise TypeError(f"{ctx}: boolean values are not allowed as numeric scalars")
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError as e:
            raise TypeError(f"{ctx}: cannot parse string as float") from e
    raise TypeError(f"{ctx}: expected int/float, got {type(val).__name__}")


def _fill_trunk_row(
    row: Dict[str, Any],
    abi: Dict[str, int],
    dim: int,
    abi_widths: Dict[str, int],
    *,
    row_index: Optional[int],
) -> torch.Tensor:
    x = torch.zeros(dim, dtype=torch.float32)
    ri = f"row {row_index}" if row_index is not None else "row"
    for name, col in abi.items():
        if col >= dim or name not in row:
            continue
        w = max(1, int(abi_widths.get(name, 1)))
        end = min(col + w, dim)
        val = row[name]
        if isinstance(val, (list, tuple)):
            if len(val) != w:
                raise ValueError(f"{ri}: {name!r} list length {len(val)} != ABI width {w}")
            for i in range(end - col):
                x[col + i] = _scalar_numeric(val[i], ctx=f"{ri}.{name}[{i}]")
        else:
            if w != 1:
                raise ValueError(f"{ri}: {name!r} must be a length-{w} list for this ABI")
            x[col] = _scalar_numeric(val, ctx=f"{ri}.{name}")
    return x


def _row_tensors(
    rows: Sequence[Dict[str, Any]],
    abi: Dict[str, int],
    dim: int,
    abi_widths: Dict[str, int],
    target_var: str,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[ProgramFailure]]:
    failures: List[ProgramFailure] = []
    target_col = abi.get(target_var)
    if target_col is None:
        failures.append(
            ProgramFailure(
                stage="train",
                kind="value",
                message=f"target_var {target_var!r} is not in the program ABI (known: {sorted(abi.keys())}).",
                detail="ValueError",
            )
        )
        return None, None, failures
    tw = max(1, int(abi_widths.get(target_var, 1)))
    if tw != 1:
        failures.append(
            ProgramFailure(
                stage="train",
                kind="value",
                message=f"train_tabular supports scalar targets only; {target_var!r} has ABI width {tw}.",
                detail="ValueError",
            )
        )
        return None, None, failures

    hs: List[torch.Tensor] = []
    ys: List[float] = []
    for i, row in enumerate(rows):
        if target_var not in row:
            failures.append(
                ProgramFailure(
                    stage="train",
                    kind="value",
                    message=f"Missing target key {target_var!r} in train/eval row index {i}.",
                    detail="KeyError",
                )
            )
            return None, None, failures
        try:
            yv = _scalar_numeric(row[target_var], ctx=f"row {i}.{target_var}")
            x = _fill_trunk_row(row, abi, dim, abi_widths, row_index=i)
            x[target_col : target_col + tw] = 0.0
            hs.append(x)
            ys.append(yv)
        except (TypeError, ValueError) as e:
            failures.append(
                ProgramFailure(
                    stage="train",
                    kind="schema",
                    message=str(e),
                    detail=type(e).__name__,
                )
            )
            return None, None, failures

    if not hs:
        return None, None, failures
    h = torch.stack(hs, dim=0)
    y = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
    return h, y, failures


def _failure(stage: str, kind: str, exc: BaseException) -> ProgramFailure:
    return ProgramFailure(
        stage=stage,
        kind=kind,
        message=str(exc),
        detail=type(exc).__name__,
    )


@dataclass
class TrainTabularOutcome:
    failures: List[ProgramFailure] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    program_metrics: List[ProgramMetric] = field(default_factory=list)
    predictions_sample: Optional[List[Dict[str, Any]]] = None
    trace_snippet: Optional[Dict[str, Any]] = None
    compile_stage_reached: str = "block"


def run_train_tabular(
    block: InterpretedBlock,
    *,
    train_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    target_var: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    predictions_sample_limit: int,
    include_trace_snippet: bool,
    score_fn: Optional[Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, float]]] = None,
    expected_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> TrainTabularOutcome:
    out = TrainTabularOutcome()
    abi = block.abi
    aw = dict(getattr(block, "abi_widths", {}) or {})
    dim = _trunk_dim(abi, aw)

    if not train_rows:
        out.failures.append(
            ProgramFailure(
                stage="train",
                kind="value",
                message="train_tabular requires non-empty train_rows.",
                detail="ValueError",
            )
        )
        return out
    if not eval_rows:
        out.failures.append(
            ProgramFailure(
                stage="train",
                kind="value",
                message="train_tabular requires non-empty eval_rows.",
                detail="ValueError",
            )
        )
        return out

    h_train, y_train, f1 = _row_tensors(train_rows, abi, dim, aw, target_var)
    out.failures.extend(f1)
    if f1 or h_train is None:
        return out

    h_eval, y_eval, f2 = _row_tensors(eval_rows, abi, dim, aw, target_var)
    out.failures.extend(f2)
    if f2 or h_eval is None:
        return out

    out.compile_stage_reached = "train"
    tc = abi[target_var]
    trainable = [p for p in block.parameters() if p.requires_grad]
    if not trainable:
        out.warnings.append(
            "no_trainable_parameters: skipping optimizer; metrics reflect the fixed (untrained) program."
        )
    else:
        opt = torch.optim.Adam(trainable, lr=float(learning_rate), weight_decay=float(weight_decay))
        block.train()
        B = h_train.shape[0]
        bs = max(1, int(batch_size))
        ep = max(0, int(epochs))
        try:
            for _ in range(ep):
                perm = torch.randperm(B)
                for start in range(0, B, bs):
                    idx = perm[start : start + bs]
                    hb = h_train[idx]
                    yb = y_train[idx]
                    opt.zero_grad(set_to_none=True)
                    pred = block(hb)[:, tc].unsqueeze(1)
                    loss = F.mse_loss(pred, yb)
                    loss.backward()
                    opt.step()
        except Exception as e:
            out.failures.append(_failure("train", "runtime", e))
            tb = traceback.format_exc(limit=8)
            out.warnings.append(f"train traceback (truncated):\n{tb}")
            return out

    block.eval()
    try:
        with torch.no_grad():
            out_tr = block(h_train)
            train_mse = F.mse_loss(out_tr[:, tc].unsqueeze(1), y_train).item()
            out_ev = block(h_eval)
            eval_mse = F.mse_loss(out_ev[:, tc].unsqueeze(1), y_eval).item()
    except Exception as e:
        out.failures.append(_failure("predict", "runtime", e))
        tb = traceback.format_exc(limit=8)
        out.warnings.append(f"eval traceback (truncated):\n{tb}")
        return out

    out.metrics["train_mse"] = float(train_mse)
    out.metrics["eval_mse"] = float(eval_mse)
    out.compile_stage_reached = "train"

    lim = max(0, int(predictions_sample_limit))
    if lim:
        preds: List[Dict[str, Any]] = []
        for i in range(min(lim, h_eval.shape[0])):
            preds.append(_abi_outputs_from_trunk_row(out_ev[i], abi, aw))
        out.predictions_sample = preds

    if include_trace_snippet and eval_rows:
        row0 = dict(eval_rows[0])
        try:
            dim_e = dim
            x = torch.zeros(1, dim_e, dtype=torch.float32)
            for name, col in abi.items():
                if name not in row0 or col >= dim_e:
                    continue
                w = max(1, int(aw.get(name, 1)))
                end = min(col + w, dim_e)
                val = row0[name]
                if isinstance(val, (list, tuple)):
                    for j in range(end - col):
                        x[0, col + j] = float(val[j]) if j < len(val) else 0.0
                else:
                    x[0, col:end] = float(val)
            tcol = abi[target_var]
            tw = max(1, int(aw.get(target_var, 1)))
            x[0, tcol : tcol + tw] = 0.0
            with torch.no_grad():
                _ot, env = block(x, return_env=True)
            trace: Dict[str, Any] = {}
            for k, v in env.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, torch.Tensor):
                    trace[k] = _env_tensor_to_python(v)
            ex = getattr(block, "_last_expert_trace", None)
            if ex:
                trace["expert_calls"] = list(ex)
            out.trace_snippet = trace
        except Exception as e:
            out.warnings.append(f"explain(first_eval_row) skipped: {e}")

    if score_fn is not None:
        if expected_rows is None:
            out.warnings.append("score_fn provided but expected_rows is missing; skipping score_fn metrics.")
        else:
            exp_list = [dict(r) for r in expected_rows]
            preds_full: List[Dict[str, Any]] = []
            for i in range(h_eval.shape[0]):
                preds_full.append(_abi_outputs_from_trunk_row(out_ev[i], abi, aw))
            if len(exp_list) != len(preds_full):
                out.failures.append(
                    ProgramFailure(
                        stage="train",
                        kind="value",
                        message=(
                            f"expected_rows length {len(exp_list)} != eval_rows length {len(preds_full)}."
                        ),
                        detail="ValueError",
                    )
                )
                return out
            try:
                raw = score_fn(preds_full, exp_list)
                out.metrics.update({str(k): float(v) for k, v in raw.items()})
            except Exception as e:
                out.failures.append(_failure("train", "metric", e))

    if not out.failures:
        out.program_metrics = [ProgramMetric(name=k, value=float(v)) for k, v in sorted(out.metrics.items())]

    return out
