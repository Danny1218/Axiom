"""Train Titanic hybrid + sklearn baselines; audit hard-rule violations."""

from __future__ import annotations

import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from axiom.api import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.copilot.train_tabular import run_train_tabular
from axiom.datasets import load_titanic, train_val_split
from axiom.engine.block_executor import InterpretedBlock
from benchmarks.titanic_hybrid.harness import (
    EPOCHS,
    FEATURES,
    GLOBAL_SEED,
    RULE_NOTE,
    ModelReport,
    accuracy,
    render_markdown,
    rule_applies,
    synthetic_edge_cases,
    violates_rule,
    write_json,
)

AX_PATH = ROOT / "examples" / "titanic_hybrid.ax"
CSV_PATH = ROOT / "examples" / "titanic.csv"
EVIDENCE_JSON = ROOT / "docs" / "evidence" / "titanic_hybrid.json"
EVIDENCE_MD = ROOT / "docs" / "evidence" / "titanic_hybrid.md"


def _compile_block() -> InterpretedBlock:
    reset_parser()
    ir = ast_to_ir(parse_ax_file(AX_PATH))
    abi = extract_global_abi(ir, max_vars=32)
    aw = extract_abi_widths(ir, max_vars=32)
    block = InterpretedBlock(ir, abi, abi_widths=aw, max_unroll=8)
    block.eval()
    return block


def _merge_rows(rows: List[dict]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for r in rows:
        row = {k: float(v) for k, v in r.items()}
        row["survived_prob"] = float(r["Survived"])
        out.append(row)
    return out


def _train_axiom(
    train_rows: List[dict], test_rows: List[dict]
) -> Tuple[AxiomModel, float, float]:
    block = _compile_block()
    merged_train = _merge_rows(train_rows)
    merged_test = _merge_rows(test_rows)
    t0 = time.perf_counter()
    tout = run_train_tabular(
        block,
        train_rows=merged_train,
        eval_rows=merged_test,
        target_var="survived_prob",
        epochs=EPOCHS,
        learning_rate=5e-2,
        weight_decay=1e-4,
        batch_size=32,
        predictions_sample_limit=0,
        include_trace_snippet=False,
    )
    train_s = time.perf_counter() - t0
    if tout.failures:
        raise RuntimeError(f"Axiom train failed: {tout.failures}")
    model = AxiomModel(block)
    preds = model.predict([{k: r[k] for k in FEATURES} for r in test_rows])
    probs = [float(p["survived_prob"]) for p in preds]
    return model, accuracy(test_rows, probs), train_s


def _sklearn_xy(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(r[k]) for k in FEATURES] for r in rows], dtype=np.float64)
    y = np.array([float(r["Survived"]) for r in rows], dtype=np.float64)
    return x, y


def _train_sklearn(
    factory, train_rows: List[dict], test_rows: List[dict]
) -> Tuple[object, float, float]:
    x_train, y_train = _sklearn_xy(train_rows)
    x_test, _ = _sklearn_xy(test_rows)
    t0 = time.perf_counter()
    model = factory()
    model.fit(x_train, y_train)
    train_s = time.perf_counter() - t0
    probs = model.predict_proba(x_test)[:, 1]
    return model, accuracy(test_rows, probs.tolist()), train_s


def _count_violations_axiom(model: AxiomModel, rows: List[dict]) -> int:
    inputs = [{k: float(r[k]) for k in FEATURES} for r in rows]
    preds = model.predict(inputs)
    if not isinstance(preds, list):
        preds = [preds]
    return sum(
        1
        for row, p in zip(rows, preds)
        if violates_rule(row, float(p.get("survived_prob", 0.0)))
    )


def _count_violations_sklearn(model: object, rows: List[dict]) -> int:
    x, _ = _sklearn_xy(rows)
    probs = model.predict_proba(x)[:, 1]
    return sum(1 for row, p in zip(rows, probs) if violates_rule(row, float(p)))


def run_audit() -> dict:
    torch.manual_seed(GLOBAL_SEED)
    rows = load_titanic(csv_path=CSV_PATH)
    train_rows, test_rows = train_val_split(rows, frac=0.8, seed=GLOBAL_SEED)
    edge = synthetic_edge_cases()

    model, ax_acc, ax_train_s = _train_axiom(train_rows, test_rows)
    lr_model, lr_acc, lr_train_s = _train_sklearn(
        lambda: LogisticRegression(max_iter=1000, random_state=GLOBAL_SEED),
        train_rows,
        test_rows,
    )
    gbr_model, gbr_acc, gbr_train_s = _train_sklearn(
        lambda: GradientBoostingClassifier(random_state=GLOBAL_SEED),
        train_rows,
        test_rows,
    )

    ax_v = _count_violations_axiom(model, edge)
    lr_v = _count_violations_sklearn(lr_model, edge)
    gbr_v = _count_violations_sklearn(gbr_model, edge)

    sample = next(r for r in edge if rule_applies(r))
    explain_row = model.explain({k: float(sample[k]) for k in FEATURES})

    best_baseline = max(lr_acc, gbr_acc)
    gap = abs(ax_acc - best_baseline)
    success = ax_v == 0 and gap <= 0.03 and (lr_v > 0 or gbr_v > 0)
    narrative = (
        f"Axiom holdout accuracy {ax_acc:.3f} vs best baseline {best_baseline:.3f} "
        f"(gap {gap:.3f}, bar <=0.03). Constraint violations on edge cases: "
        f"Axiom {ax_v}, LogisticRegression {lr_v}, GradientBoosting {gbr_v}. "
        "InterpretedBlock inference enforces symbolic clamps by construction; "
        "ExecutionGraph Sinkhorn training path does not (documented honestly). "
    )
    if success:
        narrative += "Success bar met: accuracy within 3 points and zero rule violations for Axiom."
    else:
        narrative += "Success bar not fully met — see tables; results recorded honestly."

    reports = [
        ModelReport("axiom_hybrid", ax_acc, ax_v, len(edge), ax_train_s),
        ModelReport("logistic_regression", lr_acc, lr_v, len(edge), lr_train_s),
        ModelReport("gradient_boosting", gbr_acc, gbr_v, len(edge), gbr_train_s),
    ]
    summary = {
        "version": "1.3.0",
        "benchmark": "titanic_hybrid",
        "seed": GLOBAL_SEED,
        "epochs": EPOCHS,
        "csv_path": str(CSV_PATH),
        "inference_path": "InterpretedBlock (symbolic IR + neural train_tabular)",
        "rule": RULE_NOTE,
        "success_bar_met": success,
        "narrative": narrative,
        "models": [asdict(m) for m in reports],
        "explain_sample": {"inputs": {k: sample[k] for k in FEATURES}, "trace": explain_row},
    }
    write_json(EVIDENCE_JSON, summary)
    EVIDENCE_MD.write_text(render_markdown(summary), encoding="utf-8")
    print(narrative)
    print(f"Wrote {EVIDENCE_JSON} and {EVIDENCE_MD}")
    return summary


def main() -> int:
    run_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
