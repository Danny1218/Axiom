"""Train GBM + expert-wrapped guarded .ax; audit constraints and emit certificate."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from axiom.api import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.datasets import load_titanic, train_val_split
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.expert_registry import ExpertRuntimeRegistry
from axiom.verify.interval import certify
from benchmarks.titanic_hybrid.harness import (
    FEATURES,
    GLOBAL_SEED,
    RULE_NOTE,
    ModelReport,
    accuracy,
    rule_applies,
    synthetic_edge_cases,
    violates_rule,
    write_json,
)

AX_PATH = ROOT / "examples" / "titanic_guarded.ax"
CSV_PATH = ROOT / "examples" / "titanic.csv"
HYBRID_EVIDENCE = ROOT / "docs" / "evidence" / "titanic_hybrid.json"
EVIDENCE_JSON = ROOT / "docs" / "evidence" / "titanic_guarded.json"
EVIDENCE_MD = ROOT / "docs" / "evidence" / "titanic_guarded.md"
CERT_PATH = ROOT / "docs" / "evidence" / "titanic_guarded_certificate.json"
BUNDLE_PATH = ROOT / "docs" / "evidence" / "titanic_guarded.axb"


def _compile_block(registry: ExpertRuntimeRegistry) -> InterpretedBlock:
    reset_parser()
    ir = ast_to_ir(parse_ax_file(AX_PATH))
    abi = extract_global_abi(ir, max_vars=32)
    aw = extract_abi_widths(ir, max_vars=32)
    block = InterpretedBlock(
        ir, abi, abi_widths=aw, max_unroll=8, expert_registry=registry
    )
    block.eval()
    return block


def _sklearn_xy(rows: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(r[k]) for k in FEATURES] for r in rows], dtype=np.float64)
    y = np.array([float(r["Survived"]) for r in rows], dtype=np.float64)
    return x, y


def _train_gbm(train_rows: List[dict]) -> GradientBoostingClassifier:
    x_train, y_train = _sklearn_xy(train_rows)
    model = GradientBoostingClassifier(random_state=GLOBAL_SEED)
    model.fit(x_train, y_train)
    return model


def _gbm_handler(model: GradientBoostingClassifier):
    def handler(_name: str, features: List[float]) -> float:
        row = np.array([features], dtype=np.float64)
        return float(model.predict_proba(row)[0, 1])

    return handler


def _predict_guarded(
    model: AxiomModel, rows: List[dict]
) -> Tuple[List[float], List[float]]:
    inputs = [{k: float(r[k]) for k in FEATURES} for r in rows]
    preds = model.predict(inputs)
    if not isinstance(preds, list):
        preds = [preds]
    raw_probs: List[float] = []
    final_probs: List[float] = []
    for row in rows:
        expl = model.explain({k: float(row[k]) for k in FEATURES})
        raw_probs.append(float(expl.get("raw_prob", expl.get("survived_prob", 0.0))))
        final_probs.append(float(expl.get("survived_prob", 0.0)))
    return raw_probs, final_probs


def _count_violations(probs: List[float], rows: List[dict]) -> int:
    return sum(
        1 for row, p in zip(rows, probs) if violates_rule(row, float(p))
    )


def _load_v13_hybrid_accuracy() -> Optional[float]:
    if not HYBRID_EVIDENCE.is_file():
        return None
    payload = json.loads(HYBRID_EVIDENCE.read_text(encoding="utf-8"))
    for m in payload.get("models", []):
        if m.get("name") == "axiom_hybrid":
            return float(m["holdout_accuracy"])
    return None


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_certificate(block: InterpretedBlock) -> Dict[str, Any]:
    input_bounds = {
        "Pclass": (3.0, 3.0),
        "Sex": (0.0, 0.0),
        "Age": (18.0, 100.0),
        "Fare": (0.0, 600.0),
    }
    node_bounds = {"tabular_model": (0.0, 1.0)}
    cert = certify(block, input_bounds, node_bounds=node_bounds, source_path=AX_PATH)
    payload = cert.to_dict()
    write_json(CERT_PATH, payload)
    return payload


def run_guarded_audit() -> dict:
    torch.manual_seed(GLOBAL_SEED)
    rows = load_titanic(csv_path=CSV_PATH)
    train_rows, test_rows = train_val_split(rows, frac=0.8, seed=GLOBAL_SEED)
    edge = synthetic_edge_cases()

    t0 = time.perf_counter()
    gbm = _train_gbm(train_rows)
    train_s = time.perf_counter() - t0

    x_test, _ = _sklearn_xy(test_rows)
    raw_test_probs = gbm.predict_proba(x_test)[:, 1].tolist()
    raw_acc = accuracy(test_rows, raw_test_probs)

    reg = ExpertRuntimeRegistry()
    reg.register("tabular_model", _gbm_handler(gbm))
    block = _compile_block(reg)
    save_bundle(block, BUNDLE_PATH)
    model = AxiomModel(block)

    _, guarded_test_probs = _predict_guarded(model, test_rows)
    guarded_acc = accuracy(test_rows, guarded_test_probs)

    x_edge, _ = _sklearn_xy(edge)
    raw_edge_probs = gbm.predict_proba(x_edge)[:, 1].tolist()
    raw_v = _count_violations(raw_edge_probs, edge)
    _, guarded_edge_probs = _predict_guarded(model, edge)
    guarded_v = _count_violations(guarded_edge_probs, edge)

    v13_acc = _load_v13_hybrid_accuracy()
    sample = next(r for r in edge if rule_applies(r))
    explain_row = model.explain({k: float(sample[k]) for k in FEATURES})

    cert = _write_certificate(block)
    survived_bounds = cert.get("proven_output_bounds", {}).get("survived_prob", (999.0, 999.0))
    cert_ok = float(survived_bounds[1]) <= 0.15 + 1e-9

    acc_delta = abs(raw_acc - guarded_acc)
    v13_str = f"{v13_acc:.3f}" if v13_acc is not None else "n/a"
    narrative = (
        f"Raw GBM holdout accuracy {raw_acc:.3f}; guarded wrap {guarded_acc:.3f} "
        f"(delta {acc_delta:.4f}). v1.3 pure-hybrid accuracy {v13_str}. "
        f"Constraint violations on {len(edge)} edge cases: raw GBM {raw_v}, "
        f"guarded {guarded_v}. Certificate proves survived_prob.hi="
        f"{survived_bounds[1] if isinstance(survived_bounds, (list, tuple)) else survived_bounds} "
        f"on rule region (ok={cert_ok})."
    )

    reports = [
        ModelReport(
            "raw_gradient_boosting",
            raw_acc,
            raw_v,
            len(edge),
            train_s,
            note="sklearn GradientBoostingClassifier predict_proba",
        ),
        ModelReport(
            "guarded_gbm_axiom_wrap",
            guarded_acc,
            guarded_v,
            len(edge),
            0.0,
            note="expert(tabular_model) + symbolic min clamp via InterpretedBlock",
        ),
    ]
    if v13_acc is not None:
        reports.append(
            ModelReport(
                "v1.3_axiom_hybrid",
                v13_acc,
                0,
                len(edge),
                0.0,
                note="reused from docs/evidence/titanic_hybrid.json",
            )
        )

    summary: Dict[str, Any] = {
        "version": "1.4.0",
        "benchmark": "titanic_guarded",
        "seed": GLOBAL_SEED,
        "csv_path": str(CSV_PATH),
        "ax_source": str(AX_PATH),
        "bundle_path": str(BUNDLE_PATH),
        "source_hash": _source_hash(AX_PATH),
        "rule": RULE_NOTE,
        "accuracy_delta_raw_vs_guarded": acc_delta,
        "v1_3_hybrid_accuracy": v13_acc,
        "certificate_path": str(CERT_PATH),
        "certificate_ok": cert_ok,
        "narrative": narrative,
        "models": [asdict(m) for m in reports],
        "explain_sample": {
            "inputs": {k: sample[k] for k in FEATURES},
            "trace": explain_row,
            "note": "raw_prob from expert(); survived_prob after symbolic clamp",
        },
        "certificate_summary": {
            "input_region": cert.get("input_region"),
            "proven_output_bounds": cert.get("proven_output_bounds"),
        },
    }
    write_json(EVIDENCE_JSON, summary)
    EVIDENCE_MD.write_text(_render_guarded_md(summary), encoding="utf-8")
    print(narrative)
    print(f"Wrote {EVIDENCE_JSON}, {EVIDENCE_MD}, {CERT_PATH}")
    return summary


def _render_guarded_md(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Titanic guarded — wrap strong model + certify clamp",
        "",
        f"Generated by `benchmarks/titanic_hybrid/run_guarded_audit.py` (seed={summary.get('seed')}).",
        "",
        "## Rule under audit",
        "",
        summary.get("rule", RULE_NOTE),
        "",
        "## Holdout accuracy (80/20 split, seed=42)",
        "",
        "| Model | Accuracy | Note |",
        "|-------|----------|------|",
    ]
    for m in summary.get("models", []):
        note = m.get("note") or ""
        lines.append(f"| {m['name']} | {m['holdout_accuracy']:.4f} | {note} |")
    lines.extend(
        [
            "",
            f"Accuracy delta (raw vs guarded): **{summary.get('accuracy_delta_raw_vs_guarded', 0):.4f}**",
            "",
            "## Constraint audit (500 synthetic edge cases)",
            "",
            "| Model | Violations |",
            "|-------|------------|",
        ]
    )
    for m in summary.get("models", []):
        if m["name"] in ("raw_gradient_boosting", "guarded_gbm_axiom_wrap"):
            lines.append(f"| {m['name']} | {m['constraint_violations']} |")
    lines.extend(
        [
            "",
            "## Safety certificate",
            "",
            f"Certificate OK: **{summary.get('certificate_ok')}** — see `{summary.get('certificate_path')}`.",
            "",
            "```json",
            json.dumps(summary.get("certificate_summary", {}), indent=2),
            "```",
            "",
            "## Narrative",
            "",
            summary.get("narrative", ""),
            "",
            "## Example explain trace (raw_prob vs clamped)",
            "",
            "```json",
            json.dumps(summary.get("explain_sample", {}), indent=2),
            "```",
            "",
            "## Reproduce",
            "",
            "```powershell",
            "pip install -e \".[bench]\"",
            "python benchmarks/titanic_hybrid/run_guarded_audit.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    run_guarded_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
