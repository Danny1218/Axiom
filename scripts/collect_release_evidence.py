"""Collect benchmark JSON artifacts into a release-evidence manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from axiom.copilot.benchmarks import benchmark_gate_violations


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_release_evidence(paths: List[Path]) -> Dict[str, Any]:
    suites: List[Dict[str, Any]] = []
    violations: List[str] = []
    for path in paths:
        if not path.is_file():
            violations.append(f"missing artifact: {path}")
            continue
        doc = _load_json(path)
        suites.append(
            {
                "path": str(path),
                "draft_summary": doc.get("draft_summary"),
                "search_summary": doc.get("search_summary"),
            }
        )
        violations.extend(f"{path.name}: {v}" for v in benchmark_gate_violations(doc))
    return {
        "kind": "axiom.release_evidence",
        "schema_version": 1,
        "suite_count": len(suites),
        "suites": suites,
        "gate_violations": violations,
        "gate_ok": len(violations) == 0,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate copilot benchmark JSON into release evidence.")
    ap.add_argument("artifacts", nargs="+", type=Path, help="benchmark_suite_to_dict JSON files")
    ap.add_argument("--out", type=Path, required=True, help="Write release evidence JSON here")
    args = ap.parse_args(argv)
    doc = collect_release_evidence(list(args.artifacts))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"Wrote release evidence to {args.out} (gate_ok={doc['gate_ok']})")
    return 0 if doc["gate_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
