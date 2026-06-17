from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi  # noqa: E402
from axiom.compiler.parser import parse_ax, reset_parser  # noqa: E402
from axiom.engine.block_executor import InterpretedBlock  # noqa: E402
from axiom.engine.trainer import probe_compile_diagnostics  # noqa: E402
from axiom.compiler.flow import wire_execution_graph  # noqa: E402
from axiom.engine.supernet import LatentSupernet  # noqa: E402


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    k = (len(ordered) - 1) * q
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return float(ordered[f])
    return float(ordered[f] + (ordered[c] - ordered[f]) * (k - f))


def _bench(name: str, fn: Callable[[], None], *, repeats: int, warmup: int) -> Dict[str, Any]:
    for _ in range(max(0, warmup)):
        fn()
    samples: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return {
        "name": name,
        "repeats": repeats,
        "p50_seconds": round(_percentile(samples, 0.5), 6),
        "p95_seconds": round(_percentile(samples, 0.95), 6),
        "mean_seconds": round(float(statistics.fmean(samples)), 6),
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile symbolic→neural execution overhead (JSON report).")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json-out", default="", help="Optional path for JSON output.")
    args = parser.parse_args(argv)

    reset_parser()
    parse_src = "y = x * 2.0;"
    parse_tree = parse_ax(parse_src)
    parse_ir = ast_to_ir(parse_tree)

    loop_src = """
i = 0;
while (i < 3) {
  i = i + 1;
}
"""
    loop_ir = ast_to_ir(parse_ax(loop_src))
    cond_src = """
if (x > 0) {
  y = x + 1.0;
} else {
  y = x - 1.0;
}
"""
    cond_ir = ast_to_ir(parse_ax(cond_src))

    dim = 16
    abi = extract_global_abi(parse_ir, max_vars=dim)
    aw = extract_abi_widths(parse_ir, max_vars=dim)
    block = InterpretedBlock(parse_ir, abi, abi_widths=aw)
    loop_abi = extract_global_abi(loop_ir, max_vars=dim)
    loop_aw = extract_abi_widths(loop_ir, max_vars=dim)
    loop_block = InterpretedBlock(loop_ir, loop_abi, abi_widths=loop_aw, max_unroll=8)
    sn = LatentSupernet(dim, ("then_0", "else_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    cond_graph = wire_execution_graph(cond_ir, sn, [("then_0", "else_0")])

    h = torch.zeros(8, dim)
    h[:, abi.get("x", 0)] = 2.0

    results: List[Dict[str, Any]] = []
    results.append(_bench("parse_to_ir", lambda: ast_to_ir(parse_ax(parse_src)), repeats=args.repeats, warmup=args.warmup))
    results.append(
        _bench(
            "interpreted_block_forward",
            lambda: block(h),
            repeats=args.repeats,
            warmup=args.warmup,
        )
    )
    results.append(
        _bench(
            "loop_block_forward",
            lambda: loop_block(h),
            repeats=args.repeats,
            warmup=args.warmup,
        )
    )
    with torch.no_grad():
        results.append(
            _bench(
                "conditional_graph_forward",
                lambda: cond_graph(h),
                repeats=args.repeats,
                warmup=args.warmup,
            )
        )
    compile_diag = probe_compile_diagnostics(cond_graph)
    results.append(
        {
            "name": "torch_compile_backend",
            "backend": str(compile_diag["backend"]),
            "errors": dict(compile_diag["errors"]),
        }
    )

    payload = {
        "schema_version": 1,
        "benchmarks": results,
        "config": {"repeats": args.repeats, "warmup": args.warmup},
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
