"""Axiom CLI entrypoint.

Subcommands are the stable user surface (train, predict, bundle I/O, optional HTTP, semantic copilot).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from axiom.compiler.deserializer import load_bundle, load_execution_bundle
from axiom.compiler.flow import wire_execution_graph
from axiom.compiler.ir import ast_to_ir
from axiom.compiler.parser import parse_ax_file
from axiom.compiler.serializer import save_execution_bundle
from axiom.datasets import generate_sine_wave, load_titanic, train_val_split
from axiom.engine.dataloader import AxiomDataset, LiquidSequenceLoader, load_csv_to_dicts
from axiom.engine.inference import (
    AxiomRunner,
    _abi_outputs_from_trunk_row,
    _inputs_to_tensor,
)
from axiom.engine.meta_compiler import MetaCompiler
from axiom.engine.supernet import LatentSupernet
from axiom.engine.topology import ExecutionGraph
from axiom.engine.trainer import EvolutionaryTrainer


def _compile_graph(
    ax_path: Path,
    dim: int,
    rank: int,
    *,
    loop_max_unroll: int = 8,
    loop_num_basis: int = 8,
    mutation_entropy_norm_threshold: float = 0.92,
) -> Tuple[list, LatentSupernet, ExecutionGraph]:
    ir = ast_to_ir(parse_ax_file(ax_path))
    n_cond = sum(1 for x in ir if x[0] == "OP_CONDITIONAL")
    pairs = [(f"then_{i}", f"else_{i}") for i in range(n_cond)]
    names: list[str] = [n for p in pairs for n in p]
    for j in range(max(0, 4 - len(names))):
        names.append(f"latent_{j}")
    if not names:
        names = ["latent_0", "latent_1"]
    sn = LatentSupernet(dim, names, rank=rank)
    for i in range(n_cond):
        sn.set_masks({f"then_{i}": 1.0, f"else_{i}": 1.0})
    g = wire_execution_graph(
        ir,
        sn,
        pairs,
        mutation_entropy_norm_threshold=mutation_entropy_norm_threshold,
        loop_max_unroll=loop_max_unroll,
        loop_num_basis=loop_num_basis,
    )
    return ir, sn, g


def _resolve_target_col(graph: ExecutionGraph, abi_name: str) -> int:
    c = graph.abi.get(abi_name)
    if c is None:
        raise SystemExit(
            f"ABI must include {abi_name!r} for this dataset/metric; got keys: {sorted(graph.abi.keys())}"
        )
    return c


def _cmd_train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "inference":
        prefix = args.out
        pt = Path(str(prefix) + ".pt")
        js = Path(str(prefix) + "_topology.json")
        if not pt.is_file() or not js.is_file():
            raise SystemExit(f"Inference requires saved bundle at {prefix} (.pt + _topology.json).")
        graph = load_execution_bundle(prefix).to(device)
        runner = AxiomRunner(graph)
        out_dict = runner.predict_dict({"x": 1.0}, device=device)
        print("out:", out_dict)
        return

    if args.csv is not None and args.dataset is not None:
        raise SystemExit("Use either --dataset or --csv, not both.")

    if args.csv is not None:
        if not args.target_key or not args.target_var:
            raise SystemExit("--csv requires --target_key (CSV label column) and --target_var (ABI output name).")
        rows = load_csv_to_dicts(args.csv)
        if not rows:
            raise SystemExit("CSV is empty.")
        train_rows, test_rows = train_val_split(rows, frac=args.split_frac, seed=args.seed)
        loop_unroll = args.loop_max_unroll if args.loop_max_unroll is not None else 8
        ir, sn, graph = _compile_graph(
            args.ax_path,
            args.dim,
            args.rank,
            loop_max_unroll=loop_unroll,
            loop_num_basis=args.loop_num_basis,
            mutation_entropy_norm_threshold=args.mutation_threshold,
        )
        graph = graph.to(device)
        target_col = _resolve_target_col(graph, args.target_var)
        meta = None if args.no_meta or sum(1 for x in ir if x[0] == "OP_CONDITIONAL") == 0 else MetaCompiler(sn)
        _train_tabular_and_eval(
            graph,
            ir,
            train_rows,
            test_rows,
            args,
            device,
            target_key=args.target_key,
            target_col=target_col,
            meta=meta,
            metric="mse",
            abi_var_for_metric=args.target_var,
        )
        return

    if args.dataset == "titanic":
        try:
            rows = load_titanic(csv_path=args.titanic_csv)
        except OSError as e:
            raise SystemExit(f"Could not obtain Titanic CSV: {e}") from e
        train_rows, test_rows = train_val_split(rows, frac=args.split_frac, seed=args.seed)
        ir, sn, graph = _compile_graph(
            args.ax_path,
            args.dim,
            args.rank,
            mutation_entropy_norm_threshold=args.mutation_threshold,
        )
        graph = graph.to(device)
        target_col = _resolve_target_col(graph, "survived_prob")
        meta = None if args.no_meta else MetaCompiler(sn)
        _train_tabular_and_eval(
            graph,
            ir,
            train_rows,
            test_rows,
            args,
            device,
            target_key="Survived",
            target_col=target_col,
            meta=meta,
            metric="accuracy",
            abi_var_for_metric="survived_prob",
        )
        return

    if args.dataset == "sine":
        rows = generate_sine_wave(n=args.sine_samples, seed=args.seed + 1)
        train_rows, test_rows = train_val_split(rows, frac=args.split_frac, seed=args.seed)
        loop_unroll = args.loop_max_unroll if args.loop_max_unroll is not None else 10
        ir, sn, graph = _compile_graph(
            args.ax_path,
            args.dim,
            args.rank,
            loop_max_unroll=loop_unroll,
            loop_num_basis=args.loop_num_basis,
            mutation_entropy_norm_threshold=args.mutation_threshold,
        )
        graph = graph.to(device)
        target_col = _resolve_target_col(graph, "y_pred")
        _train_tabular_and_eval(
            graph,
            ir,
            train_rows,
            test_rows,
            args,
            device,
            target_key="target",
            target_col=target_col,
            meta=None,
            metric="mse",
            abi_var_for_metric="y_pred",
        )
        return

    # Legacy: synthetic sequence loader (no tabular dataset)
    ir, sn, graph = _compile_graph(args.ax_path, args.dim, args.rank)
    graph = graph.to(device)
    seq = torch.cumsum(torch.randn(512, device=device) * 0.05, dim=0)
    loader = LiquidSequenceLoader(
        seq.cpu(),
        feature_dim=args.dim,
        batch_size=args.batch,
        baseline_var=args.baseline_var,
        device=device,
    )
    meta = MetaCompiler(sn)
    trainer = EvolutionaryTrainer(graph, lr=args.lr, shadow_fitness_epochs=5)
    for _ in range(args.epochs):
        trainer.train_epoch(loader, meta_compiler=meta)
    save_execution_bundle(graph.cpu(), args.out, ir=ir)
    print(f"Saved {args.out}.pt and {args.out}_topology.json")


def _train_tabular_and_eval(
    graph: ExecutionGraph,
    ir: list,
    train_rows: List[dict],
    test_rows: List[dict],
    args: argparse.Namespace,
    device: torch.device,
    *,
    target_key: str,
    target_col: int,
    meta: Optional[MetaCompiler],
    metric: str,
    abi_var_for_metric: str,
    trainer_lr: Optional[float] = None,
) -> None:
    abi = graph.abi
    abi_w = getattr(graph, "abi_widths", {}) or {}
    train_ds = AxiomDataset(
        train_rows, abi, trunk_dim=args.dim, target_key=target_key, abi_widths=abi_w
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    lr = float(args.lr) if trainer_lr is None else float(trainer_lr)
    trainer = EvolutionaryTrainer(
        graph, lr=lr, compile_graph=False, target_col=target_col, shadow_fitness_epochs=5
    )
    log_every = max(1, args.epochs // 10) if args.epochs >= 10 else 1
    for ep in range(args.epochs):
        loss = trainer.train_epoch(train_loader, meta_compiler=meta, device=device)
        if ep == 0 or (ep + 1) % log_every == 0 or ep + 1 == args.epochs:
            print(f"epoch {ep + 1}/{args.epochs}  mean_mse={loss:.6f}")

    graph.eval()
    runner = AxiomRunner(graph)
    preds = runner.predict_dict_batch(test_rows, device=device)

    if metric == "accuracy":
        correct = 0
        for row, pdict in zip(test_rows, preds):
            prob = float(pdict.get(abi_var_for_metric, 0.0))
            pred = 1.0 if prob > 0.5 else 0.0
            actual = float(row[target_key])
            if pred == actual:
                correct += 1
        acc = correct / max(len(test_rows), 1)
        print(f"test_accuracy={acc:.4f}  (n={len(test_rows)})")
    else:
        se = 0.0
        for row, pdict in zip(test_rows, preds):
            pred = float(pdict.get(abi_var_for_metric, 0.0))
            tgt = float(row[target_key])
            se += (pred - tgt) ** 2
        mse = se / max(len(test_rows), 1)
        print(f"test_mse={mse:.6f}  (n={len(test_rows)})")

    save_execution_bundle(graph.cpu(), args.out, ir=ir)
    print(f"Saved {args.out}.pt and {args.out}_topology.json")


def _trunk_dim_from_block_abi(block) -> int:
    abi, aw = block.abi, getattr(block, "abi_widths", {}) or {}
    return max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=16)


def _cmd_lock_bundle(args: argparse.Namespace) -> None:
    from axiom.security.genetic_lock import lock_bundle_file

    try:
        lock_bundle_file(Path(args.input), Path(args.output), args.mode)
    except ImportError as e:
        raise SystemExit('Bundle lock requires: pip install -e ".[lock]"') from e
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    print(f"Locked bundle written to {args.output}")


def _cmd_export_onnx(args: argparse.Namespace) -> None:
    try:
        from axiom.export.onnx_export import OnnxExportError, export_bundle_to_onnx
    except ImportError as e:
        raise SystemExit('ONNX export requires: pip install -e ".[export]"') from e
    try:
        export_bundle_to_onnx(
            Path(args.bundle),
            Path(args.output),
            opset_version=int(args.opset),
        )
    except OnnxExportError as e:
        raise SystemExit(str(e)) from e
    print(f"Wrote ONNX to {args.output}")


def _cmd_predict(args: argparse.Namespace) -> None:
    block = load_bundle(args.bundle)
    try:
        feats = json.loads(args.input)
    except json.JSONDecodeError as e:
        raise SystemExit(f"invalid --input JSON: {e}") from e
    if not isinstance(feats, dict):
        raise SystemExit("--input must be a JSON object")
    block.eval()
    dim = _trunk_dim_from_block_abi(block)
    dev = torch.device("cpu")
    dt = torch.float32
    aw = getattr(block, "abi_widths", {}) or {}
    h = _inputs_to_tensor(feats, block.abi, dim, device=dev, dtype=dt, abi_widths=aw)
    with torch.no_grad():
        out = block(h)
    decoded = _abi_outputs_from_trunk_row(out[0], block.abi, dict(aw))
    print(json.dumps(decoded, indent=2))


def _cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            "axiom serve requires optional deps: pip install -e \".[serve]\""
        ) from e
    from axiom.serve import create_app

    bundle = args.bundle or os.environ.get("AXIOM_BUNDLE_PATH")
    if not bundle:
        raise SystemExit("Provide --bundle or set AXIOM_BUNDLE_PATH to a .axb file.")
    bp = Path(bundle)
    if not bp.is_file():
        raise SystemExit(f"Bundle not found: {bp}")

    host_env = os.environ.get("HOST")
    host = host_env.strip() if host_env not in (None, "") else args.host
    port_env = os.environ.get("PORT")
    port = int(port_env) if port_env not in (None, "") else int(args.port)

    app = create_app(bp)
    uvicorn.run(app, host=host, port=port, log_level="info")


def _cmd_gateway_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            'axiom gateway-serve requires: pip install -e ".[gateway]"'
        ) from e
    from axiom.api import load
    from axiom.gateway.server import create_gateway_app

    bundle = Path(args.bundle)
    if not bundle.is_file():
        raise SystemExit(f"Bundle not found: {bundle}")
    policy_src = None
    if args.policy_source:
        ps = Path(args.policy_source)
        if not ps.is_file():
            raise SystemExit(f"Policy source not found: {ps}")
        policy_src = ps.read_text(encoding="utf-8")
    model = load(bundle)
    app = create_gateway_app(
        model,
        policy_src,
        downstream_url=str(args.downstream_url),
        approve_threshold=float(args.approve_threshold),
        audit_path_on_block=args.audit_path,
    )
    host_env = os.environ.get("HOST")
    host = host_env.strip() if host_env not in (None, "") else args.host
    port_env = os.environ.get("PORT")
    port = int(port_env) if port_env not in (None, "") else int(args.port)
    uvicorn.run(app, host=host, port=port, log_level="info")


def _cmd_inspect(_args: argparse.Namespace) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit('Glass Box requires: pip install -e ".[inspect]"') from None

    import axiom.tools

    inspector = Path(axiom.tools.__file__).resolve().parent / "inspector.py"
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(inspector),
            "--server.fileWatcherType",
            "none",
        ]
    )


def _cmd_copilot_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            'axiom copilot-serve requires FastAPI/uvicorn: pip install -e ".[serve]"'
        ) from e
    _require_requests_for_copilot()
    from axiom.copilot.backend import build_copilot_expert
    from axiom.copilot.server import create_app

    url = (args.expert_url or "").strip()
    model = (args.expert_model or "").strip()
    if not url:
        raise SystemExit("--expert-url is required.")
    if not model:
        raise SystemExit("--expert-model is required.")
    key = args.expert_api_key
    if key is None or str(key).strip() == "":
        key = os.environ.get("AXIOM_EXPERT_API_KEY")
    try:
        expert = build_copilot_expert(
            args.backend, expert_url=url, expert_model=model, expert_api_key=key
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    app = create_app(expert)
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")


def _cmd_copilot_studio(_args: argparse.Namespace) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        raise SystemExit('Copilot Studio requires Streamlit: pip install -e ".[inspect]"') from None
    try:
        import requests  # noqa: F401
    except ImportError:
        raise SystemExit('Copilot Studio requires requests: pip install -e ".[copilot]"') from None

    import axiom.tools

    studio = Path(axiom.tools.__file__).resolve().parent / "copilot_studio.py"
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(studio),
            "--server.fileWatcherType",
            "none",
        ]
    )


_COPILOT_INSTALL = 'pip install -e ".[copilot]"'


def _require_requests_for_copilot() -> None:
    try:
        import requests  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"Semantic copilot commands require the [copilot] extra ({_COPILOT_INSTALL})."
        ) from e


def _make_copilot_expert(args: argparse.Namespace):
    """Return a :class:`~axiom.experts.base.SemanticExpert` from CLI flags (no hardcoded endpoints)."""
    from axiom.copilot.backend import build_copilot_expert

    backend = str(args.backend).strip().lower().replace("_", "-")
    url = (args.expert_url or "").strip()
    model = (args.expert_model or "").strip()
    if backend == "onyx-qwen":
        _require_requests_for_copilot()
        if not url:
            raise SystemExit("--expert-url is required for onyx-qwen.")
        if not model:
            raise SystemExit("--expert-model is required for onyx-qwen.")
    elif backend != "benchmark-dispatch":
        raise SystemExit(f"Unsupported --backend {args.backend!r} (expected onyx-qwen or benchmark-dispatch).")
    key = args.expert_api_key
    if key is None or str(key).strip() == "":
        key = os.environ.get("AXIOM_EXPERT_API_KEY")
    tout = getattr(args, "expert_timeout", None)
    if tout is not None:
        tout = float(tout)
    try:
        return build_copilot_expert(
            backend,
            expert_url=url,
            expert_model=model,
            expert_api_key=key,
            timeout=tout,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e


def _load_examples_json(path: Path) -> Tuple[List[dict], List[dict]]:
    """Load row-based eval examples.

    Format: JSON array of objects, each with ``inputs`` and ``expected`` dicts::

        [{"inputs": {"x": 1.0}, "expected": {"y": 0.5}}, ...]
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise SystemExit(f"Cannot read --examples-json: {e}") from e
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in --examples-json: {e}") from e
    if not isinstance(raw, list):
        raise SystemExit("--examples-json must be a JSON array.")
    inputs: List[dict] = []
    expected: List[dict] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise SystemExit(f"--examples-json[{i}] must be an object.")
        if "inputs" not in row or "expected" not in row:
            raise SystemExit(f"--examples-json[{i}] must have \"inputs\" and \"expected\" keys.")
        ins, exp = row["inputs"], row["expected"]
        if not isinstance(ins, dict) or not isinstance(exp, dict):
            raise SystemExit(f"--examples-json[{i}]: inputs and expected must be objects.")
        inputs.append(dict(ins))
        expected.append(dict(exp))
    if not inputs:
        raise SystemExit("--examples-json array is empty.")
    return inputs, expected


def _load_tabular_json(path: Path) -> Any:
    """Load tabular train/eval JSON for ``--train-tabular`` (see :mod:`axiom.copilot.tabular_json`)."""
    from axiom.copilot.tabular_json import parse_tabular_json_text

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"Cannot read --tabular-json file: {e}") from e
    try:
        return parse_tabular_json_text(text)
    except ValueError as e:
        raise SystemExit(f"Invalid tabular JSON: {e}") from e


def _default_predict_score_fn() -> Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, float]]:
    """Higher ``neg_mse`` is better (matches copilot search ranking)."""

    def score_fn(preds: List[Dict[str, Any]], exp: List[Dict[str, Any]]) -> Dict[str, float]:
        total = 0.0
        n = 0
        for p, e in zip(preds, exp):
            for k, ev in e.items():
                if k not in p:
                    continue
                total += (float(p[k]) - float(ev)) ** 2
                n += 1
        mse = total / max(n, 1)
        return {"neg_mse": float(-mse)}

    return score_fn


def _serialize_evaluation_report(rep: Any) -> dict:
    from axiom.copilot.artifacts import evaluation_report_to_dict

    return evaluation_report_to_dict(rep)


def _completion_overrides_from_args(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}
    if getattr(args, "temperature", None) is not None:
        out["temperature"] = float(args.temperature)
    if getattr(args, "top_p", None) is not None:
        out["top_p"] = float(args.top_p)
    return out if out else None


def _build_copilot_search_config(args: argparse.Namespace, expert) -> Any:
    """Shared by ``copilot-search`` and ``copilot-run`` (same evaluation modes and artifacts)."""
    from axiom.copilot.search import CopilotSearchConfig

    train_tab = bool(getattr(args, "train_tabular", False))
    tab_path = getattr(args, "tabular_json", None)
    if train_tab and args.compile_only:
        raise SystemExit("Cannot use --train-tabular with --compile-only.")
    if train_tab and args.examples_json is not None:
        raise SystemExit("Cannot use --train-tabular together with --examples-json.")
    if train_tab and tab_path is None:
        raise SystemExit("--train-tabular requires --tabular-json PATH.")
    if tab_path is not None and not train_tab:
        raise SystemExit("--tabular-json requires --train-tabular.")

    example_in: Optional[List[dict]] = None
    example_exp: Optional[List[dict]] = None
    if args.examples_json is not None:
        example_in, example_exp = _load_examples_json(args.examples_json)

    tab_train: Optional[List[dict]] = None
    tab_eval: Optional[List[dict]] = None
    tab_target: Optional[str] = None
    tab_params = None
    tab_eval_exp: Optional[List[dict]] = None
    if train_tab:
        pld = _load_tabular_json(tab_path)
        tab_train = list(pld.train_rows)
        tab_eval = list(pld.eval_rows)
        tab_target = pld.target_var
        tab_params = pld.params
        tab_eval_exp = list(pld.eval_expected_rows)

    if getattr(args, "repair_valid_with_metrics", False) and getattr(args, "no_repair_valid_with_metrics", False):
        raise SystemExit("Cannot combine --repair-valid-with-metrics with --no-repair-valid-with-metrics.")

    if args.compile_only:
        mode: str = "compile_only"
        score_fn = None
        sort_key = None
        repair_valid = False
    elif train_tab:
        mode = "train_tabular"
        score_fn = _default_predict_score_fn()
        sort_key = "neg_mse"
        repair_valid = not bool(getattr(args, "no_repair_valid_with_metrics", False))
    elif example_in is not None:
        mode = "predict_rows"
        score_fn = _default_predict_score_fn()
        sort_key = "neg_mse"
        repair_valid = not bool(getattr(args, "no_repair_valid_with_metrics", False))
    else:
        mode = "compile_only"
        score_fn = None
        sort_key = None
        repair_valid = False

    metric_below = getattr(args, "metric_repair_if_below", None)

    summarize = bool(getattr(args, "summarize_traces", False))
    return CopilotSearchConfig(
        expert=expert,
        goal=args.goal,
        domain_context=args.context,
        example_input_rows=example_in,
        expected_rows=example_exp,
        max_iterations=max(1, int(args.iterations)),
        mode=mode,  # type: ignore[arg-type]
        score_fn=score_fn,
        score_sort_key=sort_key,
        include_trace_snippet=summarize,
        summarize_traces=summarize,
        artifact_dir=args.artifact_dir,
        tabular_train_rows=tab_train,
        tabular_eval_rows=tab_eval,
        tabular_target_var=tab_target,
        tabular_train_params=tab_params,
        tabular_eval_expected_rows=tab_eval_exp,
        repair_valid_with_metrics=repair_valid,
        metric_repair_if_below=float(metric_below) if metric_below is not None else None,
        completion_overrides=_completion_overrides_from_args(args),
    )


_COPILOT_DOCTOR_DEFAULT_GOAL = (
    "Write a valid Axiom .ax program in this repo's DSL that computes y = x * 2.0;"
)


def _cmd_copilot_doctor(args: argparse.Namespace) -> None:
    from axiom.copilot.evaluator import evaluate_program, validate_program
    from axiom.copilot.models import ProgramCandidate
    from axiom.copilot.search import build_draft_context
    from axiom.copilot.stability_report import DEFAULT_NEAR_NEG_MSE
    from axiom.experts.base import ExpertDraftRequest
    from axiom.experts.onyx_qwen import COMPLETION_OVERRIDES_CONTEXT_KEY, OnyxQwenError, ax_source_metadata_flags

    expert = _make_copilot_expert(args)
    ctx = dict(
        build_draft_context(
            domain_context=None,
            example_input_rows=None,
            expected_rows=None,
        )
    )
    co: Dict[str, Any] = {}
    if getattr(args, "temperature", None) is not None:
        co["temperature"] = float(args.temperature)
    else:
        co["temperature"] = 0.0
    if getattr(args, "top_p", None) is not None:
        co["top_p"] = float(args.top_p)
    ctx[COMPLETION_OVERRIDES_CONTEXT_KEY] = co

    try:
        resp = expert.draft_program(ExpertDraftRequest(goal=args.goal, context=ctx))
    except OnyxQwenError as e:
        print(f"connection: fail ({type(e).__name__}: {e})", file=sys.stderr)
        raise SystemExit(1) from e

    meta = dict(resp.metadata)
    raw_len = int(meta.get("raw_chars", 0))
    ax = resp.ax_source
    preview = ax.replace("\n", "\\n")
    if len(preview) > 100:
        preview = preview[:97] + "..."

    print("connection: ok")
    print(f"raw_chars: {raw_len}")
    print(f"ax_source: {len(ax)} chars, preview: {preview}")

    rep = validate_program(ProgramCandidate(source=ax))
    parse_ok = not any(f.stage == "parse" for f in rep.failures)
    ir_ok = not any(f.stage == "ir" for f in rep.failures)
    block_ok = rep.success
    print(f"parse: {'ok' if parse_ok else 'fail'}")
    print(f"ir: {'ok' if ir_ok else 'fail'}")
    print(f"block: {'ok' if block_ok else 'fail'}")

    warn_keys = (
        "assign_colon_eq",
        "print_call",
        "indexed_variable_warning",
        "output_call_warning",
        "suspicious_numeric_literal_warning",
    )
    forbidden = list(meta.get("forbidden_tokens_detected") or [])
    active: list[str] = []
    if "assign_colon_eq" in forbidden:
        active.append("assign_colon_eq")
    if "print_call" in forbidden:
        active.append("print_call")
    for k in ("indexed_variable_warning", "output_call_warning", "suspicious_numeric_literal_warning"):
        if meta.get(k):
            active.append(k)
    if active:
        print("anti_pattern: " + ", ".join(active))
    else:
        print("anti_pattern: (none)")

    flags = ax_source_metadata_flags(ax)
    print(f"neural: {'yes' if flags.get('uses_neural') else 'no'}")

    if not rep.success:
        raise SystemExit(1)

    ex_path = getattr(args, "examples_json", None)
    if ex_path is not None:
        inp, exp = _load_examples_json(Path(ex_path))
        ev = evaluate_program(
            ProgramCandidate(source=ax),
            mode="predict_rows",
            input_rows=inp,
            expected_rows=exp,
            score_fn=_default_predict_score_fn(),
            row_comparison_limit=0,
            include_trace_snippet=False,
        )
        print(f"evaluation: {'ok' if ev.success else 'fail'}")
        print(f"metrics: {json.dumps(dict(ev.metrics), sort_keys=True)}")
        if ev.success and "neg_mse" in ev.metrics:
            neg = float(ev.metrics["neg_mse"])
            exact = abs(neg) <= 1e-15
            near = neg >= DEFAULT_NEAR_NEG_MSE
            print(f"examples: exact={'yes' if exact else 'no'} near_threshold={'yes' if near else 'no'}")
        if not ev.success:
            raise SystemExit(1)


def _cmd_copilot_draft(args: argparse.Namespace) -> None:
    from axiom.copilot.search import CopilotSearchConfig, run_copilot_draft

    expert = _make_copilot_expert(args)
    ex_in = None
    ex_out = None
    mode = "compile_only"
    if getattr(args, "examples_json", None) is not None:
        ex_in, ex_out = _load_examples_json(Path(args.examples_json))
        mode = "predict_rows"
    cfg = CopilotSearchConfig(
        expert=expert,
        goal=args.goal,
        domain_context=args.context,
        example_input_rows=ex_in,
        expected_rows=ex_out,
        mode=mode,
    )
    _, resp = run_copilot_draft(cfg)
    ax = resp.ax_source.rstrip() + "\n"
    print(ax, end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(ax, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)


def _cmd_copilot_search(args: argparse.Namespace) -> None:
    from axiom.copilot.search import run_copilot_search

    expert = _make_copilot_expert(args)
    cfg = _build_copilot_search_config(args, expert)
    summarize = bool(getattr(args, "summarize_traces", False))
    result = run_copilot_search(cfg)

    for rec in result.iterations:
        ev = rec.evaluation
        print(
            f"[iter {rec.index}] success={ev.success} stage={ev.compile_stage_reached!r} "
            f"failures={len(ev.failures)} metrics={dict(ev.metrics)}",
            file=sys.stderr,
        )
    print(result.best_source.rstrip() + "\n", end="")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.best_source.rstrip() + "\n", encoding="utf-8")
        print(f"Wrote best program to {args.out}", file=sys.stderr)
    if args.report_out is not None:
        payload = {
            "converged": result.converged,
            "convergence_reason": result.convergence_reason,
            "metric_repair": {
                "enabled": result.metric_repair_enabled,
                "threshold_effective": result.metric_repair_threshold_effective,
            },
            "best_source": result.best_source,
            "best_evaluation": _serialize_evaluation_report(result.best_evaluation),
            "final_report": _serialize_evaluation_report(result.final_report),
            "iterations": [
                {
                    "index": rec.index,
                    "source": rec.source,
                    "evaluation": _serialize_evaluation_report(rec.evaluation),
                    "producing_payload": rec.producing_payload,
                    "producing_expert": rec.producing_expert,
                    "outgoing_repair_error_report": rec.outgoing_repair_error_report,
                    "semantic_trace_summary": rec.semantic_trace_summary,
                }
                for rec in result.iterations
            ],
        }
        if summarize:
            payload["semantic_summaries"] = {
                "enabled": True,
                "per_iteration": [
                    {"index": r.index, "semantic_trace_summary": r.semantic_trace_summary}
                    for r in result.iterations
                ],
            }
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.report_out}", file=sys.stderr)
    if args.artifact_dir is not None:
        print(f"Wrote artifact bundle to {args.artifact_dir.resolve()}", file=sys.stderr)


def _cmd_copilot_run(args: argparse.Namespace) -> None:
    from axiom.copilot.pipeline import CopilotPipelineConfig, copilot_pipeline_summary_dict, run_copilot_pipeline

    expert = _make_copilot_expert(args)
    cfg = _build_copilot_search_config(args, expert)
    summarize = bool(getattr(args, "summarize_traces", False))
    restarts = max(1, int(getattr(args, "restarts", 1)))
    pcfg = CopilotPipelineConfig(
        search=cfg,
        best_ax_path=args.out,
        summary_json_path=getattr(args, "summary_out", None),
        final_validate=not bool(getattr(args, "no_final_validate", False)),
        restarts=restarts,
    )
    result = run_copilot_pipeline(pcfg)
    sr = result.search_result
    prefix = "[copilot-run]"
    fv = result.final_validation
    fv_ok = fv.success if fv is not None else None
    print(
        f"{prefix} converged={sr.converged} convergence_reason={sr.convergence_reason!r} "
        f"best_eval_ok={sr.best_evaluation.success} final_validation_ok={fv_ok}"
        + (
            f" restarts={result.restarts} winning_restart_index={result.winning_restart_index}"
            if result.restarts > 1
            else ""
        ),
        file=sys.stderr,
    )
    if fv is not None and not fv.success:
        print(
            f"{prefix} FINAL VALIDATION FAILED: compile_stage_reached={fv.compile_stage_reached!r} "
            f"failures={len(fv.failures)} (champion source did not re-compile cleanly).",
            file=sys.stderr,
        )
        for f in fv.failures:
            print(f"{prefix}   [{f.stage}/{f.kind}] {f.message}", file=sys.stderr)
    for rec in sr.iterations:
        ev = rec.evaluation
        print(
            f"{prefix} [iter {rec.index}] success={ev.success} stage={ev.compile_stage_reached!r} "
            f"metrics={dict(ev.metrics)}",
            file=sys.stderr,
        )
    print(sr.best_source.rstrip() + "\n", end="")
    if args.out is not None:
        print(f"Wrote best program to {args.out}", file=sys.stderr)
    summ_path = getattr(args, "summary_out", None)
    if summ_path is not None:
        doc = copilot_pipeline_summary_dict(
            result,
            artifact_dir_resolved=result.artifact_dir,
            summarize_traces=summarize,
        )
        summ_path.parent.mkdir(parents=True, exist_ok=True)
        summ_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"Wrote pipeline summary to {summ_path}", file=sys.stderr)
    if cfg.artifact_dir is not None:
        root = cfg.artifact_dir.resolve()
        if restarts > 1:
            print(
                f"{prefix} Wrote artifact bundles under {root} (restart_0 … restart_{restarts - 1}/).",
                file=sys.stderr,
            )
        else:
            print(f"{prefix} Wrote artifact bundle to {root}", file=sys.stderr)


def _cmd_copilot_stability_report(args: argparse.Namespace) -> None:
    from axiom.copilot.stability_report import (
        collect_stability_report,
        format_discovery_failure_message,
        stability_report_to_dict,
    )

    paths = list(getattr(args, "paths", None) or [])
    parent = getattr(args, "parent", None)
    near = float(getattr(args, "near_threshold", -1e-9))
    runs, agg, text, discovery = collect_stability_report(paths, parent=parent, near_floor=near)
    if not runs:
        print(format_discovery_failure_message(discovery), file=sys.stderr)
        raise SystemExit(1)
    print(text)
    jo = getattr(args, "json_out", None)
    if jo is not None:
        doc = stability_report_to_dict(runs, agg, discovery=discovery)
        outp = Path(jo)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote JSON to {outp}", file=sys.stderr)


def _print_copilot_benchmark_summary(doc: dict) -> None:
    prefix = "[copilot-benchmark]"
    if doc.get("draft_summary"):
        s = doc["draft_summary"]
        print(
            f"{prefix} draft: n={s['task_count']} compile_ok={s['compile_ok_count']} "
            f"compile_rate={100.0 * float(s['compile_success_rate']):.1f}% "
            f"metric_ok={s['metric_ok_count']} "
            f"metric_rate={100.0 * float(s['metric_success_rate']):.1f}%",
            file=sys.stderr,
        )
    if doc.get("search_summary"):
        s = doc["search_summary"]
        print(
            f"{prefix} search: n={s['task_count']} compile_ok={s['compile_ok_count']} "
            f"compile_rate={100.0 * float(s['compile_success_rate']):.1f}% "
            f"metric_ok={s['metric_ok_count']} "
            f"metric_rate={100.0 * float(s['metric_success_rate']):.1f}%",
            file=sys.stderr,
        )


def _cmd_copilot_benchmark(args: argparse.Namespace) -> None:
    from axiom.copilot.benchmarks import benchmark_suite_to_dict, load_benchmark_tasks_json_path, run_benchmark_suite

    expert = _make_copilot_expert(args)
    if args.draft_only and args.search_only:
        raise SystemExit("Cannot use --draft-only and --search together.")
    run_draft = not args.search_only
    run_search = not args.draft_only
    task_list = None
    if args.task_json is not None:
        try:
            task_list = load_benchmark_tasks_json_path(args.task_json)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            raise SystemExit(f"Invalid --task-json: {e}") from e
    suite = run_benchmark_suite(
        expert,
        tasks=task_list,
        max_iterations=max(1, int(args.max_iterations)),
        run_draft=run_draft,
        run_search=run_search,
        completion_overrides=_completion_overrides_from_args(args),
    )
    doc = benchmark_suite_to_dict(suite)
    _print_copilot_benchmark_summary(doc)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote benchmark JSON to {args.out}", file=sys.stderr)


def _add_copilot_benchmark_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend",
        choices=["onyx-qwen", "benchmark-dispatch"],
        required=True,
        help="Semantic expert implementation (`benchmark-dispatch` is deterministic/offline for CI).",
    )
    p.add_argument(
        "--expert-url",
        type=str,
        required=False,
        default=None,
        help="Base URL for chat/completions (required for onyx-qwen only).",
    )
    p.add_argument(
        "--expert-model",
        type=str,
        required=False,
        default=None,
        help="Remote model id (required for onyx-qwen only).",
    )
    p.add_argument(
        "--expert-api-key",
        type=str,
        default=None,
        help="Optional API key (else AXIOM_EXPERT_API_KEY).",
    )


def _add_copilot_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--backend",
        choices=["onyx-qwen"],
        required=True,
        help="Semantic expert implementation (OpenAI-style chat; requires [copilot] / requests).",
    )
    p.add_argument("--goal", type=str, required=True, help="Natural-language goal for the .ax program.")
    p.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional domain or task notes (embedded in expert context).",
    )
    p.add_argument(
        "--expert-url",
        type=str,
        required=True,
        help="Base URL for chat/completions (e.g. https://api.example.com/v1/).",
    )
    p.add_argument(
        "--expert-model",
        type=str,
        required=True,
        help="Remote model id (passed through to the chat API).",
    )
    p.add_argument(
        "--expert-api-key",
        type=str,
        default=None,
        help="Optional API key (else AXIOM_EXPERT_API_KEY).",
    )
    p.add_argument("--out", type=Path, default=None, help="Optional path to write the best/latest .ax source.")


def _add_copilot_search_loop_args(p: argparse.ArgumentParser) -> None:
    """Draft→repair search options (shared by ``copilot-search`` and ``copilot-run``)."""
    p.add_argument(
        "--iterations",
        type=int,
        default=8,
        metavar="N",
        help="Maximum evaluation rounds (default: 8).",
    )
    p.add_argument(
        "--examples-json",
        type=Path,
        default=None,
        help='Optional JSON file: [{"inputs":{...},"expected":{...}}, ...] for predict_rows scoring.',
    )
    p.add_argument(
        "--train-tabular",
        action="store_true",
        help="Evaluate candidates with in-memory train+eval (requires --tabular-json; see tabular_json module).",
    )
    p.add_argument(
        "--tabular-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="JSON object: target_var, train_rows, eval_rows (rows: inputs+expected), optional epochs/lr/weight_decay/batch_size.",
    )
    p.add_argument(
        "--compile-only",
        action="store_true",
        help="Validate with compile_only (ignore --examples-json for execution).",
    )
    p.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Write reproducible bundle: best.ax, iterations.json, search_report.json (creates dir).",
    )
    p.add_argument(
        "--summarize-traces",
        action="store_true",
        help="After each iteration, call the expert summarize_trace API (optional; extra latency).",
    )
    p.add_argument(
        "--repair-valid-with-metrics",
        action="store_true",
        help="Force metric-driven repair after a successful compile (default: on for --examples-json / --train-tabular).",
    )
    p.add_argument(
        "--no-repair-valid-with-metrics",
        action="store_true",
        help="Stop after the first successful compile even if the metric is poor (disables metric-driven repair).",
    )
    p.add_argument(
        "--metric-repair-if-below",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Repair while the score sort key is strictly below this (built-in neg_mse default: -1e-9).",
    )


def _add_copilot_completion_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Optional chat completion temperature (OpenAI-style; expert draft+repair when set). "
        "0 or lower uses greedy decoding (Onyx: do_sample=false, temperature omitted from the HTTP body).",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=None,
        dest="top_p",
        metavar="FLOAT",
        help="Optional nucleus sampling top_p (expert draft+repair when set).",
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="axiom",
        description="Axiom neural compiler CLI (train, predict, copilot-draft, copilot-doctor, copilot-search, copilot-run, copilot-stability-report, copilot-benchmark, copilot-serve, copilot-studio, lock-bundle, export-onnx, inspect, serve, gateway-serve).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Compile .ax, train execution graph, save bundle (or inference).")
    p_train.add_argument("ax_path", type=Path, nargs="?", default=Path("train.ax"), help="Source .ax file")
    p_train.add_argument("--mode", choices=["train", "inference"], default="train")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--dim", type=int, default=16)
    p_train.add_argument("--rank", type=int, default=4)
    p_train.add_argument("--batch", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=1e-2)
    p_train.add_argument("--baseline-var", type=float, default=0.05, dest="baseline_var")
    p_train.add_argument("--out", type=Path, default=Path("axiom_bundle"))
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument(
        "--dataset",
        choices=["titanic", "sine"],
        default=None,
        help="Built-in dataset (tabular AxiomDataset + test metric). Omit for legacy LiquidSequenceLoader.",
    )
    p_train.add_argument("--csv", type=Path, default=None, help="Custom CSV; requires --target_key and --target_var.")
    p_train.add_argument("--target_key", type=str, default="", help="CSV column name for the training label.")
    p_train.add_argument(
        "--target_var",
        type=str,
        default="",
        help="Graph ABI variable name to supervise (e.g. survived_prob, y_pred).",
    )
    p_train.add_argument(
        "--split-frac",
        type=float,
        default=0.8,
        dest="split_frac",
        help="Train fraction for tabular splits (default 0.8).",
    )
    p_train.add_argument(
        "--titanic-csv",
        type=Path,
        default=Path("examples/titanic.csv"),
        help="Path for Titanic CSV (downloaded if missing).",
    )
    p_train.add_argument("--sine-samples", type=int, default=1000, help="Sample count for --dataset sine.")
    p_train.add_argument(
        "--loop-max-unroll",
        type=int,
        default=None,
        help="Liquid loop unroll (default: 10 for sine, 8 otherwise).",
    )
    p_train.add_argument("--loop-num-basis", type=int, default=8, help="Liquid-KAN basis count.")
    p_train.add_argument(
        "--mutation-threshold",
        type=float,
        default=0.99,
        dest="mutation_threshold",
        help="Sinkhorn mutation entropy threshold for tabular training.",
    )
    p_train.add_argument(
        "--no-meta",
        action="store_true",
        help="Disable MetaCompiler (Titanic / conditional CSV graphs).",
    )
    p_train.set_defaults(_handler=_cmd_train)

    p_inspect = sub.add_parser("inspect", help="Launch Glass Box Streamlit visualizer.")
    p_inspect.set_defaults(_handler=_cmd_inspect)

    p_copilot_studio = sub.add_parser(
        "copilot-studio",
        help='Copilot Studio: Streamlit UI for draft/search (needs pip install -e ".[inspect,copilot]").',
    )
    p_copilot_studio.set_defaults(_handler=_cmd_copilot_studio)

    p_copilot_serve = sub.add_parser(
        "copilot-serve",
        help='HTTP copilot API: /draft, /search, /run, /summarize, /benchmarks/run (needs pip install -e ".[serve,copilot]").',
    )
    p_copilot_serve.add_argument(
        "--backend",
        choices=["onyx-qwen"],
        default="onyx-qwen",
        help="Semantic expert backend (default: onyx-qwen).",
    )
    p_copilot_serve.add_argument(
        "--expert-url",
        type=str,
        required=True,
        help="Base URL for chat/completions (e.g. https://api.example.com/v1/).",
    )
    p_copilot_serve.add_argument(
        "--expert-model",
        type=str,
        required=True,
        help="Remote model id for the chat API.",
    )
    p_copilot_serve.add_argument(
        "--expert-api-key",
        type=str,
        default=None,
        help="Optional API key (else AXIOM_EXPERT_API_KEY). POST routes may also require AXIOM_COPILOT_API_KEY.",
    )
    p_copilot_serve.add_argument("--host", type=str, default="127.0.0.1", help="Bind address.")
    p_copilot_serve.add_argument("--port", type=int, default=8020, help="TCP port (default: 8020).")
    p_copilot_serve.set_defaults(_handler=_cmd_copilot_serve)

    p_predict = sub.add_parser(
        "predict",
        help="Run a saved InterpretedBlock .axb on one JSON feature row (unlocks locked bundles if the environment matches).",
    )
    p_predict.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to .axb from save_bundle (locked .axb decrypts when allowed).",
    )
    p_predict.add_argument(
        "--input",
        type=str,
        required=True,
        help='JSON object of ABI feature names, e.g. \'{"volatility":0.6,"drawdown":0.1}\'',
    )
    p_predict.set_defaults(_handler=_cmd_predict)

    p_lock = sub.add_parser(
        "lock-bundle",
        help="Re-save an .axb with AES-256-CTR encrypted neural weights (topology stays readable).",
    )
    p_lock.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source .axb (unlocked). Example bundles are often gitignored — train first, e.g. python examples/train_portfolio.py.",
    )
    p_lock.add_argument("--output", type=Path, required=True, help="Destination .axb.")
    p_lock.add_argument(
        "--mode",
        choices=["device", "host", "env-secret"],
        required=True,
        help="device=CUDA identity, host=machine identity, env-secret=AXIOM_BUNDLE_SECRET.",
    )
    p_lock.set_defaults(_handler=_cmd_lock_bundle)

    p_onnx = sub.add_parser(
        "export-onnx",
        help="Export an InterpretedBlock .axb to ONNX (dense tensor in/out; inference-only).",
    )
    p_onnx.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to .axb (InterpretedBlock bundle from save_bundle).",
    )
    p_onnx.add_argument("--output", type=Path, required=True, help="Destination .onnx file.")
    p_onnx.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    p_onnx.set_defaults(_handler=_cmd_export_onnx)

    p_gw = sub.add_parser(
        "gateway-serve",
        help="Policy gateway HTTP server (POST /gateway/chat; requires Axiom [gateway] deps).",
    )
    p_gw.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="InterpretedBlock .axb (policy bundle).",
    )
    p_gw.add_argument(
        "--downstream-url",
        type=str,
        required=True,
        help="URL to forward approved requests (JSON body includes message).",
    )
    p_gw.add_argument(
        "--policy-source",
        type=Path,
        default=None,
        help="Optional .ax source text for Glass Box HTML audit reports.",
    )
    p_gw.add_argument(
        "--audit-path",
        type=Path,
        default=None,
        help="When policy blocks, also write audit HTML to this path.",
    )
    p_gw.add_argument(
        "--approve-threshold",
        type=float,
        default=0.5,
        help="Minimum is_approved trace value to allow downstream (default: 0.5).",
    )
    p_gw.add_argument("--host", type=str, default="127.0.0.1", help="Bind address.")
    p_gw.add_argument("--port", type=int, default=8010, help="TCP port (default: 8010).")
    p_gw.set_defaults(_handler=_cmd_gateway_serve)

    p_serve = sub.add_parser(
        "serve",
        help="Serve one .axb bundle over HTTP (FastAPI: /health, /predict, /explain, /report).",
    )
    p_serve.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Path to .axb (default: env AXIOM_BUNDLE_PATH).",
    )
    p_serve.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind address when HOST env is unset (Docker: set HOST=0.0.0.0).",
    )
    p_serve.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port when PORT env is unset.",
    )
    p_serve.set_defaults(_handler=_cmd_serve)

    p_cd = sub.add_parser(
        "copilot-draft",
        help="Ask a semantic expert to draft .ax source from a goal (requires pip install -e \".[copilot]\").",
    )
    _add_copilot_backend_args(p_cd)
    p_cd.add_argument(
        "--examples-json",
        type=Path,
        default=None,
        help="Optional JSON array of {inputs, expected} rows; enables exact fast-path draft inference when rows match.",
    )
    p_cd.set_defaults(_handler=_cmd_copilot_draft)

    p_cdoc = sub.add_parser(
        "copilot-doctor",
        help="Smoke-test expert HTTP + one deterministic draft + parse/IR/block; optional --examples-json eval (Phase 85b).",
    )
    p_cdoc.add_argument(
        "--backend",
        choices=["onyx-qwen"],
        required=True,
        help="Semantic expert implementation (requires [copilot] / requests).",
    )
    p_cdoc.add_argument(
        "--goal",
        type=str,
        default=_COPILOT_DOCTOR_DEFAULT_GOAL,
        help="NL goal for the single draft request (default: double-x smoke goal).",
    )
    p_cdoc.add_argument(
        "--expert-url",
        type=str,
        required=True,
        help="Base URL for chat/completions (e.g. https://api.example.com/v1/).",
    )
    p_cdoc.add_argument(
        "--expert-model",
        type=str,
        required=True,
        help="Remote model id (passed through to the chat API).",
    )
    p_cdoc.add_argument(
        "--expert-api-key",
        type=str,
        default=None,
        help="Optional API key (else AXIOM_EXPERT_API_KEY).",
    )
    p_cdoc.add_argument(
        "--timeout",
        type=float,
        default=None,
        dest="expert_timeout",
        metavar="SEC",
        help="Per-request HTTP timeout in seconds (default: 120).",
    )
    p_cdoc.add_argument(
        "--examples-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional [{inputs},{expected}] rows JSON; after compile OK, run predict_rows + neg_mse (Phase 85b).",
    )
    _add_copilot_completion_args(p_cdoc)
    p_cdoc.set_defaults(_handler=_cmd_copilot_doctor)

    p_cs = sub.add_parser(
        "copilot-search",
        help="Draft / evaluate / repair loop with an expert until success or iteration budget (see Phase 60).",
    )
    _add_copilot_backend_args(p_cs)
    p_cs.add_argument(
        "--timeout",
        "--expert-timeout",
        type=float,
        default=None,
        dest="expert_timeout",
        metavar="SEC",
        help="Per-request HTTP timeout in seconds (default: 120).",
    )
    _add_copilot_search_loop_args(p_cs)
    _add_copilot_completion_args(p_cs)
    p_cs.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Write structured JSON (iterations, metrics, sources) to this path.",
    )
    p_cs.set_defaults(_handler=_cmd_copilot_search)

    p_csr = sub.add_parser(
        "copilot-stability-report",
        help="Summarize metrics across copilot artifact dirs or pipeline summary JSON files (Phase 83+83b; no network).",
    )
    p_csr.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[],
        help="Scan root(s): recurse for **/search_report.json, pipeline *.json, and restart_* bundles.",
    )
    p_csr.add_argument(
        "--parent",
        type=Path,
        default=None,
        help="Extra scan root (same recursion as paths); use for sweep folders with many run JSON files.",
    )
    p_csr.add_argument(
        "--json-out",
        type=Path,
        default=None,
        dest="json_out",
        help="Write full stability report JSON (runs + aggregate + summary_text + discovery).",
    )
    p_csr.add_argument(
        "--near-threshold",
        type=float,
        default=-1e-9,
        dest="near_threshold",
        metavar="FLOAT",
        help="For neg_mse tasks, count near-hit when neg_mse >= this (default: -1e-9).",
    )
    p_csr.set_defaults(_handler=_cmd_copilot_stability_report)

    p_cr = sub.add_parser(
        "copilot-run",
        help="End-to-end NL→.ax pipeline: search + optional artifact bundle + pipeline summary JSON + final compile check (Phase 71).",
    )
    _add_copilot_backend_args(p_cr)
    p_cr.add_argument(
        "--timeout",
        "--expert-timeout",
        type=float,
        default=None,
        dest="expert_timeout",
        metavar="SEC",
        help="Per-request HTTP timeout in seconds (default: 120).",
    )
    _add_copilot_search_loop_args(p_cr)
    _add_copilot_completion_args(p_cr)
    p_cr.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write pipeline summary JSON (disclaimer, iterations, evaluations, final_validation).",
    )
    p_cr.add_argument(
        "--no-final-validate",
        action="store_true",
        help="Skip the extra compile-only pass on the champion source after search.",
    )
    p_cr.add_argument(
        "--restarts",
        type=int,
        default=1,
        metavar="N",
        help="Run N independent searches and keep the overall best (default: 1). With --artifact-dir, use restart_0 … restart_{N-1} subdirs when N>1.",
    )
    p_cr.set_defaults(_handler=_cmd_copilot_run)

    p_cb = sub.add_parser(
        "copilot-benchmark",
        help="Run semantic copilot NL→.ax benchmark suite (draft vs search; requires pip install -e \".[copilot]\").",
    )
    _add_copilot_benchmark_backend_args(p_cb)
    p_cb.add_argument(
        "--task-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional JSON file with {\"tasks\":[...]} (same as axiom.copilot.benchmarks fixtures); default built-in tasks.",
    )
    p_cb.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write full benchmark_suite_to_dict JSON to this path.",
    )
    p_cb.add_argument(
        "--max-iterations",
        type=int,
        default=4,
        metavar="N",
        help="Search arm iteration budget per task (default: 4).",
    )
    p_cb.add_argument(
        "--draft-only",
        action="store_true",
        help="Run only the draft+eval arm (skip search/repair loop).",
    )
    p_cb.add_argument(
        "--search",
        action="store_true",
        dest="search_only",
        help="Run only the search arm (skip draft-only baseline).",
    )
    _add_copilot_completion_args(p_cb)
    p_cb.set_defaults(_handler=_cmd_copilot_benchmark)

    args = ap.parse_args(argv)
    handler = args._handler
    if handler is _cmd_inspect:
        raise SystemExit(handler(args))
    if handler is _cmd_copilot_studio:
        raise SystemExit(handler(args))
    if handler is _cmd_predict:
        handler(args)
        return
    if handler is _cmd_lock_bundle:
        handler(args)
        return
    if handler is _cmd_serve:
        handler(args)
        return
    if handler is _cmd_gateway_serve:
        handler(args)
        return
    if handler is _cmd_export_onnx:
        handler(args)
        return
    if handler is _cmd_copilot_draft:
        handler(args)
        return
    if handler is _cmd_copilot_doctor:
        handler(args)
        return
    if handler is _cmd_copilot_search:
        handler(args)
        return
    if handler is _cmd_copilot_run:
        handler(args)
        return
    if handler is _cmd_copilot_stability_report:
        handler(args)
        return
    if handler is _cmd_copilot_benchmark:
        handler(args)
        return
    if handler is _cmd_copilot_serve:
        handler(args)
        return
    handler(args)


if __name__ == "__main__":
    main()
