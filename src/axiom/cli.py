from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="axiom",
        description="Axiom neural compiler CLI (train, predict, lock-bundle, export-onnx, inspect, serve, gateway-serve).",
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

    args = ap.parse_args(argv)
    handler = args._handler
    if handler is _cmd_inspect:
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
    handler(args)


if __name__ == "__main__":
    main()
