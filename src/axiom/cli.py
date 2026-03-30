from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from axiom.compiler.deserializer import load_execution_bundle
from axiom.compiler.flow import wire_execution_graph
from axiom.compiler.ir import ast_to_ir
from axiom.compiler.parser import parse_ax_file
from axiom.compiler.serializer import save_execution_bundle
from axiom.engine.dataloader import LiquidSequenceLoader
from axiom.engine.inference import AxiomRunner
from axiom.engine.meta_compiler import MetaCompiler
from axiom.engine.supernet import LatentSupernet
from axiom.engine.topology import ExecutionGraph
from axiom.engine.trainer import EvolutionaryTrainer


def build_from_ax(ax_path: Path, dim: int, rank: int) -> tuple[list, LatentSupernet, ExecutionGraph]:
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
    g = wire_execution_graph(ir, sn, pairs)
    return ir, sn, g


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

    ir, sn, graph = build_from_ax(args.ax_path, args.dim, args.rank)
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

    out = args.out
    save_execution_bundle(graph.cpu(), out, ir=ir)
    print(f"Saved {out}.pt and {out}_topology.json")


def _cmd_inspect(_args: argparse.Namespace) -> int:
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
    ap = argparse.ArgumentParser(prog="axiom", description="Axiom neural compiler CLI.")
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
    p_train.set_defaults(_handler=_cmd_train)

    p_inspect = sub.add_parser("inspect", help="Launch Glass Box Streamlit visualizer.")
    p_inspect.set_defaults(_handler=_cmd_inspect)

    args = ap.parse_args(argv)
    handler = args._handler
    if handler is _cmd_inspect:
        raise SystemExit(handler(args))
    handler(args)


if __name__ == "__main__":
    main()
