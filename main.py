from __future__ import annotations

import argparse
from pathlib import Path

import torch

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax_file
from compiler.deserializer import load_execution_bundle
from compiler.serializer import save_execution_bundle
from engine.dataloader import LiquidSequenceLoader
from engine.inference import AxiomRunner
from engine.meta_compiler import MetaCompiler
from engine.supernet import LatentSupernet
from engine.topology import ExecutionGraph
from engine.trainer import EvolutionaryTrainer


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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Axiom: compile .ax, train execution graph, save bundle.")
    ap.add_argument("ax_path", type=Path, nargs="?", default=Path("train.ax"), help="Source .ax file")
    ap.add_argument("--mode", choices=["train", "inference"], default="train")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--baseline-var", type=float, default=0.05, dest="baseline_var")
    ap.add_argument("--out", type=Path, default=Path("axiom_bundle"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

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


if __name__ == "__main__":
    main()
