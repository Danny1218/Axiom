"""
Train Axiom on a synthetic sine regression task with a ``while`` loop (Liquid-KAN sequence).

Requires: ``pip install -e .`` from the repo root.

Run::

    python examples/run_sequence.py
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]

from axiom.compiler.flow import wire_execution_graph
from axiom.compiler.ir import ast_to_ir
from axiom.compiler.parser import parse_ax_file
from axiom.engine.dataloader import AxiomDataset
from axiom.engine.inference import AxiomRunner
from axiom.engine.supernet import LatentSupernet
from axiom.engine.trainer import EvolutionaryTrainer


def build_graph(ax_path: Path, dim: int, rank: int, *, loop_max_unroll: int) -> tuple:
    ir = ast_to_ir(parse_ax_file(ax_path))
    n_cond = sum(1 for x in ir if x[0] == "OP_CONDITIONAL")
    if n_cond != 0:
        raise ValueError("sequence.ax must not contain conditionals")
    pairs: list[tuple[str, str]] = []
    names = [n for p in pairs for n in p]
    for j in range(max(0, 4 - len(names))):
        names.append(f"latent_{j}")
    if not names:
        names = ["latent_0", "latent_1"]
    sn = LatentSupernet(dim, names, rank=rank)
    g = wire_execution_graph(
        ir,
        sn,
        pairs,
        mutation_entropy_norm_threshold=0.99,
        loop_max_unroll=loop_max_unroll,
        loop_num_basis=8,
    )
    return ir, sn, g


def train_val_split(
    rows: list[dict[str, float]], *, frac: float, seed: int
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * frac)
    return shuffled[:n_train], shuffled[n_train:]


def make_rows(n: int, *, seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    rows: list[dict[str, float]] = []
    for _ in range(n):
        x = rng.uniform(0.0, 2.0 * math.pi)
        rows.append({"x": x, "y_pred": 0.0, "target": math.sin(x)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Sine sequence crucible (Liquid-KAN loop).")
    ap.add_argument(
        "--ax",
        type=Path,
        default=_REPO / "examples" / "sequence.ax",
        help="Path to sequence.ax",
    )
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000, help="Synthetic sample count")
    ap.add_argument("--split", type=float, default=0.8)
    ap.add_argument(
        "--loop-unroll",
        type=int,
        default=10,
        help="Must match loop semantics (step 0..9 then exit at 10)",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = make_rows(args.n, seed=args.seed + 1)
    train_rows, test_rows = train_val_split(rows, frac=args.split, seed=args.seed)

    _ir, _sn, graph = build_graph(args.ax, args.dim, args.rank, loop_max_unroll=args.loop_unroll)
    graph = graph.to(device)
    abi = graph.abi
    for k in ("x", "y_pred"):
        if k not in abi:
            raise SystemExit(f"Expected {k} in ABI, got: {sorted(abi.keys())}")

    target_col = abi["y_pred"]
    train_ds = AxiomDataset(train_rows, abi, trunk_dim=args.dim, target_key="target")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    trainer = EvolutionaryTrainer(graph, lr=args.lr, compile_graph=False, target_col=target_col)
    for ep in range(args.epochs):
        loss = trainer.train_epoch(train_loader, meta_compiler=None, device=device)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep + 1}/{args.epochs}  mean_mse={loss:.6f}")

    graph.eval()
    runner = AxiomRunner(graph)
    preds = runner.predict_dict_batch(test_rows, device=device)

    se = 0.0
    for row, pdict in zip(test_rows, preds):
        pred = float(pdict.get("y_pred", 0.0))
        tgt = float(row["target"])
        se += (pred - tgt) ** 2
    mse = se / max(len(test_rows), 1)
    print(f"test_mse={mse:.6f}  (n={len(test_rows)})")


if __name__ == "__main__":
    main()
