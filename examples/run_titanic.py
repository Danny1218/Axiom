"""
Train Axiom on Titanic CSV: compile ``titanic.ax``, EvolutionaryTrainer + AxiomDataset, test accuracy.

Install the package (once): ``pip install -e .`` from the repo root.

Run from repo root::

    python examples/run_titanic.py --epochs 50

Sabotage experiment (useless ``Fare > 100000`` rule in ``titanic.ax``): keep **MetaCompiler** on
(default) so Sinkhorn entropy can unmask shadow experts; bundle is written for the Glass Box::

    python examples/run_titanic.py --epochs 50
    axiom inspect   # Glass Box; sidebar path prefix ``axiom_bundle`` (no extension)

Use ``--no-meta`` to disable DNAS unmasking; ``--no-save`` to skip ``axiom_bundle*.pt/json``.
"""
from __future__ import annotations

import argparse
import random
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]

from axiom.compiler.flow import wire_execution_graph
from axiom.compiler.ir import ast_to_ir
from axiom.compiler.parser import parse_ax_file
from axiom.compiler.serializer import save_execution_bundle
from axiom.engine.dataloader import AxiomDataset, load_csv_to_dicts
from axiom.engine.inference import AxiomRunner
from axiom.engine.meta_compiler import MetaCompiler
from axiom.engine.supernet import LatentSupernet
from axiom.engine.trainer import EvolutionaryTrainer

TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)


def build_graph(ax_path: Path, dim: int, rank: int):
    ir = ast_to_ir(parse_ax_file(ax_path))
    n_cond = sum(1 for x in ir if x[0] == "OP_CONDITIONAL")
    pairs = [(f"then_{i}", f"else_{i}") for i in range(n_cond)]
    names = [n for p in pairs for n in p]
    for j in range(max(0, 4 - len(names))):
        names.append(f"latent_{j}")
    if not names:
        names = ["latent_0", "latent_1"]
    sn = LatentSupernet(dim, names, rank=rank)
    for i in range(n_cond):
        sn.set_masks({f"then_{i}": 1.0, f"else_{i}": 1.0})
    g = wire_execution_graph(ir, sn, pairs, mutation_entropy_norm_threshold=0.99)
    return ir, sn, g


def ensure_titanic_csv(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Titanic CSV to {path} ...")
    urllib.request.urlretrieve(TITANIC_URL, path)


def train_val_split(
    rows: list[dict[str, float]], *, frac: float, seed: int
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * frac)
    return shuffled[:n_train], shuffled[n_train:]


def main() -> None:
    ap = argparse.ArgumentParser(description="Axiom Titanic survival (hybrid DNAS).")
    ap.add_argument(
        "--csv",
        type=Path,
        default=_REPO / "examples" / "titanic.csv",
        help="Path to titanic.csv (downloaded if missing)",
    )
    ap.add_argument(
        "--ax",
        type=Path,
        default=_REPO / "examples" / "titanic.ax",
        help="Path to titanic.ax",
    )
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", type=float, default=0.8, help="Train fraction")
    ap.add_argument(
        "--no-meta",
        action="store_true",
        help="Disable MetaCompiler (default: MetaCompiler unmasks shadow experts from router entropy)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO / "axiom_bundle",
        help="Bundle path prefix for save_execution_bundle",
    )
    ap.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write bundle after training",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ensure_titanic_csv(args.csv)
    except OSError as e:
        raise SystemExit(f"Could not obtain Titanic CSV: {e}") from e

    rows = load_csv_to_dicts(args.csv)
    if not rows:
        raise SystemExit("CSV is empty.")
    if "Survived" not in rows[0]:
        raise SystemExit("CSV must include a Survived column.")

    train_rows, test_rows = train_val_split(rows, frac=args.split, seed=args.seed)

    ir, sn, graph = build_graph(args.ax, args.dim, args.rank)
    graph = graph.to(device)
    abi = graph.abi
    if "survived_prob" not in abi:
        raise SystemExit(f"Expected survived_prob in ABI, got keys: {sorted(abi.keys())}")

    target_col = abi["survived_prob"]
    train_ds = AxiomDataset(train_rows, abi, trunk_dim=args.dim, target_key="Survived")
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    mc = None if args.no_meta else MetaCompiler(sn)
    trainer = EvolutionaryTrainer(graph, lr=args.lr, compile_graph=False, target_col=target_col)
    for ep in range(args.epochs):
        loss = trainer.train_epoch(train_loader, meta_compiler=mc, device=device)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep + 1}/{args.epochs}  mean_mse={loss:.6f}")

    graph.eval()
    runner = AxiomRunner(graph)
    preds = runner.predict_dict_batch(test_rows, device=device)

    correct = 0
    for row, pdict in zip(test_rows, preds):
        prob = pdict.get("survived_prob", 0.0)
        pred = 1.0 if prob > 0.5 else 0.0
        actual = float(row["Survived"])
        if pred == actual:
            correct += 1
    acc = correct / max(len(test_rows), 1)
    print(f"test_accuracy={acc:.4f}  (n={len(test_rows)})")

    if not args.no_save:
        prefix = args.out
        save_execution_bundle(graph.cpu(), prefix, ir=ir)
        print(f"saved bundle prefix {prefix}  ({prefix}_topology.json + {prefix}.pt)")


if __name__ == "__main__":
    main()
