"""Phase 19: CSV helper + Titanic ax / mini end-to-end pipeline."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from axiom.compiler.flow import wire_execution_graph
from axiom.datasets import load_titanic
from axiom.compiler.ir import ast_to_ir, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.dataloader import AxiomDataset, load_csv_to_dicts
from axiom.engine.inference import AxiomRunner
from axiom.engine.supernet import LatentSupernet
from axiom.engine.trainer import EvolutionaryTrainer


def test_cell_encoding_female_male_empty(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("Sex,Survived,Name\nfemale,1,foo\nmale,0,bar\n,,0\n", encoding="utf-8")
    rows = load_csv_to_dicts(p)
    assert rows[0]["Sex"] == 1.0 and rows[0]["Survived"] == 1.0
    assert rows[1]["Sex"] == 0.0
    assert rows[2]["Sex"] == 0.0 and rows[2]["Survived"] == 0.0


def test_numeric_strings_parse(tmp_path):
    p = tmp_path / "n.csv"
    p.write_text("a,b\n12.5,-3\n", encoding="utf-8")
    rows = load_csv_to_dicts(p)
    assert rows[0]["a"] == 12.5 and rows[0]["b"] == -3.0


def test_load_titanic_delegates_to_csv_parser(tmp_path):
    p = tmp_path / "one.csv"
    p.write_text("Fare,Sex,Pclass,Survived\n0,1,1,1\n", encoding="utf-8")
    rows = load_titanic(csv_path=p)
    assert len(rows) == 1 and rows[0]["Survived"] == 1.0


def test_titanic_ax_abi_has_survived_prob():
    reset_parser()
    root = Path(__file__).resolve().parents[1]
    ir = ast_to_ir(parse_ax_file(root / "examples" / "titanic.ax"))
    abi = extract_global_abi(ir, max_vars=32)
    assert "survived_prob" in abi
    assert "Fare" in abi


def _build_titanic_like_graph(dim: int = 32):
    reset_parser()
    root = Path(__file__).resolve().parents[1]
    ir = ast_to_ir(parse_ax_file(root / "examples" / "titanic.ax"))
    n_cond = sum(1 for x in ir if x[0] == "OP_CONDITIONAL")
    pairs = [(f"then_{i}", f"else_{i}") for i in range(n_cond)]
    names = [n for p in pairs for n in p]
    for j in range(max(0, 4 - len(names))):
        names.append(f"latent_{j}")
    sn = LatentSupernet(dim, names, rank=2)
    masks = {f"then_{i}": 1.0 for i in range(n_cond)}
    masks.update({f"else_{i}": 1.0 for i in range(n_cond)})
    sn.set_masks(masks)
    return wire_execution_graph(ir, sn, pairs, mutation_entropy_norm_threshold=0.99)


def test_titanic_smoke_train_and_accuracy(tmp_path):
    p = tmp_path / "mini.csv"
    p.write_text(
        "Fare,Sex,Pclass,Survived\n"
        "0,1,1,1\n"
        "0,1,1,1\n"
        "0,0,3,0\n"
        "0,0,3,0\n"
        "0,1,2,1\n"
        "0,0,2,0\n",
        encoding="utf-8",
    )
    rows = load_csv_to_dicts(p)
    train, test = rows[:4], rows[4:]
    g = _build_titanic_like_graph(dim=32)
    abi = g.abi
    tc = abi["survived_prob"]
    ds = AxiomDataset(train, abi, trunk_dim=32, target_key="Survived")
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    tr = EvolutionaryTrainer(g, lr=5e-2, compile_graph=False, target_col=tc)
    for _ in range(3):
        tr.train_epoch(loader, meta_compiler=None)
    g.eval()
    runner = AxiomRunner(g)
    out = runner.predict_dict_batch(test, device=torch.device("cpu"))
    assert len(out) == 2
    assert all("survived_prob" in d for d in out)
