"""Phase 27: readme narrative ↔ repo facts (examples IR, CLI surface, public APIs)."""

from pathlib import Path

import pytest

from axiom.cli import main
from axiom.compiler.ir import ast_to_ir, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_readme_has_narrative_sections():
    text = (_root() / "readme.md").read_text(encoding="utf-8")
    assert "## Understanding Axiom" in text
    assert "## Where Axiom shines" in text
    assert "## Road ahead" in text
    assert "symbolic" in text.lower() and "neural" in text.lower()


def test_readme_version_matches_pyproject():
    readme = (_root() / "readme.md").read_text(encoding="utf-8")
    pyproject = (_root() / "pyproject.toml").read_text(encoding="utf-8")
    assert "1.1.0" in readme
    assert 'version = "1.1.0"' in pyproject


def test_examples_titanic_ax_has_conditional_ir():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "titanic.ax"))
    assert any(x[0] == "OP_CONDITIONAL" for x in ir)


def test_examples_sequence_ax_has_loop_no_conditional():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "sequence.ax"))
    assert any(x[0] == "OP_LOOP" for x in ir)
    assert not any(x[0] == "OP_CONDITIONAL" for x in ir)


def test_examples_football_ax_has_gd_pred_abi():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "football.ax"))
    assert sum(1 for x in ir if x[0] == "OP_CONDITIONAL") == 2
    abi = extract_global_abi(ir, max_vars=32)
    assert abi.get("gd_pred") is not None and "B365H" in abi


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--help"],
        ["inspect", "--help"],
    ],
)
def test_cli_subcommands_help_exits_ok(argv: list):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0


def test_cli_source_wires_documented_train_features():
    """Readme/plan mention --dataset, built-ins, and mutual exclusion — keep them in cli.py."""
    src = (_root() / "src" / "axiom" / "cli.py").read_text(encoding="utf-8")
    assert "--dataset" in src
    assert "load_titanic" in src
    assert "load_football" in src
    assert "generate_sine_wave" in src
    assert "Use either --dataset or --csv" in src


def test_cli_rejects_dataset_and_csv_together():
    with pytest.raises(SystemExit):
        main(
            [
                "train",
                str(_root() / "train.ax"),
                "--dataset",
                "sine",
                "--csv",
                "x.csv",
                "--target_key",
                "y",
                "--target_var",
                "z",
                "--epochs",
                "1",
                "--out",
                "o",
            ]
        )


def test_glass_box_inspector_entrypoint_exists():
    import axiom.tools.inspector as inspector

    assert Path(inspector.__file__).name == "inspector.py"


def test_datasets_module_public_api(tmp_path: Path):
    from axiom.datasets import generate_sine_wave, load_titanic, train_val_split

    rows = generate_sine_wave(n=2, seed=1)
    assert len(rows) == 2
    a, b = train_val_split(rows, frac=0.5, seed=0)
    assert len(a) + len(b) == 2
    csv = tmp_path / "t.csv"
    csv.write_text("Fare,Sex,Pclass,Survived\n0,1,1,1\n", encoding="utf-8")
    trows = load_titanic(csv_path=csv)
    assert len(trows) == 1 and trows[0]["Survived"] == 1.0
    assert (_root() / "examples" / "titanic.ax").is_file()
