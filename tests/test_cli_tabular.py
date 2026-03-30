"""Phase 25+: `axiom train` with --dataset / --csv."""

from pathlib import Path
import pytest

from axiom.cli import main


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_cli_train_dataset_sine_smoke(tmp_path: Path):
    ax = _root() / "examples" / "sequence.ax"
    out = tmp_path / "bundle"
    main(
        [
            "train",
            str(ax),
            "--dataset",
            "sine",
            "--epochs",
            "1",
            "--dim",
            "16",
            "--batch",
            "8",
            "--sine-samples",
            "48",
            "--out",
            str(out),
            "--seed",
            "0",
        ]
    )
    assert Path(str(out) + ".pt").is_file()
    assert Path(str(out) + "_topology.json").is_file()


def test_cli_train_dataset_titanic_smoke(tmp_path: Path):
    csv = tmp_path / "mini.csv"
    csv.write_text(
        "Fare,Sex,Pclass,Survived\n0,1,1,1\n0,1,1,1\n0,0,3,0\n0,0,3,0\n",
        encoding="utf-8",
    )
    ax = _root() / "examples" / "titanic.ax"
    out = tmp_path / "tb"
    main(
        [
            "train",
            str(ax),
            "--dataset",
            "titanic",
            "--epochs",
            "1",
            "--dim",
            "16",
            "--batch",
            "2",
            "--titanic-csv",
            str(csv),
            "--out",
            str(out),
            "--seed",
            "0",
            "--no-meta",
        ]
    )
    assert Path(str(out) + ".pt").is_file()


def test_cli_csv_requires_target_fields(tmp_path: Path):
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    ax = _root() / "train.ax"
    out = tmp_path / "o"
    with pytest.raises(SystemExit):
        main(
            [
                "train",
                str(ax),
                "--csv",
                str(csv),
                "--epochs",
                "1",
                "--out",
                str(out),
            ]
        )


def test_cli_dataset_and_csv_mutually_exclusive(tmp_path: Path):
    csv = tmp_path / "d.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    ax = _root() / "train.ax"
    out = tmp_path / "o"
    with pytest.raises(SystemExit):
        main(
            [
                "train",
                str(ax),
                "--dataset",
                "sine",
                "--csv",
                str(csv),
                "--target_key",
                "b",
                "--target_var",
                "x",
                "--epochs",
                "1",
                "--out",
                str(out),
            ]
        )
