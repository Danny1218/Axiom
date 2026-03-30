"""Phase 25+: `axiom train` with --dataset / --csv."""

from pathlib import Path
from unittest.mock import patch

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


def test_cli_train_dataset_football_smoke(tmp_path: Path):
    csv_text = "B365H,B365D,B365A,FTHG,FTAG\n2.0,3.0,4.0,1,0\n2.2,3.1,3.5,1,1\n"

    def fake_retrieve(url, dest, *a, **kw):
        Path(dest).write_text(csv_text, encoding="utf-8")

    ax = _root() / "examples" / "football.ax"
    out = tmp_path / "fb"
    with patch("axiom.datasets.urllib.request.urlretrieve", fake_retrieve):
        main(
            [
                "train",
                str(ax),
                "--dataset",
                "football",
                "--epochs",
                "1",
                "--dim",
                "16",
                "--batch",
                "2",
                "--out",
                str(out),
                "--seed",
                "0",
                "--no-meta",
            ]
        )
    assert Path(str(out) + ".pt").is_file()


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
