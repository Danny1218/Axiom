from pathlib import Path

import main as axiom_main


def test_main_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    ax = root / "train.ax"
    out = tmp_path / "bundle"
    axiom_main.main(
        [
            str(ax),
            "--epochs",
            "1",
            "--batch",
            "8",
            "--dim",
            "8",
            "--out",
            str(out),
            "--seed",
            "1",
        ]
    )
    assert Path(str(out) + ".pt").is_file()
    assert Path(str(out) + "_topology.json").is_file()
