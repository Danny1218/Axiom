"""Phase 25+: axiom.datasets helpers."""

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from axiom.datasets import generate_sine_wave, load_football, load_titanic, train_val_split


def test_generate_sine_wave_shape_and_target():
    rows = generate_sine_wave(n=5, seed=0)
    assert len(rows) == 5
    for r in rows:
        assert set(r.keys()) >= {"x", "y_pred", "target"}
        assert abs(math.sin(r["x"]) - r["target"]) < 1e-9
        assert r["y_pred"] == 0.0


def test_train_val_split_80_20():
    rows = [{"i": float(i)} for i in range(10)]
    a, b = train_val_split(rows, frac=0.8, seed=123)
    assert len(a) == 8 and len(b) == 2


def test_load_titanic_from_existing_csv(tmp_path: Path):
    p = tmp_path / "t.csv"
    p.write_text("Fare,Sex,Pclass,Survived\n7.25,0,3,0\n71.28,1,1,1\n", encoding="utf-8")
    rows = load_titanic(csv_path=p)
    assert len(rows) == 2
    assert rows[0]["Survived"] == 0.0 and rows[1]["Survived"] == 1.0


def test_load_football_parses_minimal_csv():
    csv_text = "B365H,B365D,B365A,FTHG,FTAG\n2.0,3.0,4.0,2,1\n1.5,4.0,6.0,0,0\n"

    def fake_retrieve(url, dest, *a, **kw):
        Path(dest).write_text(csv_text, encoding="utf-8")

    with patch("axiom.datasets.urllib.request.urlretrieve", fake_retrieve):
        rows = load_football(season="2324")
    assert len(rows) == 2
    r0 = rows[0]
    assert r0["B365H"] == 2.0 and r0["B365D"] == 3.0 and r0["B365A"] == 4.0
    assert r0["gd_pred"] == 0.0 and r0["target_gd"] == 1.0
    assert rows[1]["target_gd"] == 0.0


def test_load_titanic_empty_raises(tmp_path: Path):
    p = tmp_path / "e.csv"
    p.write_text("Fare,Survived\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_titanic(csv_path=p)
