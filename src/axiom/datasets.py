"""Built-in datasets for `axiom train --dataset` (Phase 25+)."""

from __future__ import annotations

import csv
import math
import random
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from axiom.engine.dataloader import load_csv_to_dicts

TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)


def ensure_titanic_csv(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Titanic CSV to {path} ...")
    urllib.request.urlretrieve(TITANIC_URL, path)


def load_titanic(*, csv_path: Path) -> List[Dict[str, float]]:
    """Load Titanic rows (float dicts). Downloads canonical CSV if ``csv_path`` is missing."""
    ensure_titanic_csv(csv_path)
    rows = load_csv_to_dicts(csv_path)
    if not rows:
        raise ValueError("Titanic CSV is empty")
    if "Survived" not in rows[0]:
        raise ValueError("CSV must include a Survived column")
    return rows


def generate_sine_wave(*, n: int = 1000, seed: int = 42) -> List[Dict[str, float]]:
    """Synthetic sine regression rows for ``sequence.ax`` (``x``, ``y_pred`` placeholder, ``target``)."""
    rng = random.Random(seed)
    rows: List[Dict[str, float]] = []
    for _ in range(n):
        x = rng.uniform(0.0, 2.0 * math.pi)
        rows.append({"x": x, "y_pred": 0.0, "target": math.sin(x)})
    return rows


def train_val_split(
    rows: List[Dict[str, float]], *, frac: float = 0.8, seed: int = 0
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * frac)
    return shuffled[:n_train], shuffled[n_train:]


def _football_float(cell: Optional[str]) -> Optional[float]:
    s = (cell or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_football(*, season: str = "2324") -> List[Dict[str, float]]:
    """Premier League match rows from football-data.co.uk (Bet365 odds + full-time goals).

    Target: ``target_gd = FTHG - FTAG``. Placeholder ``gd_pred`` is 0 until the graph runs.
    Rows with missing or non-positive odds are skipped.
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
    out: List[Dict[str, float]] = []
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmpf:
        tmp_path = Path(tmpf.name)
    try:
        print(f"Downloading Premier League data {season} from football-data.co.uk ...")
        urllib.request.urlretrieve(url, tmp_path)
        with tmp_path.open(newline="", encoding="utf-8", errors="replace") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                bh = _football_float(row.get("B365H"))
                bd = _football_float(row.get("B365D"))
                ba = _football_float(row.get("B365A"))
                fthg = _football_float(row.get("FTHG"))
                ftag = _football_float(row.get("FTAG"))
                if bh is None or bd is None or ba is None or fthg is None or ftag is None:
                    continue
                if bh <= 1.0 or bd <= 1.0 or ba <= 1.0:
                    continue
                if fthg < 0 or ftag < 0:
                    continue
                gd = fthg - ftag
                out.append(
                    {
                        "B365H": bh,
                        "B365D": bd,
                        "B365A": ba,
                        "gd_pred": 0.0,
                        "target_gd": gd,
                    }
                )
    finally:
        tmp_path.unlink(missing_ok=True)
    if not out:
        raise ValueError(f"No valid rows loaded for season {season!r} (check URL or CSV columns).")
    return out
