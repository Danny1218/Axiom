"""Built-in datasets for `axiom train --dataset` (Phase 25+)."""

from __future__ import annotations

import math
import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

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
