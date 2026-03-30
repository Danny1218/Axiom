# Axiom Engine

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/Danny1218/Axiom)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**A Differentiable Neural Architecture Search (DNAS) compiler.** Write explicit symbolic rules, compile them into a continuous-time neural network, and let the AI evolve to handle the edge cases.

---

## Understanding Axiom (the layman’s bridge)

Modern software sits between two broken extremes:

| **Traditional code** | **Standard AI (deep nets / LLMs)** |
|----------------------|-------------------------------------|
| You write strict rules (`if X > 5 then Y`). Reliable and obedient—but the real world is messy; when data doesn’t fit, logic fails hard. | You feed a huge black box millions of examples and hope it infers the rules. Great on messy data—but it can hallucinate; you can’t *force* it to obey physics or policy. |

**Axiom is the hybrid path:** a **symbolic–neural** engine. You write normal-looking rules in an **`.ax`** file (the **symbolic** skeleton). The compiler then **wraps** a fluid, trainable neural graph around that skeleton (the **neural** reflexes)—same program, one differentiable stack.

**Metaphor — self-driving:** Traditional code is a strict lane line: line disappears → crash. Pure AI is a blindfolded learner: crash 10,000 times and hope for intuition. **Axiom** is **GPS + reflexes**: the map is your explicit code (where the car *must* go); the nets learn potholes and edge cases the map never listed.

---

## Why Axiom?

Axiom is a **hybrid symbolic–neural** system: you program in a small language (`.ax` files) that looks like JavaScript; the compiler lowers `if`, `else`, and `while` into **differentiable** PyTorch graphs. Symbolic paths encode what you *know*; **Tensor-Train LoRA (TT-LoRA)** adapters on a shared trunk learn what you *don’t*.

### Core features

| | |
|--|--|
| **Hybrid execution** | Hardcoded logic runs as interpreted IR on the latent trunk; TT-LoRA experts learn probabilistic residuals so the net still fits data when symbolic rules are wrong or incomplete. |
| **Dynamic routing (MoE)** | `if` / `else` compiles to **Sinkhorn**-balanced mixture-of-experts routing. **MetaCompiler** can **unmask** shadow experts when router entropy signals high uncertainty—new capacity appears only when needed. |
| **Continuous memory** | `while` loops become **Liquid Kolmogorov–Arnold Networks (KANs)** over unrolled timesteps, with **high-dimensional RBF splines**—a differentiable alternative to stacking static RNN cells for sequence-shaped IR. |
| **The Glass Box** | The stack is **interpretable by design**: launch a **Streamlit** dashboard to see the graph, ABI variables, and routing weights evolve—not a black box. |

### Why it’s different from “just scale the Transformer”

Much of the industry optimizes **scaling laws**—bigger GPUs, bigger black boxes. Axiom optimizes **structure** instead:

- **Zero-shot human knowledge:** Encode facts and invariants in code so the model doesn’t waste capacity re-learning basics; nets focus on **residuals** and messy regions.
- **Glass-box audits:** Because execution sits on top of your IR, **`axiom inspect`** lets you see **which branch** was taken and **how much** neural routing moved the needle—valuable in regulated settings.
- **Physical growth (DNAS):** The graph can **raise** capacity when uncertain—Sinkhorn entropy feeds **MetaCompiler**, which can **unmask** shadow LoRA experts instead of freezing a single static brain.
- **Time as continuity:** Loops compile to **Liquid KANs** with **RBF splines** over unrolled time—closer to a continuous dynamical view than a purely discrete tick-tock RNN story (see `examples/sequence.ax`).

### What is an `.ax` file?

An **`.ax`** file is source code for Axiom: assignments, comparisons, **`if` / `else`**, and **`while`** with C/JavaScript-like syntax. The parser builds an AST → IR bytecode (`OP_ASSIGN`, `OP_CONDITIONAL`, `OP_LOOP`, …) → an **execution graph** (`ExecutionGraph`) of PyTorch modules. You train that graph like any other model; gradients flow through both symbolic and neural pieces where the IR is differentiable.

---

## Installation

```bash
git clone https://github.com/Danny1218/Axiom.git
cd Axiom
pip install -e .
```

This installs the **`axiom-engine`** package and the global **`axiom`** CLI. Requires **Python 3.10+** and **PyTorch 2+**.

---

## Quickstart — Native Python API (Jupyter / scripts)

Load a trained **`.axb`** bundle and run inference with ordinary Python dicts—no manual trunk layout. Batches are a **list of dicts**; **pandas** `DataFrame` rows work too (optional: install `pandas` separately).

```python
import axiom

model = axiom.load("examples/portfolio_trained.axb")  # after training
out = model.predict(
    {"volatility": 0.6, "drawdown": 0.1, "momentum": -0.8, "volume": 1.5}
)
# out is a dict of ABI names → floats (or lists for vector columns)

batch = model.predict(
    [
        {"volatility": 0.5, "drawdown": 0.0, "momentum": 0.1, "volume": 1.0},
        {"volatility": 0.6, "drawdown": 0.1, "momentum": -0.8, "volume": 1.5},
    ]
)

# Optional: entire DataFrame (same column names as ABI inputs)
# import pandas as pd
# market = pd.read_csv("spy_daily.csv")
# preds = model.predict(market)
```

---

## Quickstart 1 — Tabular crucible (Titanic)

The bundled `examples/titanic.ax` uses a **deliberate sabotage** rule (impossible Fare threshold) so symbolic logic alone is useless—the hybrid stack must learn from data. Your own programs use the same **`if` / `else`** shape (e.g. branching on `Sex`, `Pclass`, etc.).

```javascript
// examples/titanic.ax (excerpt)
if (Fare > 100000.0) {
  survived_prob = 1.0;
} else {
  survived_prob = 0.0;
}
```

Train on the built-in Titanic dataset (CSV is downloaded if missing), 80/20 split, then **test accuracy** on the holdout set:

```bash
axiom train examples/titanic.ax --dataset titanic --epochs 30
```

Optional: `--dim 32`, `--no-meta`, `--out my_bundle`, `--titanic-csv path/to.csv`.

---

## Quickstart 2 — Sequence crucible (sine wave)

`examples/sequence.ax` drives a **Liquid-KAN** loop: a prelude seeds the ABI (including `y_pred = x * 0.0` so `x` is not clobbered), then a **`while`** integrates `step` for 10 iterations—compiled to a fixed-unroll sequence fed to the KAN.

```javascript
// examples/sequence.ax (excerpt)
y_pred = x * 0.0;
step = 0.0;
while (step < 10.0) {
  step = step + 1.0;
}
```

```bash
axiom train examples/sequence.ax --dataset sine --epochs 30 --dim 32
```

`--dim 32` matches a comfortable trunk width for this example; you will see **test MSE** on the synthetic `sin(x)` task printed after training.

---

## Live SPY — neuro-symbolic trading (optional)

Install extras: **`pip install -e ".[spy]"`** (**pandas**, **yfinance**). **`examples/spy_alpha.ax`** trains a **`neural([momentum_1d, momentum_5d, volatility])`** signal on real **SPY** history, but **symbolically** forces **`prediction = 0.0`** (cash) when **daily** **`(High − Low) / Open > 2.5%`**. Run:

```bash
python examples/train_spy.py
```

This downloads ~6y of data, trains 50 epochs, writes **`examples/spy_trained.axb`**, reloads it with **`axiom.load`**, runs **`model.predict`** on the held-out last **500** trading days, and prints **cumulative strategy return** (long / short / cash from the sign of the prediction) vs **buy-and-hold**.

---

## The Glass Box visualizer

After training, artifacts are written as **`{prefix}.pt`** + **`{prefix}_topology.json`** (default prefix `axiom_bundle`).

```bash
axiom inspect
```

This starts **Streamlit**. In the UI, set the bundle path prefix (same as `--out` without extension), adjust ABI inputs if needed, and run inference. Expand **routing / signals** to watch **Sinkhorn weights** and entropy-style diagnostics shift as different inputs traverse **conditional** blocks—your “aha!” moment for how symbolic branches became continuous routing.

---

## More CLI (cheat sheet)

| Goal | Command |
|------|--------|
| Legacy synthetic sequence (no CSV) | `axiom train train.ax --epochs 10 --out axiom_bundle` |
| Custom CSV | `axiom train my.ax --csv data.csv --target_key label --target_var my_output_abi` |
| Load saved bundle, one-off inference | `axiom train --mode inference --out axiom_bundle` |

---

## Compiler pipeline (30 seconds)

1. **Parse** `.ax` → AST  
2. **Lower** → IR (`OP_*` bytecode)  
3. **Wire** → `ExecutionGraph` (NetworkX + PyTorch)  
4. **Train** with `EvolutionaryTrainer` / `AxiomDataset`; **inspect** with `AxiomRunner` + Glass Box  

---

## Philosophy

Brute-force scaling hits walls; **structure** scales. Axiom gives you an algorithmic skeleton you can read and audit, and lets **DNAS-style** sparsity and meta-compilation grow the right neural “muscle” where uncertainty demands it—on hardware you already have.

---

## Where Axiom shines (example domains)

These are **illustrative**—not shipped products—but they match the design center: **hard constraints in code**, **learning in the gaps**.

| Domain | Symbolic (your rules) | Neural (adapters / KAN) |
|--------|------------------------|---------------------------|
| **Trading / risk** | Hard limits (`if loss > threshold then flatten`) | Momentum, microstructure, regime patterns |
| **Med / biotech** | Physiological impossibilities, contraindications | Subtle biomarker correlations inside guardrails |
| **Robotics** | Safety envelopes, no-go zones | Smooth motion, efficiency inside constraints |
| **Games / sims** | Gravity, collision, authored laws | NPCs, weather, adaptive behavior without breaking physics |

---

## Road ahead

Three honest forks after v1.0:

| Path | Idea | Trade-off |
|------|------|-----------|
| **A — Killer app** | Stop extending the compiler; ship a **domain vertical** (trading, sports, weather) that proves ROI. | Proves value; less time on core R&D. |
| **B — Language** | Grow **`.ax`** toward **functions, arrays, classes** (Turing-complete, reusable modules). | Huge compiler/graph-design lift; long horizon. |
| **C — Community** | **PyPI**, articles, tutorials, issues—grow users and contributors. | Recognition and help; maintainer time on support and docs. |

**Pragmatic default:** drive **Path A** once—pick a dataset you care about, beat or match a baseline, *then* invest in B or C with evidence.

---

## Links

- **Repository:** [github.com/Danny1218/Axiom](https://github.com/Danny1218/Axiom)  
- **Tests:** `python -m pytest tests -q`  
- **Project state (maintainers):** see `plan.md` in this repo.
