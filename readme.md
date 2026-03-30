# Axiom Engine

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/Danny1218/Axiom)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**A Differentiable Neural Architecture Search (DNAS) compiler.** Write explicit symbolic rules, compile them into a continuous-time neural network, and let the AI evolve to handle the edge cases.

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

## Links

- **Repository:** [github.com/Danny1218/Axiom](https://github.com/Danny1218/Axiom)  
- **Tests:** `python -m pytest tests -q`  
- **Project state (maintainers):** see `plan.md` in this repo.
