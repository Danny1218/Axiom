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

This installs the **`axiom-engine`** package and the global **`axiom`** CLI. Requires **Python 3.10+** and **PyTorch 2+**. Core dependencies are **torch**, **lark**, and **networkx** only.

Optional extras:

| Extra | Purpose |
|-------|---------|
| **`[inspect]`** | Glass Box (`axiom inspect`): Streamlit + `graphviz` bindings |
| **`[serve]`** | HTTP bundle API (`axiom serve`): FastAPI + uvicorn |
| **`[lock]`** | Genetic lock on **`.axb`** neural weights (`axiom lock-bundle`) |
| **`[export]`** | ONNX export (`axiom export-onnx`) |
| **`[gateway]`** | Policy gateway HTTP + examples (`requests`, Streamlit, overlaps `[serve]` on FastAPI) |
| **`[copilot]`** | Semantic copilot CLI (`axiom copilot-draft`, `axiom copilot-search`) — `requests` for Onyx/Qwen-style chat APIs |
| **`[dev]`** | Run the test suite (`pytest` + Glass Box deps for `inspect` / `glass_box` tests) |

Run tests locally:

```powershell
pip install -e ".[dev]"
python -m pytest tests -q
```

---

## From compile to production

1. **Compile & train** — **`axiom train`** on an **`.ax`** file; you get a **`.axb`** bundle (serialized `InterpretedBlock` + weights).  
2. **Bundle** — The **`.axb`** is the portable artifact; load it with **`axiom.load`** or **`axiom predict`**.  
3. **Serve** — Optional **`pip install -e ".[serve]"`**, then **`axiom serve`** exposes **`/health`**, **`/predict`**, **`/explain`**, **`/report`** over HTTP.  
4. **Secure** — Optional **`pip install -e ".[lock]"`**, then **`axiom lock-bundle`** encrypts neural weights; **`AXIOM_BUNDLE_SECRET`** / device unlock at load time.  
5. **Export** — Optional **`pip install -e ".[export]"`**, then **`axiom export-onnx`** for inference-only ONNX (no **`explain`** parity).  
6. **Policy gateway** — Optional **`pip install -e ".[gateway]"`**, then **`axiom gateway-serve`** for **`POST /gateway/chat`** (scan + explain + allow/deny + optional downstream forward).
7. **Semantic copilot** — Optional **`pip install -e ".[copilot]"`**, then **`axiom copilot-draft`** / **`axiom copilot-search`** to draft and repair **`.ax`** programs via an OpenAI-compatible chat endpoint (e.g. Onyx + Qwen).

---

## Semantic copilot CLI

Install **`[copilot]`** so `requests` is available. Pass **`--expert-url`** (API base, e.g. `https://your-host/v1/`), **`--expert-model`**, and optionally **`--expert-api-key`** or set **`AXIOM_EXPERT_API_KEY`**. Iteration logs and “wrote file” lines go to **stderr**; the generated **`.ax`** source is printed to **stdout** so you can redirect it.

**Reproducible runs:** **`axiom copilot-search --artifact-dir path/to/run/`** writes a fixed bundle — **`best.ax`**, **`iterations.json`** (per-iteration source, metrics, failure summaries, expert **`metadata`**), **`search_report.json`** (run header + **`failures_metrics_summary`**) — only when that flag is set (no silent writes).

**Draft** (goal → single program):

```powershell
pip install -e ".[copilot]"
axiom copilot-draft --backend onyx-qwen --goal "Binary classifier with neural([a,b]) output survived_prob" `
  --context "Titanic-style features" `
  --expert-url "https://api.example.com/v1/" --expert-model "qwen-7b" `
  --out drafted.ax
```

**Search** (draft → compile/evaluate → repair loop; optional row eval JSON):

Row file format (JSON array): each element is `{"inputs": {...}, "expected": {...}}` for **`predict_rows`** scoring (default metric: **`neg_mse`**, higher is better). With **`--compile-only`**, examples are still passed into the expert context but evaluation stays compile-only.

```powershell
$examples = @'
[{"inputs": {}, "expected": {"y": 0.5}}]
'@
Set-Content -Path examples.json -Value $examples -Encoding utf8
axiom copilot-search --backend onyx-qwen --goal "Output y from defaults" `
  --expert-url "https://api.example.com/v1/" --expert-model "qwen-7b" `
  --iterations 5 --examples-json examples.json `
  --out best.ax --report-out search_report.json --artifact-dir ./copilot_run_01
```

---

## `axiom serve`

Serves **one** `.axb` at startup via FastAPI: **`GET /health`**, **`POST /predict`**, **`POST /explain`**, **`POST /report`** (JSON **`inputs`**; report can return inline HTML). Install **`pip install -e ".[serve]"`**.

Optional **`AXIOM_API_KEY`**: mutating routes accept **`Authorization: Bearer …`** or **`X-API-Key`**; **`GET /health`** is unauthenticated. **`AXIOM_BUNDLE_PATH`** selects the bundle if **`--bundle`** is omitted.

**Examples:**

```powershell
pip install -e ".[serve]"
axiom serve --bundle examples/portfolio_trained.axb --host 127.0.0.1 --port 8000
```

```powershell
$env:AXIOM_BUNDLE_PATH = "examples/portfolio_trained.axb"
$env:AXIOM_API_KEY = "secret"
axiom serve --host 0.0.0.0 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"inputs\": {\"volatility\": 0.6}}"
```

---

## Locked bundles

**Genetic lock** (`src/axiom/security/genetic_lock.py`) optionally encrypts **serialized neural weights** inside the **`.axb`** with AES-256-CTR; **topology / ABI / IR** stay readable. Modes include **`device`** (CUDA identity), **`host`**, and **`env-secret`** ( **`AXIOM_BUNDLE_SECRET`** ). Install **`pip install -e ".[lock]"`**.

**Examples:**

```powershell
pip install -e ".[lock]"
$env:AXIOM_BUNDLE_SECRET = "dev-secret"
axiom lock-bundle --input examples/portfolio_trained.axb --output examples/portfolio_locked.axb --mode env-secret
axiom predict --bundle examples/portfolio_locked.axb --input '{"volatility":0.6,"drawdown":0.1,"momentum":-0.8,"volume":1.5}'
```

In Docker, set **`AXIOM_BUNDLE_SECRET`** if the mounted bundle is **`env-secret`** locked (see **Docker deployment**).

---

## Docker deployment

Production-style image for **`axiom serve`**: one **`.axb`**, FastAPI on **`HOST`** / **`PORT`**. The **`Dockerfile`** installs **`pip install ".[serve,lock]"`**. Optional env: **`AXIOM_API_KEY`** (Bearer / **`X-API-Key`** on **`/predict`**, **`/explain`**, **`/report`**), **`AXIOM_BUNDLE_SECRET`** (unlock **`env-secret`** locked bundles). No bundle is baked in—set **`AXIOM_BUNDLE_PATH`** at runtime.

### Build

```bash
docker build -t axiom-engine:latest .
```

**Building only creates the image.** Nothing listens on **`8000`** until you **run** a container (next section) or **`docker compose up`**.

### Run (start the server)

Mount a local **`.axb`** and set **`AXIOM_BUNDLE_PATH`** inside the container to match the mount path:

```bash
docker run --rm -p 8000:8000 \
  -e AXIOM_BUNDLE_PATH=/bundle/model.axb \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -v /absolute/path/to/model.axb:/bundle/model.axb:ro \
  axiom-engine:latest
```

**PowerShell** (repo root; uses **`examples/portfolio_trained.axb`** after **`python examples/train_portfolio.py`**):

```powershell
docker run --rm -p 8000:8000 `
  -e AXIOM_BUNDLE_PATH=/bundle/model.axb `
  -e HOST=0.0.0.0 `
  -e PORT=8000 `
  -v "${PWD}/examples/portfolio_trained.axb:/bundle/model.axb:ro" `
  axiom-engine:latest
```

Leave that terminal open while testing. Add optional **`AXIOM_API_KEY`** / **`AXIOM_BUNDLE_SECRET`** with **`-e`** as needed.

### Compose

Create **`bundles/`**, copy or symlink your **`model.axb`** there (compose expects **`/bundles/model.axb`**), then:

```bash
docker compose up --build
```

**Note:** the default **`docker-compose.yml`** sets **`AXIOM_API_KEY=change-me-in-production`**, so **`POST /predict`** requires **`Authorization: Bearer …`** (see below). **`GET /health`** stays unauthenticated.

### Example `curl`

In another terminal, after the container is running:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": {}}'
```

When **`AXIOM_API_KEY`** is set (e.g. **`change-me-in-production`** in **`docker-compose.yml`**):

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{"inputs": {}}'
```

---

## ONNX export

InterpretedBlock **`.axb`** only: dense tensor in/out via **`torch.onnx.export`**; **inference-only**—it does **not** preserve **`explain`** / Glass Box semantics, and some IR graphs may fail to export.

```powershell
pip install -e ".[export]"
axiom export-onnx --bundle examples/portfolio_trained.axb --output examples/portfolio.onnx --opset 17
```

Optional round-trip tests use **`onnxruntime`** (not installed by **`[export]`**).

---

## Policy gateway

**`pip install -e ".[gateway]"`** pulls **`requests`**, Streamlit (for **`examples/enterprise_ui.py`**), FastAPI, and uvicorn. **`axiom.gateway`** scans text (or accepts pre-extracted signals), runs **`AxiomModel.explain`**, blocks or forwards to a downstream URL, and can emit Glass Box HTML on deny via **`export_report`** / **`render_html_report`**.

**HTTP server:**

```powershell
axiom gateway-serve --bundle policy.axb --downstream-url http://127.0.0.1:8000/api/chat --policy-source examples/enterprise_policy.ax --host 127.0.0.1 --port 8010
```

**`POST /gateway/chat`** accepts JSON **`message`** and optional **`signals`**. Examples **`examples/onyx_gateway.py`** and **`examples/enterprise_ui.py`** use **`default_scan_text`** from **`axiom.gateway.core`**.

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

model.export_report({"volatility": 0.6, "drawdown": 0.1, "momentum": -0.8, "volume": 1.5}, "report.html")

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

Install extras: **`pip install -e ".[spy]"`** (**pandas**, **yfinance**). **`examples/spy_alpha.ax`** feeds **six** features into **`neural(...)`** (momentum, daily range vol, **SMA 10/50** divergence vs price, **20d** return volatility), and **symbolically** forces **`prediction = 0.0`** (cash) when **daily** **`(High − Low) / Open > 2.5%`**. Training swaps in a **deeper custom PyTorch stack** (via **`custom_neural_registry`**) instead of the default tiny MLP. Run:

```bash
python examples/train_spy.py
```

This downloads ~6y of data, trains 50 epochs, writes **`examples/spy_trained.axb`**, reloads with **`axiom.load(..., custom_neural_registry=...)`** (same architecture as training), runs **`model.predict`** on the held-out last **500** trading days, and prints **cumulative returns**, **annualized Sharpe ratios**, and **max drawdowns** (strategy vs buy-and-hold)—better risk-aware readouts than raw return alone. It ends with an **Autopsy** on the worst single-day strategy loss: **`model.explain({...})`** dumps **`alpha_signal`**, **`prediction`**, and inputs so you can audit why the model traded.

For any **`.axb`**, **`model.explain({"feature": ...})`** returns a JSON-friendly dict of symbolic variable values after one forward pass (Phase 41).

---

## The Glass Box visualizer

After training, artifacts are written as **`{prefix}.pt`** + **`{prefix}_topology.json`** (default prefix `axiom_bundle`).

```powershell
pip install -e ".[inspect]"
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
| HTTP bundle API (needs **`[serve]`**) | `axiom serve --bundle model.axb` |
| Lock weights (needs **`[lock]`**) | `axiom lock-bundle --input in.axb --output out.axb --mode env-secret` |
| ONNX (needs **`[export]`**) | `axiom export-onnx --bundle model.axb --output model.onnx` |
| Policy gateway (needs **`[gateway]`**) | `axiom gateway-serve --bundle policy.axb --downstream-url https://…` |

---

## Compiler pipeline (30 seconds)

1. **Parse** `.ax` → AST  
2. **Lower** → IR (`OP_*` bytecode)  
3. **Wire** → `ExecutionGraph` (NetworkX + PyTorch)  
4. **Train** → save **`.axb`**; **serve** / **lock** / **export** / **gateway** as optional deployment steps (see **From compile to production**)  
5. **Inspect** with **`axiom inspect`** (`[inspect]`) or **`AxiomRunner`** + Glass Box  

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
- **Tests:** `pip install -e ".[dev]"` then `python -m pytest tests -q`  
- **Project state (maintainers):** see `plan.md` in this repo.
