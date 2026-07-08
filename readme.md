# Axiom Engine

[![Version](https://img.shields.io/badge/version-1.4.0-blue.svg)](https://github.com/Danny1218/Axiom)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**A Differentiable Neural Architecture Search (DNAS) compiler.** Write explicit symbolic rules, compile them into a continuous-time neural network, and let the AI evolve to handle the edge cases.

---

## Benchmarks: where Axiom wins (and where it doesn't)

Reproducible CPU benchmarks (fixed seeds, no network) live under `benchmarks/` and write committed evidence to `docs/evidence/`. Install extras: `pip install -e ".[bench]"`.

### Guarded Titanic — wrap the strong model, prove the clamp

**Claim:** Call your best tabular model via `expert()` and keep symbolic safety rules in `.ax`; holdout accuracy matches the wrapped model while constraints are machine-checkable.

```powershell
python benchmarks/titanic_hybrid/run_guarded_audit.py
```

| Model | Holdout accuracy | Rule violations (500 edge cases) |
|-------|------------------|----------------------------------|
| **Raw GradientBoosting** | ~0.85 | ~128 |
| **Guarded GBM (Axiom wrap)** | ~0.85 (Δ≈0) | **0** |
| v1.3 pure hybrid (neural) | ~0.63 | 0 |

Certificate excerpt from `docs/evidence/titanic_guarded_certificate.json`:

```json
{
  "input_region": {"Pclass": [3.0, 3.0], "Sex": [0.0, 0.0], "Age": [18.0, 100.0]},
  "assumptions": {"tabular_model": [0.0, 1.0]},
  "proven_output_bounds": {"survived_prob": [0.0, 0.15]}
}
```

### Guardrail pattern

1. Train or load a strong baseline (sklearn, custom Python, etc.).
2. Wrap in `.ax`: `raw = expert("your_model", features);` plus symbolic `if` / `min` / `max` rules.
3. Wire the handler at runtime (`ExpertRuntimeRegistry`) or in `axiom serve`.
4. Run **`axiom certify`** with input region + declared expert bounds → JSON safety certificate.

### Extrapolation showdown (symbolic recovery vs sklearn)

**Claim:** When labels come from a known formula plus modest noise, Axiom's **tolerant symbolic inference** recovers the closed form and extrapolates outside the training range.

```powershell
python benchmarks/baseline_showdown/run_showdown.py
```

| Result | Detail |
|--------|--------|
| **Wins** | **9/10** in-family extrapolation wins (`docs/evidence/baseline_showdown.md`) with **unclipped** 3% Gaussian noise |
| **Honest loss** | `clamped_affine_three` declines — noisy labels vs clamp family |
| **Sabotage** | `sin(x)` and `exp(-x)`: Axiom **declines** (2/2) |
| **Gates (v1.4)** | Scale-relative row tolerance + 5% relative RMSE; benchmark noise clipping removed |

### Titanic hybrid (v1.3 reference)

v1.3 neural hybrid (`examples/titanic_hybrid.ax`) enforced constraints but lagged GBM accuracy (~0.63). v1.4 **`titanic_guarded.ax`** wraps GBM instead (see above). Reproduce v1.3 numbers:

```powershell
python benchmarks/titanic_hybrid/run_hybrid_audit.py
```

---

## 60-second quickstart

```powershell
git clone https://github.com/Danny1218/Axiom.git
cd Axiom
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,copilot,serve]"

# Train hybrid Titanic example → bundle
axiom train examples/titanic.ax --dataset titanic --epochs 10 --out titanic_bundle

# Serve and predict (local dev; set AXIOM_ALLOW_INSECURE_SERVE=1)
$env:AXIOM_ALLOW_INSECURE_SERVE="1"
axiom serve --bundle titanic_bundle.axb --port 8010
axiom predict --bundle titanic_bundle.axb --inputs '{"Sex":0,"Pclass":3,"Fare":7.25,"Age":22}'

# Semantic copilot (local LM Studio with qwen/qwen3-8b on :1234)
axiom copilot-doctor --backend lmstudio
```

---

## What works / what's experimental

| Area | Status | Notes |
|------|--------|-------|
| **`.ax` → IR → train → `.axb`** | Stable | `axiom train`, `axiom load`, `AxiomModel.predict/explain` |
| **HTTP bundle serve** | Stable | `axiom serve` with optional API key |
| **Exact symbolic fast paths** | Stable | Deterministic draft for common affine / min-max families |
| **Tolerant symbolic inference (v1.2)** | Stable | Least-squares fit on noisy rows before LLM; no network required for robustness tasks |
| **LLM copilot (LM Studio / Onyx)** | Stable | Draft → evaluate → repair; output normalizer canonicalizes almost-valid `.ax` |
| **Offline benchmark gate** | Stable | `axiom copilot-benchmark --backend benchmark-dispatch` on four JSON suites |
| **`torch.compile(fullgraph=True)`** | Experimental | Some builds cannot trace strict-mode `ContextVar`; tests skip or use hoisted bools |
| **MetaCompiler / DNAS unmasking** | Experimental | Sinkhorn routing + shadow expert growth |
| **Policy gateway / ONNX export** | Optional extras | `[gateway]`, `[export]` — less CI coverage than core path |

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

**External expert hook (in-program):** `expert("backend_name", feature_expr)` lowers to **`OP_EXPERT`** (not `neural()`). It is **not differentiable**. At runtime, pass **`InterpretedBlock(..., expert_handler=callable)`** or **`expert_fallback=float`**; otherwise execution raises a clear error. The handler receives `(backend_name, list[float])` for one batch row and must return a **scalar float**. **`model.explain(...)`** adds an **`expert_calls`** list (backend names used). **ONNX export** rejects bundles that contain `expert()`.

---

## Installation

```bash
git clone https://github.com/Danny1218/Axiom.git
cd Axiom
pip install -e .
```

This installs the **`onyx-axiom`** package (importable as `axiom`, also available on PyPI: `pip install onyx-axiom`) and the global **`axiom`** CLI. Requires **Python 3.10+** and **PyTorch 2+**. Core dependencies are **torch**, **lark**, and **networkx** only.

Optional extras:

| Extra | Purpose |
|-------|---------|
| **`[inspect]`** | Glass Box (`axiom inspect`) + Copilot Studio (`axiom copilot-studio`): Streamlit + `graphviz` bindings |
| **`[serve]`** | HTTP bundle API (`axiom serve`): FastAPI + uvicorn |
| **`[lock]`** | Genetic lock on **`.axb`** neural weights (`axiom lock-bundle`) |
| **`[export]`** | ONNX export (`axiom export-onnx`) |
| **`[gateway]`** | Policy gateway HTTP + examples (`requests`, Streamlit, overlaps `[serve]` on FastAPI) |
| **`[copilot]`** | Semantic copilot CLI (`axiom copilot-draft`, `axiom copilot-search`, `axiom copilot-run`) — `requests` for Onyx/Qwen-style chat APIs |
| **`[dev]`** | Run the test suite (`pytest` + Glass Box deps for `inspect` / `glass_box` tests) |
| **`[bench]`** | CPU baseline benchmarks (`scikit-learn` for sklearn baselines in `benchmarks/`) |

Run tests locally (CI-parity install):

```powershell
pip install -e ".[dev,copilot,serve]"
pip install -r constraints-dev.txt
python -m pytest tests -q
```

For a minimal dev-only install, `pip install -e ".[dev]"` still works but skips serve/copilot integration tests.

### Continuous integration

| Workflow | Platform | What it runs |
|----------|----------|--------------|
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Ubuntu, Python 3.10–3.12 | `python -m pytest tests -q`, plus targeted `test_copilot_golden.py` and `test_smoke_happy_path.py` |
| [`.github/workflows/copilot-milestone.yml`](.github/workflows/copilot-milestone.yml) | Windows, Python 3.12 | `pytest -q`, `scripts/smoke_copilot_draft.ps1`, four offline `axiom copilot-benchmark --backend benchmark-dispatch` suites |

**Execution overhead baseline (local diagnostic, not CI-gated):**

```powershell
python scripts/profile_execution_overhead.py --json-out debug_execution_overhead.json
```

Compare JSON before/after changes; entries include p50/p95/mean timings and `torch_compile_backend` with per-backend `errors`.

---

## From compile to production

1. **Compile & train** — **`axiom train`** on an **`.ax`** file; you get a **`.axb`** bundle (serialized `InterpretedBlock` + weights).  
2. **Bundle** — The **`.axb`** is the portable artifact. Unlocked neural bundles also write a **sidecar** **`model.axb.weights.pt`** — copy **both** files together when moving or deploying (locked bundles embed ciphertext in the JSON instead).  
3. **Serve** — Optional **`pip install -e ".[serve]"`**, then **`axiom serve`** exposes **`/health`**, **`/predict`**, **`/explain`**, **`/report`** over HTTP. Production binds (**`0.0.0.0`**, Docker) require **`AXIOM_API_KEY`** unless **`AXIOM_ALLOW_INSECURE_SERVE=1`** (local dev only).  
4. **Secure** — Optional **`pip install -e ".[lock]"`**, then **`axiom lock-bundle`** encrypts neural weights; **`AXIOM_BUNDLE_SECRET`** / device unlock at load time.  
5. **Export** — Optional **`pip install -e ".[export]"`**, then **`axiom export-onnx`** for inference-only ONNX (no **`explain`** parity).  
6. **Policy gateway** — Optional **`pip install -e ".[gateway]"`**, then **`axiom gateway-serve`** for **`POST /gateway/chat`** (scan + explain + allow/deny + optional downstream forward).
7. **Semantic copilot** — Optional **`pip install -e ".[copilot]"`**, then **`axiom copilot-draft`** / **`axiom copilot-search`** / **`axiom copilot-run`** to draft, repair, or run the full “goal → best **`.ax`** + reports” pipeline via an OpenAI-compatible chat endpoint (e.g. Onyx + Qwen). **`axiom copilot-stability-report`** (Phase 83 / 83b) recursively scans local artifact trees and pipeline **`*.json`** summaries — no network — to summarize sweep or multi-restart stability.

---

## Semantic copilot CLI

Install **`[copilot]`** so `requests` is available. For **local development**, start [LM Studio](https://lmstudio.ai/) with **`qwen/qwen3-8b`** on **`http://127.0.0.1:1234`**, then use **`--backend lmstudio`** (defaults: URL **`http://127.0.0.1:1234/v1/`**, model **`qwen/qwen3-8b`**, no API key). Remote/on-prem endpoints use **`--backend onyx-qwen`** with **`--expert-url`**, **`--expert-model`**, and optionally **`--expert-api-key`** or **`AXIOM_EXPERT_API_KEY`**.

**Qwen3 thinking blocks:** the backend strips `` spans before code extraction (metadata **`stripped_think_block: true`**) and sends **`enable_thinking: false`** plus a **`/no_think`** suffix on user prompts when the model id contains **`qwen`**.

**Smoke check:** **`axiom copilot-doctor`** (default **`--backend lmstudio`**) requests greedy sampling via internal **`temperature: 0`** → HTTP **`do_sample: false`**. It prints **`connection:`**, **`parse` / `ir` / `block`**, and optional example-row diagnostics. Use **`--validate-source path.ax`** to check local **`.ax`** without calling the expert.

```powershell
# Local LM Studio (default backend)
axiom copilot-doctor

# Remote OpenAI-compatible server
axiom copilot-doctor --backend onyx-qwen --expert-url "https://your-host/v1/" --expert-model "qwen-7b"
```

The **`onyx-qwen`** backend’s system prompts and response parsing are tuned for **this repository’s** **`.ax`** DSL (JavaScript-like **`=`** assignments, **`;`**-terminated statements, **`neural(...)`**, **`if`/`while`**): they explicitly disambiguate from Macaulay2, the Axiom computer algebra system, and generic pseudocode. Repair output is expected to be program text only; iteration artifacts record extraction metadata (e.g. **`extraction_mode`**, **`forbidden_tokens_detected`** when **`:=`** or **`print(`** slip through, **`stripped_language_tag`** / **`code_line_count`** when a stray fence language line was removed from the model text) for inspection. When **`--examples-json`** (or HTTP **`examples`**) supplies rows, draft/repair user prompts require **one** reusable program over the real variable names (**`x`**, **`y`**, …), forbid **row-indexed** names (**`x_0`**, **`y_1`**, …) and **`output(...)`** (not in this DSL), and ask for generalization beyond the sample rows. If the model still emits indexed names or **`output(`**, repair prompts add **collapse** hints; **`metadata`** may include **`indexed_variable_warning`** / **`output_call_warning`** without mutating the source text.

**Reproducible runs:** **`axiom copilot-search --artifact-dir path/to/run/`** writes a fixed bundle — **`best.ax`**, **`iterations.json`** (per-iteration source, metrics, failure summaries, expert **`metadata`**, optional **`semantic_trace_summary`**), **`search_report.json`** (run header + **`failures_metrics_summary`** + sibling **`semantic_summaries`** when summarization ran) — only when that flag is set (no silent writes).

**End-to-end pipeline (Phase 71):** **`axiom copilot-run`** runs the same draft→evaluate→repair loop as **`copilot-search`**, then optionally writes a **pipeline summary JSON** (**`--summary-out`**) with a short **disclaimer** (this path does **not** train **`.axb`** or export ONNX), runs an extra **compile-only** pass on the champion source by default (disable with **`--no-final-validate`**), and prints explicit **stderr** lines if that final validation fails. Use **`--artifact-dir`**, **`--out`**, and the same mode flags as search (**`--compile-only`**, **`--examples-json`**, **`--train-tabular`** / **`--tabular-json`**, **`--summarize-traces`**).

**Multi-restart best-of-N (Phase 80):** **`axiom copilot-run --restarts N`** (default **1**) runs **`N`** independent full searches and keeps the overall best candidate using the same ranking as within-search (including **`adjusted_sort_score`** when applicable). The pipeline summary JSON includes **`restarts`** (**`total`**, **`winning_index`**, **`per_restart`**). With **`--artifact-dir`** and **`N > 1`**, each run writes under **`restart_0/`**, **`restart_1/`**, …; **`POST /run`** accepts **`restarts`** and returns **`restarts_total`**, **`winning_restart_index`**, **`per_restart_summaries`**.

**Stability report (Phase 83 / 83b):** **`axiom copilot-stability-report`** recursively scans each path and **`--parent`** root for **`search_report.json`**, pipeline-style **`*.json`** (e.g. **`risk_score_run_1.json`** beside **`--summary-out`** from **`copilot-run`**), and multi-restart directories (**`restart_0/`** …). **`search_report.json`** files only under **`restart_*/`** are not separate runs (they belong to the parent bundle). Pipeline summary JSON that references **`artifact_dir`** dedupes against the same artifact directory so you are not double-counted. Prints exact/near-hit counts (for **`neg_mse`** vs **`--near-threshold`**, default **-1e-9**), convergence-reason tallies, winning-restart tallies, and the overall best **`best.ax`** path. Use **`--json-out report.json`** for machine-readable output (includes **`discovery`** stats). No **`[copilot]`** dependency required (filesystem only).

**Sampling controls (Phase 82 / 82b):** optional **`--temperature`** and **`--top-p`** on **`copilot-search`**, **`copilot-run`**, and **`copilot-doctor`** forward OpenAI-style parameters to the expert for **both** draft and repair calls (merged into an internal context key, not into the visible “Context (JSON)” prompt text). Omitted flags preserve prior behavior (no extra JSON fields on the HTTP payload). **`temperature` of 0 or lower** is sent to Onyx as **`do_sample: false`** with **`temperature` omitted** (and **`top_p` omitted** when greedy, to avoid conflicting sampling knobs). **`POST /search`** and **`POST /run`** accept **`temperature`** and **`top_p`** in the JSON body. **Copilot Studio:** optional sidebar fields for the same (parsed when non-empty).

**Trace summaries (optional):** pass **`--summarize-traces`** on **`copilot-search`** to call the expert’s **`summarize_trace`** after each iteration (natural-language narration of explain trace + metrics + failures). Default is off; if the call fails, search still completes and the summary field is empty. Scalar metrics stay in **`metrics`**; prose lives in **`semantic_trace_summary`** / **`semantic_summaries`** only.

**Copilot Studio (optional UI):** install **`[inspect]`** and **`[copilot]`** (`pip install -e ".[inspect,copilot]"`), then run **`axiom copilot-studio`**. It opens a separate Streamlit app from Glass Box (`axiom inspect`): enter expert URL / model / API key, goal, optional context, iteration limit, optional **Summarize traces**, search mode **`compile_only`** / **`predict_rows`** / **`train_tabular`** (plus JSON text areas for row examples or tabular train/eval), then use **Draft once** or **Run search** — nothing calls the network until you click. You get tables for iteration summaries, expandable eval/metrics/failure JSON, and download buttons for **`draft.ax`**, **`best.ax`**, and **`copilot_report.json`**.

**Copilot HTTP server (Phase 67+):** headless FastAPI app for **`/draft`**, **`/search`**, **`/run`** (Phase 71 — same body shape as **`/search`** plus optional **`final_validate`**, response includes **`disclaimer`** and **`final_validation`**), **`/benchmarks/run`**, **`/summarize`**, and **`/health`** — not **`axiom serve`** (bundles) and not the policy gateway. **`POST /search`** accepts optional **`train_tabular`** (target + train/eval row lists + optional Adam hyperparameters); it cannot be combined with **`compile_only`** or **`examples`** in the same request. Install **`pip install -e ".[serve,copilot]"`**, then e.g. **`axiom copilot-serve --expert-url https://your-host/v1/ --expert-model qwen-7b --port 8020`**. Optional **`AXIOM_COPILOT_API_KEY`**: when set, POST routes require **`Authorization: Bearer …`** or **`X-API-Key`** (health stays open). Downstream chat auth for the expert remains **`AXIOM_EXPERT_API_KEY`** / **`--expert-api-key`** as for the CLI.

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

**Metric-driven repair (Phase 76):** for **`--examples-json`** or **`--train-tabular`**, the search loop keeps repairing after a **valid** candidate until the metric meets a threshold or iterations are exhausted (not merely “first compile success”). Default is **on**; disable with **`--no-repair-valid-with-metrics`**. Override the stop threshold with **`--metric-repair-if-below FLOAT`**; for the built-in **`neg_mse`** scorer, if you omit that flag the default is **`≈ -1e-9`** (near-perfect fit). Reports and pipeline summaries include **`metric_repair`** + **`convergence_reason`** (threshold met vs budget vs compile-only success).

**Row-wise repair signal (Phase 77):** when **`predict_rows`** runs with **`expected`** rows, evaluation records **`row_comparisons`** (inputs, predicted vs expected outputs, per-target **`abs_error`**, worst rows first). Repair prompts include a **`## Row-wise mismatches`** JSON block so the expert can correct coefficients; evaluation JSON and iteration artifacts include **`row_comparisons`** when present. For math-like goals, an extra hint prefers symbolic arithmetic over **`neural(...)`** for exact example fits (library default; not a new CLI flag).

**Exact-symbolic / anti-neural bias (Phase 78):** for small example-driven goals that look like affine or **`max`/`min`** clamps (e.g. **`risk_score`** in the goal), the draft/repair context sets **`exact_symbolic_examples_task`** so the Onyx prompts push **direct arithmetic** and **`REPAIR_NEURAL_TO_SYMBOLIC_BLOCK`** when the current program still uses **`neural(...)`**. Candidate selection uses **`adjusted_sort_score`** (raw metric minus small penalties for **`neural`** on these tasks, indexed **`x_0`** names, **`output(`**, and suspicious numerics like **`03`**); reports include **`ranking_penalty`** and **`adjusted_sort_score`**. No new CLI flags.

**Exact-symbolic / control-flow fast paths (deterministic):**
- **What they do:** before the first expert draft call, search runs a local symbolic inference over examples and emits canonical `.ax` immediately when it can prove an exact fit.
- **Tolerant inference (v1.2 headline):** when rows are *almost* exact (label noise, sparse anchors), **`tolerant_inference.py`** fits affine / interaction / abs families by least squares (default **5% relative RMSE**), picks the simplest model, and skips the LLM entirely. Robustness tasks like **`noisy_affine_thermometer`**, **`signed_cross_term_noisy`**, and **`near_abs_with_bias`** are solved deterministically in tests.
- **LLM output normalizer:** every expert response passes through **`compiler/normalizer.py`** (`else if` → nested blocks, `&&`/`||` → nested `if`s, `clip(...)` → `min`/`max`, shorthand assignments, comment/prose stripping) before parsing.
- **Why they exist:** they reduce latency and backend load for common symbolic tasks, avoid unnecessary LLM variance, and keep outputs deterministic when the examples already define a closed form.
- **When they activate:** only in **`predict_rows`** when **`exact_symbolic_examples_task`** is true.
- **Supported task shapes today:**
  - one-input affine (**double_x family**): **`y = a * x + b`**
  - one-input piecewise threshold identity / zero-floor (**piecewise_threshold family**): **`if (x < 0.0) y = 0.0 else y = x`**
  - exact three-region nested piecewise identity/cap: **`if (x < 0.0) y = 0.0 else { if (x < 1.0) y = x else y = 1.0 }`**
  - exact two-input max: **`score = max(a, b)`**
  - exact two-input min/max blend: **`out = max(0.0, min(a + b, 1.0));`**
  - clamped two-input affine (**risk_score family**): **`out = max(0.0, min(1.0, a * x1 + b * x2 + c));`**
  - exact two-input interaction: **`out = w_ab * a * b + w_a * a + w_b * b + c`**
  - exact three-input affine (**three_input_affine family**): **`out = sum(w_i * x_i) + b`**
  - exact three-way max/min: **`score = max(min(a, b), c)`**
- **Fallback to Onyx expert backend:** if inference is ambiguous/underdetermined, rows are non-numeric, shape constraints do not match, or any row fails exact validation (including clamp edge rows), search proceeds with normal expert draft/repair.
- **Concrete examples:** `double_x`, `piecewise_threshold`, `nested_piecewise`, `max_of_two`, `minmax_blend`, `risk_score`, `quadratic_with_cross_term`, `three_input_affine`, and `three_way_maxmin` are now covered by deterministic fast paths.

**What is now proven working:**
- deterministic fast-path emission for these symbolic/copilot task families when rows define an exact fit: `double_x`, `piecewise_threshold`, `nested_piecewise`, `max_of_two`, `minmax_blend`, `risk_score`, `quadratic_with_cross_term`, `three_input_affine`, and `three_way_maxmin`
- canonical `.ax` output for direct arithmetic and nested control-flow forms, including the required `else { if (...) { ... } else { ... } }` shape
- exact row validation gate before accepting a fast path
- full search benchmark quality on the current milestone: `10/10` compile and `10/10` metric
- backend-only smoke: `12/12` quality checks passed
- strict fallback to expert draft/repair when rows are ambiguous/noisy/out-of-shape

**What is still model-dependent:**
- draft-only quality on the current benchmark (`5/10` compile, `2/10` metric)
- tasks outside current exact fast-path families
- any noisy or underdetermined example set (intent cannot be proven exactly)
- generalization quality after fallback to LLM draft/repair
- prompt-following quality under different expert backends/models

**Non-blocking follow-up:** remaining ONNX tracer warnings are follow-up cleanup work, not a blocker for the completed symbolic copilot milestone.

**Current milestone snapshot (2026-06-17, local venv with CI-parity extras):**
- `pytest`: see [GitHub Actions](https://github.com/Danny1218/Axiom/actions) for authoritative green CI; local full suite may show environment-specific `torch.compile` / gateway gaps
- offline benchmark-dispatch (CI): symbolic **10/10**, next symbolic **9/9**, generalization stress **8/8**, robustness/ambiguity **8/8**
- live Onyx operator snapshot (not CI): draft quality varies by task; search symbolic suites are green offline

**Known local-only failures (not CI blockers):** some `torch.compile` fullgraph probes and `httpx` TestClient gateway tests fail on certain Windows/PyTorch builds — track via `plan.md` and open issues rather than treating as regressions.

**Recommended next milestone:** `HTTP serve polish, profiler baselines, copilot onboarding diagnostics`

```powershell
# Local smoke bundles (Windows PowerShell)
.\scripts\smoke_copilot_symbolic.ps1
.\scripts\smoke_copilot_non_fast_path.ps1

# double_x — copilot-search
axiom copilot-search --backend onyx-qwen --goal "Compute y as double of x." `
  --expert-url "http://127.0.0.1:8000" --expert-model "onyx-qwen-production-v1" `
  --examples-json ./examples/double_x.json --iterations 6 `
  --artifact-dir ./debug_double_x --out ./debug_double_x/best.ax --report-out ./debug_double_x/search_report.json

# double_x — copilot-run
axiom copilot-run --backend onyx-qwen --goal "Compute y as double of x." `
  --expert-url "http://127.0.0.1:8000" --expert-model "onyx-qwen-production-v1" `
  --examples-json ./examples/double_x.json --iterations 6 `
  --artifact-dir ./showcase_double_x --out ./showcase_double_x.ax --summary-out ./showcase_double_x/pipeline_summary.json

# risk_score — copilot-search
axiom copilot-search --backend onyx-qwen `
  --goal "Compute risk_score = max(0.0, min(1.0, 0.7 * risk_a + 0.3 * risk_b));" `
  --expert-url "http://127.0.0.1:8000" --expert-model "onyx-qwen-production-v1" `
  --examples-json ./examples/risk_score_v3.json --iterations 8 `
  --artifact-dir ./debug_risk_score --out ./debug_risk_score/best.ax --report-out ./debug_risk_score/search_report.json

# risk_score — copilot-run
axiom copilot-run --backend onyx-qwen `
  --goal "Compute risk_score = max(0.0, min(1.0, 0.7 * risk_a + 0.3 * risk_b));" `
  --expert-url "http://127.0.0.1:8000" --expert-model "onyx-qwen-production-v1" `
  --examples-json ./examples/risk_score_v3.json --iterations 8 `
  --artifact-dir ./showcase_risk_score --out ./showcase_risk_score.ax --summary-out ./showcase_risk_score/pipeline_summary.json

# piecewise_threshold — copilot-search
axiom copilot-search --backend onyx-qwen `
  --goal "Write a valid Axiom .ax program in this repo's DSL that computes y = x when x > 0, otherwise y = 0.0." `
  --expert-url "http://127.0.0.1:8000" --expert-model "onyx-qwen-production-v1" `
  --examples-json ./examples/piecewise_threshold.json --iterations 8 `
  --artifact-dir ./debug_piecewise_threshold --out ./debug_piecewise_threshold/best.ax --report-out ./debug_piecewise_threshold/search_report_cli.json

# piecewise_threshold — copilot-run
axiom copilot-run --backend onyx-qwen `
  --goal "Write a valid Axiom .ax program in this repo's DSL that computes y = x when x > 0, otherwise y = 0.0." `
  --expert-url "http://127.0.0.1:8000" --expert-model "onyx-qwen-production-v1" `
  --examples-json ./examples/piecewise_threshold.json --iterations 8 `
  --artifact-dir ./showcase_piecewise_threshold --out ./showcase_piecewise_threshold.ax --summary-out ./showcase_piecewise_threshold/pipeline_summary.json

# three_input_affine — copilot-search
axiom copilot-search --backend onyx-qwen `
  --goal "Write a valid Axiom .ax program in this repo's DSL that computes score = 0.5 * a + 0.3 * b + 0.2 * c." `
  --expert-url "http://127.0.0.1:8000" --expert-model "onyx-qwen-production-v1" `
  --examples-json ./examples/three_input_affine.json --iterations 8 `
  --artifact-dir ./debug_three_input_affine --out ./debug_three_input_affine/best.ax --report-out ./debug_three_input_affine/search_report_cli.json

# three_input_affine — copilot-run
axiom copilot-run --backend onyx-qwen `
  --goal "Write a valid Axiom .ax program in this repo's DSL that computes score = 0.5 * a + 0.3 * b + 0.2 * c." `
  --expert-url "http://127.0.0.1:8000" --expert-model "onyx-qwen-production-v1" `
  --examples-json ./examples/three_input_affine.json --iterations 8 `
  --artifact-dir ./showcase_three_input_affine --out ./showcase_three_input_affine.ax --summary-out ./showcase_three_input_affine/pipeline_summary.json
```

**Limitations / next targets:**
- no multi-output symbolic inference yet
- no symbolic inference for products/nonlinear terms beyond current bounded affine shape
- no robust symbolic handling for noisy-but-close data (current behavior requires exact fit and otherwise falls back)

**Train-tabular search:** use **`--train-tabular`** with **`--tabular-json path.json`** (do not combine with **`--compile-only`** or **`--examples-json`**). The file is one JSON **object**: **`target_var`**, **`train_rows`**, **`eval_rows`** (each row is **`{"inputs": {...}, "expected": {...}}`**; keys are merged for the evaluator), optional **`epochs`**, **`learning_rate`**, **`weight_decay`**, **`batch_size`**. Same schema as **`axiom.copilot.tabular_json`**. Metric-driven repair defaults **on** here as well (same CLI flags).

```powershell
$examples = @'
[{"inputs": {}, "expected": {"y": 0.5}}]
'@
Set-Content -Path examples.json -Value $examples -Encoding utf8
axiom copilot-search --backend onyx-qwen --goal "Output y from defaults" `
  --expert-url "https://api.example.com/v1/" --expert-model "qwen-7b" `
  --iterations 5 --examples-json examples.json `
  --out best.ax --report-out search_report.json --artifact-dir ./copilot_run_01 `
  --summarize-traces
```

Omit **`--summarize-traces`** to skip the extra expert round-trips.

**Pipeline (goal → best `.ax` + summary JSON + optional artifact dir):**

```powershell
axiom copilot-run --backend onyx-qwen --goal "Small policy on x" `
  --expert-url "https://api.example.com/v1/" --expert-model "qwen-7b" `
  --iterations 5 --compile-only `
  --artifact-dir ./copilot_e2e --summary-out pipeline_summary.json --out best.ax
```

**Multi-restart** (same flags; add **`--restarts N`** — artifacts go under **`restart_0/`** … when **`N > 1`**):

```powershell
axiom copilot-run --backend onyx-qwen --goal "…" `
  --expert-url "https://api.example.com/v1/" --expert-model "qwen-7b" `
  --iterations 10 --examples-json ./examples/risk_score_v3.json `
  --restarts 5 --artifact-dir ./copilot_run --summary-out ./pipeline.json --out ./best.ax
```

**Benchmark harness:** `axiom.copilot.benchmarks` defines tiny NL tasks (`DEFAULT_BENCHMARK_TASKS`), compares draft-only vs full search, and serializes with **`benchmark_suite_to_dict`**. **CLI:** **`axiom copilot-benchmark`** (expert flags like other copilot commands; optional **`--task-json`**, **`--out`**, **`--draft-only`** or **`--search`**, **`--max-iterations`**). **HTTP:** with **`pip install -e ".[serve,copilot]"`**, **`POST /benchmarks/run`** on the copilot server accepts optional inline **`tasks`**, **`max_iterations`**, **`draft_only`**, **`search_only`**; response wraps the same JSON document under **`suite`**. Extra tasks can be loaded from **`axiom/copilot/fixtures/benchmark_tasks.json`** or your own file matching that schema.

For local Onyx evaluation of symbolic + generalization tasks, use **`benchmarks/copilot_symbolic_and_generalization_tasks.json`** (includes labels like **`fast_path_expected`** and **`category`**; loader ignores unknown fields):

```powershell
axiom copilot-benchmark --backend onyx-qwen `
  --expert-url "http://127.0.0.1:8000" --expert-model "onyx-qwen-production-v1" `
  --task-json "./benchmarks/copilot_symbolic_and_generalization_tasks.json" `
  --max-iterations 6 --out "./benchmarks/copilot_symbolic_and_generalization_results.json"
```

**In-memory tabular training (library API):** **`evaluate_program(..., mode="train_tabular")`** trains a compiled **`InterpretedBlock`** on **`train_rows`** with Adam (numeric dict rows, ABI-aware trunk fill, **`target_var`** column blinded in inputs like **`AxiomDataset`**), reports **`train_mse`** / **`eval_mse`** on **`eval_rows`**, optional **`TrainTabularParams`** (**`epochs`**, **`learning_rate`**, **`weight_decay`**, **`batch_size`**) and **`max_unroll`**. Purely symbolic programs get a **`no_trainable_parameters`** warning and eval metrics only—no subprocess and not a replacement for full **`axiom train`**.

---

## `axiom serve`

Serves **one** `.axb` at startup via FastAPI: **`GET /health`**, **`POST /predict`**, **`POST /explain`**, **`POST /report`** (JSON **`inputs`**; report can return inline HTML). Install **`pip install -e ".[serve]"`**.

Programmatic **`serve.create_app(path, expert_registry=..., expert_handler=..., expert_fallback=...)`** mirrors **`axiom.load`** for bundles that use **`expert()`**; otherwise **`POST /predict`** (and **`/explain`** / **`/report`**) return **503** with an explicit message when **`expert()`** is present but unwired.

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

Production-style image for **`axiom serve`**: one **`.axb`** (+ optional **`.axb.weights.pt`** sidecar for unlocked neural bundles), FastAPI on **`HOST`** / **`PORT`**. The **`Dockerfile`** installs **`pip install ".[serve,lock]"`** and sets **`AXIOM_REQUIRE_API_KEY=1`**. Required env: **`AXIOM_API_KEY`**. Optional: **`AXIOM_BUNDLE_SECRET`** (unlock **`env-secret`** locked bundles). No bundle is baked in—set **`AXIOM_BUNDLE_PATH`** at runtime.

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

**Note:** set **`AXIOM_API_KEY`** in **`.env`** (see **`.env.example`**) before **`docker compose up`**, so **`POST /predict`** requires **`Authorization: Bearer …`** (see below). **`GET /health`** stays unauthenticated.

### Example `curl`

In another terminal, after the container is running:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": {}}'
```

When **`AXIOM_API_KEY`** is set (via **`.env`** or compose):

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer $AXIOM_API_KEY" \
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

**`expert()` / `OP_EXPERT` (Phase 66, 72):** if the program calls **`expert("name", features)`**, the bundle does **not** store Python handlers. After **`axiom.load`**, attach runtime callables with **`model.set_expert_registry({"name": fn})`** (or **`ExpertRuntimeRegistry`**), **`model.set_expert_handler(fn)`** for one dispatch callable, or **`model.set_expert_fallback(float)`**. This registry is **not** the semantic copilot **`SemanticExpert`** (LLM HTTP). For **`axiom serve`**, pass **`expert_registry=`** / **`expert_handler=`** / **`expert_fallback=`** into **`serve.create_app(...)`**; **`POST /predict`**, **`/explain`**, and **`/report`** return **503** if the bundle uses **`expert()`** and nothing is wired.

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

**Pragmatic default:** drive **Path A** once—pick a dataset you care about, beat or match a baseline, *then* invest in B or C with evidence. Reproducible starter: **`python scripts/run_path_a_portfolio_vertical.py`** (finance portfolio; writes **`artifacts/path_a_portfolio/report.json`** and a portable **`.axb`** + **`.weights.pt`** pair).

---

## Links

- **Repository:** [github.com/Danny1218/Axiom](https://github.com/Danny1218/Axiom)  
- **Tests:** CI-parity: `pip install -e ".[dev,copilot,serve]"` + `pip install -r constraints-dev.txt` then `python -m pytest tests -q`  
- **Project state (maintainers):** see `plan.md` in this repo.

## License

Axiom is licensed under the **GNU Affero General Public License v3.0** (see `LICENSE`).
In short: you may use, study, and modify it freely, but if you build a product or
service on it, your modifications must be open-sourced under the same license.
For a commercial license without AGPL obligations, contact Onyx Protocol.
