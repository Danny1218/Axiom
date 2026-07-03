# Axiom — Architecture & status

**Version:** 1.2.0 (in progress) · **Stack:** Python 3.10+, PyTorch, Lark, NetworkX

## What this is

Axiom is a hybrid **symbolic–neural compiler**: `.ax` source is parsed (Lark) into IR, lowered to an
`InterpretedBlock` (PyTorch `nn.Module`), trained, serialized to `.axb` bundles, and served over HTTP.
The **semantic copilot** drafts and repairs `.ax` programs from goals + example rows via an injectable
expert backend (LM Studio locally, deterministic dispatch offline).

## Two execution paths

| Path | When | How |
|------|------|-----|
| **Interpreted** | Default training & inference | IR walked step-by-step; autograd-safe ops only |
| **Compiled** | Opt-in `torch.compile(fullgraph=True)` | Hot loops/conditionals; strict mode uses plain bools outside traced regions |

Public entry points (frozen): `axiom.load`, `AxiomModel.predict/explain/export_report`, CLI subcommands
listed in `tests/test_architecture_baseline.py`.

## Copilot pipeline

```
goal + examples → exact fast paths (search.py)
                → tolerant symbolic regression (tolerant_inference.py)  [v1.2]
                → LLM draft/repair (onyx_qwen / lmstudio alias)
                → normalizer (canonical .ax) → parse → evaluate → repair loop
```

**Backends:** `benchmark-dispatch` (CI/offline), `onyx-qwen`, `lmstudio` (OpenAI-compatible local default).

**Benchmark suites** (under `benchmarks/`):

1. `copilot_symbolic_and_generalization_tasks.json` — core 10-task gate
2. `copilot_symbolic_next_milestone_tasks.json` — harder symbolic families
3. `copilot_symbolic_generalization_stress_tasks.json` — paraphrase / reorder stress
4. `copilot_symbolic_robustness_ambiguity_stress_tasks.json` — noisy / underdetermined rows

CI runs all four offline via `benchmark-dispatch` (`.github/workflows/copilot-milestone.yml`).

## Test & benchmark status

| Check | Target |
|-------|--------|
| Full suite | `python -m pytest tests -q` → 0 failures |
| Fast loop | `python -m pytest tests -m "not slow and not compile" -q` |
| Offline benchmarks | draft + search 10/10, 8/8, 8/8, 8/8 on dispatch |
| Robustness (v1.2) | noisy tasks solved by tolerant inference without LLM |

Known environment caveat: `torch.compile(fullgraph=True)` cannot trace `ContextVar` in strict mode on
some PyTorch builds; affected tests are skipped with a precise reason or fixed by hoisting strict flags.

## Package layout

```
src/axiom/
  compiler/   Lark parse, IR, serializer, normalizer
  engine/     InterpretedBlock, trainer, inference, strict
  copilot/    search, tolerant_inference, benchmarks, server
  experts/    onyx_qwen, registry
  gateway/    policy gateway (optional)
  serve.py    bundle HTTP server
  cli.py      train, predict, serve, copilot-* subcommands
tests/        contract + unit + integration
examples/     titanic.ax, portfolio.ax, train scripts
```

## Intentionally frozen (do not break without explicit milestone)

- Public API: `axiom.__all__`, `AxiomModel` methods, CLI subcommand names
- Optional extras in `pyproject.toml`: `spy`, `cartpole`, `inspect`, `gateway`, `serve`, `lock`, `export`, `copilot`, `dev`
- Gateway exports (`axiom.gateway.__all__`)
- Bundle v2 manifest format and locked-bundle crypto checks
- Four benchmark JSON schemas and `benchmark-dispatch` reference programs
- Exact fast-path detectors in `search.py` (extend, do not remove families)

## Local copilot setup (default)

1. Start LM Studio with `qwen/qwen3-8b` at `http://127.0.0.1:1234`
2. `pip install -e ".[copilot]"`
3. `axiom copilot-doctor --backend lmstudio`

## Release checklist (v1.2.0)

- [ ] Tolerant inference wired and tested on robustness tasks
- [ ] Normalizer on all LLM responses
- [ ] README quickstart + honest capability table
- [ ] `python -m build` succeeds; tag `v1.2.0`
- [ ] Root clean (no generated JSON/PT artifacts)
