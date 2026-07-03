# Axiom — Architecture & status

**Version:** 1.4.0 · **Stack:** Python 3.10+, PyTorch, Lark, NetworkX

## What this is

Axiom is a hybrid **symbolic–neural compiler**: `.ax` source is parsed (Lark) into IR, lowered to an
`InterpretedBlock` (PyTorch `nn.Module`), trained, serialized to `.axb` bundles, and served over HTTP.
The **semantic copilot** drafts and repairs `.ax` programs from goals + example rows via an injectable
expert backend (LM Studio locally, deterministic dispatch offline).

## v1.4 evidence benchmarks

| Benchmark | Command | What it proves |
|-----------|---------|----------------|
| **titanic_guarded** | `python benchmarks/titanic_hybrid/run_guarded_audit.py` | `expert()` wraps GBM (~0.85 acc), 0 violations, interval certificate |
| **baseline_showdown** | `python benchmarks/baseline_showdown/run_showdown.py` | 9/10 extrap wins, unclipped noise, scale-relative gates |
| **titanic_hybrid** (v1.3) | `python benchmarks/titanic_hybrid/run_hybrid_audit.py` | Neural hybrid constraint baseline |

Evidence: `docs/evidence/titanic_guarded.{json,md}`, `titanic_guarded_certificate.json`,
`baseline_showdown.{json,md}`. CLI: **`axiom certify`**.

## Two execution paths

| Path | When | How |
|------|------|-----|
| **Interpreted** | Default training & inference | IR walked step-by-step; autograd-safe ops only |
| **Compiled** | Opt-in `torch.compile(fullgraph=True)` | Hot loops/conditionals; strict mode uses plain bools outside traced regions |

Public entry points (frozen): `axiom.load`, `AxiomModel.predict/explain/export_report`, CLI subcommands
listed in `tests/test_architecture_baseline.py` (+ **`certify`** in v1.4).

## Copilot pipeline

```
goal + examples → exact fast paths (search.py)
                → tolerant symbolic regression (tolerant_inference.py)  [scale-relative gates v1.4]
                → LLM draft/repair (onyx_qwen / lmstudio alias)
                → normalizer (canonical .ax) → parse → evaluate → repair loop
```

## Test & benchmark status

| Check | Target |
|-------|--------|
| Full suite | `python -m pytest tests -q` → 0 failures |
| Guarded Titanic | GBM accuracy preserved, 0 violations, certificate hi ≤ 0.15 |
| Extrapolation showdown | 9/10 in-family wins; 2/2 sabotage declined; noise unclipped |

## Intentionally frozen (do not break without explicit milestone)

- Public API: `axiom.__all__`, `AxiomModel` methods, CLI subcommand names (except additive `certify`)
- Optional extras: `spy`, `cartpole`, `inspect`, `gateway`, `serve`, `lock`, `export`, `copilot`, `dev`, `bench`
- Four benchmark JSON schemas and `benchmark-dispatch` reference programs

## Release checklist (v1.4.0)

- [x] `examples/titanic_guarded.ax` + `run_guarded_audit.py` + certificate
- [x] Scale-relative tolerant gates; showdown noise un-rigged
- [x] `src/axiom/verify/interval.py` + `axiom certify`
- [x] README / plan; tag `v1.4.0`
