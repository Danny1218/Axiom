# Axiom — Architecture & status

**Version:** 1.3.0 · **Stack:** Python 3.10+, PyTorch, Lark, NetworkX

## What this is

Axiom is a hybrid **symbolic–neural compiler**: `.ax` source is parsed (Lark) into IR, lowered to an
`InterpretedBlock` (PyTorch `nn.Module`), trained, serialized to `.axb` bundles, and served over HTTP.
The **semantic copilot** drafts and repairs `.ax` programs from goals + example rows via an injectable
expert backend (LM Studio locally, deterministic dispatch offline).

## v1.3 evidence benchmarks

| Benchmark | Command | What it proves |
|-----------|---------|----------------|
| **baseline_showdown** | `python benchmarks/baseline_showdown/run_showdown.py` | Tolerant symbolic inference extrapolates on 7/10 formula families; declines sabotage tasks |
| **titanic_hybrid** | `python benchmarks/titanic_hybrid/run_hybrid_audit.py` | InterpretedBlock hybrid enforces hard rules (0 violations); sklearn baselines violate |

Evidence: `docs/evidence/baseline_showdown.{json,md}`, `docs/evidence/titanic_hybrid.{json,md}`.
Optional extra: `pip install -e ".[bench]"`.

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

**Backends:** `benchmark-dispatch` (CI/offline), `onyx-qwen`, `lmstudio` (OpenAI-compatible local default). Local default: `axiom copilot-doctor --backend lmstudio`.

## Test & benchmark status

| Check | Target |
|-------|--------|
| Full suite | `python -m pytest tests -q` → 0 failures |
| Extrapolation showdown | 7/10 in-family extrap wins (>=10x vs ML); 2/2 sabotage declined |
| Titanic hybrid | 0 constraint violations (InterpretedBlock); accuracy gap documented |

## Intentionally frozen (do not break without explicit milestone)

- Public API: `axiom.__all__`, `AxiomModel` methods, CLI subcommand names
- Optional extras: `spy`, `cartpole`, `inspect`, `gateway`, `serve`, `lock`, `export`, `copilot`, `dev`, `bench`
- Four benchmark JSON schemas and `benchmark-dispatch` reference programs

## Release checklist (v1.3.0)

- [x] `benchmarks/baseline_showdown/` + committed evidence
- [x] `benchmarks/titanic_hybrid/` + `examples/titanic_hybrid.ax`
- [x] `[bench]` optional extra (scikit-learn)
- [x] README benchmark section; tag `v1.3.0`
