# Axiom — project state (source of truth)

## Current phase

**Phase 1 (complete):** Grammar (`.ax`), Lark parser → AST, IR bridge, `LatentSupernet` + `TTLoRAAdapter`, `test.ax`, pytest.

**Phase 2 (complete):** `SinkhornRouter`, `ExecutionGraph` / `ConditionalSinkhornBlock`, `wire_execution_graph()` for `OP_CONDITIONAL`.

**Phase 3 (complete):** Uncertainty-driven **MutationSignal** (normalized routing entropy), **MetaCompiler** (unmask inactive expert in shadow), **shadow mode** (`is_shadow` + detached contributions in supernet / conditional), **fitness** (`ShadowFitnessEvaluator`, `run_shadow_training_epochs`, `apply_shadow_verdict`), `ExecutionGraph.shadow_locals()` for localized loss inputs.

## Layout

- `compiler/` — `grammar.lark`, `parser.py`, `ir.py`, `flow.py`
- `engine/` — `supernet.py`, `router.py`, `topology.py`, `signals.py`, `meta_compiler.py`, `fitness.py`
- `tests/` — phases 1–3 coverage
- `requirements.txt` — `torch`, `lark`, `networkx`, `pytest`

## IR → topology

- `wire_execution_graph(..., mutation_entropy_norm_threshold=...)` forwards to conditional routers.
- **Meta NAS:** run forward → read `router.last_mutation_signal` / `g.routers()` → `MetaCompiler.react_to_router_signals` → optional `unmask_next_inactive(shadow=True)`.
- **Shadow training:** `g(x)` → `g.shadow_locals()` → localized loss on raw adapter outputs → `ShadowFitnessEvaluator` over 5 epochs → `apply_shadow_verdict`.

## IR opcodes (Phase 1)

`OP_CONST`, `OP_LOAD`, `OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_DIV`, `OP_NEG`, `OP_CMP_*`, `OP_ASSIGN`, `OP_EXPR_STMT`, `OP_CONDITIONAL`.

## Next (not started)

Full IR stack execution, multi-merge DAG nodes, log-domain Sinkhorn, batched meta-steps (see `readme.md`).

## Verify locally (Windows PowerShell)

```powershell
cd "...\Axiom"
pip install -r requirements.txt
python -m pytest tests -q
```
