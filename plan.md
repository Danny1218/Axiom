# Axiom — project state (source of truth)

## Current phase

**Phase 1–4:** Parser/IR, supernet + topology + Sinkhorn + shadow meta + Liquid KAN / `OP_LOOP` (see prior sections in git history).

**Phase 5 (complete):** **`LiquidSequenceLoader`** (sequential float → `(B, D)` + **`baseline_var` noise**), **`EvolutionaryTrainer`** (forward → `loss.backward` → Adam → optional **`MetaCompiler`**; end-of-epoch **shadow localized MSE** + **`ShadowFitnessEvaluator`** → **`apply_shadow_verdict`**; **`rebuild_optimizer()`** after mask/shadow changes), **`save_execution_bundle`** (`*.pt` `state_dict` + **`*_topology.json`** with nodes/edges/topo_order + optional **IR JSON**), **`main.py`** (default **`train.ax`**, 10 epochs, save bundle).

## Layout

- `main.py` — CLI entry
- `train.ax` — default training sketch
- `compiler/serializer.py` — bundle I/O
- `engine/dataloader.py`, `engine/trainer.py`
- `primitives/`, `engine/*` (prior phases), `tests/`

## IR opcodes

`OP_CONST`, `OP_LOAD`, `OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_DIV`, `OP_NEG`, `OP_CMP_*`, `OP_ASSIGN`, `OP_EXPR_STMT`, `OP_CONDITIONAL`, `OP_LOOP`.

## Run training (PowerShell)

```powershell
cd "...\Axiom"
pip install -r requirements.txt
python main.py train.ax --epochs 10 --out axiom_bundle
python -m pytest tests -q
```

## Next (not started)

Reload bundle into a reconstructed `ExecutionGraph`, real IR interpreter in-loop, distributed dataloader (see `readme.md`).
