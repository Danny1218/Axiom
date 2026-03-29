# Axiom — project state (source of truth)

## Current phase

**Phase 1–5:** Parser/IR, supernet + topology + Sinkhorn + shadow meta + Liquid KAN / `OP_LOOP`, dataloader, evolutionary trainer, serializer, `main.py`.

**Phase 6 (complete):** **`engine/interpreter.py`** — stack IR eval, `run_while_loop` / `run_loop_snapshots` (prelude stmts + seed map + padded state vectors). **`InterpretedLiquidLoop`** — runs interpreted snapshots per batch row, **`LiquidKANNode.forward_sequence_tensors`** for sequence memory, falls back to **`forward`** when the loop body never runs (T=0). **`build_execution_graph_from_ir`** — `OP_LOOP` → `InterpretedLiquidLoop`; contiguous **`OP_ASSIGN` / `OP_EXPR_STMT`** before a loop are absorbed as prelude (no duplicate Identity nodes). IR remains on graph nodes for tooling.

## Layout

- `main.py` — CLI entry
- `train.ax` — default training sketch
- `compiler/serializer.py` — bundle I/O
- `engine/dataloader.py`, `engine/trainer.py`, `engine/interpreter.py`, `engine/loop_executor.py`
- `primitives/`, `engine/*`, `tests/`

## IR opcodes

`OP_CONST`, `OP_LOAD`, `OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_DIV`, `OP_NEG`, `OP_CMP_*`, `OP_ASSIGN`, `OP_EXPR_STMT`, `OP_CONDITIONAL`, `OP_LOOP`.

## Run training (PowerShell)

```powershell
cd "...\Axiom"
pip install -r requirements.txt
python main.py train.ax --epochs 10 --out axiom_bundle
python -m pytest tests -q
```

## Next

Reload bundle into a reconstructed `ExecutionGraph`; optional gradients through IR seeds; distributed dataloader (see `readme.md`).
