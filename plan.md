# Axiom — project state (source of truth)

## Current phase

**Phase 1–5:** Parser/IR, supernet + topology + Sinkhorn + shadow meta + Liquid KAN / `OP_LOOP`, dataloader, evolutionary trainer, serializer, `main.py`.

**Phase 6 (complete):** **`engine/interpreter.py`** — stack IR eval, `run_while_loop` / `run_loop_snapshots` (prelude stmts + seed map + padded state vectors). **`InterpretedLiquidLoop`** — runs interpreted snapshots per batch row, **`LiquidKANNode.forward_sequence_tensors`** for sequence memory, falls back to **`forward`** when the loop body never runs (T=0). **`build_execution_graph_from_ir`** — `OP_LOOP` → `InterpretedLiquidLoop`; contiguous **`OP_ASSIGN` / `OP_EXPR_STMT`** before a loop are absorbed as prelude (no duplicate Identity nodes). IR remains on graph nodes for tooling.

**Phase 7 (complete):** **`compiler/serializer.py`** — `execution_topology_to_dict` adds **`supernet_config`** (dim, adapter_names, rank), **`router_config`** (Sinkhorn iters/eps/mutation threshold from the first conditional block, else defaults), **`loop_config`** (num_basis/max_unroll from the first loop), plus per-node **`expert_then` / `expert_else`** and **`loop_num_basis` / `loop_max_unroll`**. **`compiler/deserializer.py`** — **`load_execution_bundle(path_prefix)`** rebuilds `nx.DiGraph`, `LatentSupernet`, `ConditionalSinkhornBlock` / `InterpretedLiquidLoop` / `Identity`, then **`load_state_dict`** from `*.pt`. IR lists from JSON are tuple-normalized via **`_ir_from_json`**; loop **`seed_map`** is recomputed with **`make_seed_map`**.

**Phase 8 (complete):** **`engine/interpreter.py`** — IR env is **`Dict[str, torch.Tensor]`** (0D tensors). **`eval_expr`** / **`exec_stmt`** / **`run_while_loop`** take **`device`** and **`dtype`**; stack ops use PyTorch math; comparisons use **`torch.where`**. **`OP_DIV`** uses a **safe denominator** (`torch.where(mask, b, 1)` then divide, then mask to 0) so **`a/b` is never evaluated at `b==0`** — avoids backward **0×Inf → NaN** from eager **`torch.where(..., a/b, 0)`**. **`truthy`** uses **`detach().item()`** (branch choice is discrete, not differentiated). **`snapshot_env`** returns **`torch.stack(...)`**; **`run_loop_snapshots`** seeds with **`flat[idx].reshape(())`** (no **`.item()`**), builds **`torch.stack(snaps, dim=0)`** so the **(T, D)** sequence stays on the autograd graph. **`InterpretedLiquidLoop`** passes **`h.device`** / **`h.dtype`** into **`run_loop_snapshots`**. Tests: **`tests/test_interpreter_autograd.py`**, **`tests/test_interpreter_nan_trap.py`**, extended **`tests/test_interpreter.py`**.

## Layout

- `main.py` — CLI entry
- `train.ax` — default training sketch
- `compiler/serializer.py`, `compiler/deserializer.py` — bundle save / reload
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

Distributed dataloader (see `readme.md`). Bundles saved before Phase 7 lack `supernet_config` / expert fields — reload requires a freshly saved topology JSON.
