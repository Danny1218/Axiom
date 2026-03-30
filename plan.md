# Axiom — project state (source of truth)

## Current phase

**Phase 1–5:** Parser/IR, supernet + topology + Sinkhorn + shadow meta + Liquid KAN / `OP_LOOP`, dataloader, evolutionary trainer, serializer, `main.py`.

**Phase 6 (complete):** **`engine/interpreter.py`** — stack IR eval, `run_while_loop` / `run_loop_snapshots` (prelude stmts + seed map + padded state vectors). **`InterpretedLiquidLoop`** — runs interpreted snapshots per batch row, **`LiquidKANNode.forward_sequence_tensors`** for sequence memory, falls back to **`forward`** when **`max_unroll==0`** (empty sequence). **`build_execution_graph_from_ir`** — `OP_LOOP` → `InterpretedLiquidLoop`; contiguous **`OP_ASSIGN` / `OP_EXPR_STMT`** before a loop are absorbed as prelude (no duplicate Identity nodes). IR remains on graph nodes for tooling.

**Phase 7 (complete):** **`compiler/serializer.py`** — `execution_topology_to_dict` adds **`supernet_config`** (dim, adapter_names, rank), **`router_config`** (Sinkhorn iters/eps/mutation threshold from the first conditional block, else defaults), **`loop_config`** (num_basis/max_unroll from the first loop), plus per-node **`expert_then` / `expert_else`** and **`loop_num_basis` / `loop_max_unroll`**. **`compiler/deserializer.py`** — **`load_execution_bundle(path_prefix)`** rebuilds `nx.DiGraph`, `LatentSupernet`, `ConditionalSinkhornBlock` / `InterpretedLiquidLoop` / `Identity`, then **`load_state_dict`** from `*.pt`. IR lists from JSON are tuple-normalized via **`_ir_from_json`**; loop **`seed_map`** is recomputed with **`make_seed_map`**.

**Phase 8 (complete):** Differentiable IR (tensor stack, safe **`OP_DIV`**, **`torch.where`** compares). **`truthy`** remains for scalar/B=1 checks only.

**Phase 9 (complete):** **`engine/interpreter.py`** — batched SIMT: env values are **`(B,)`** tensors; **`eval_expr(..., B=...)`**; **`OP_CONST`** → **`torch.full((B,), ...)`**; **`OP_CONDITIONAL`** → clone env, run then/else under **`active_mask`**, merge with **`torch.where(cond, then, else)`** then **`torch.where(active_mask, ...)`**; **`run_while_loop`** uses **`entering = scope & (cond_val != 0)`** each iter; body updates under **`entering`**. **`Phase 12`** removed the early **`break`** when **`~entering.any()`** so every run performs exactly **`max_unroll`** iterations (phantom steps: env frozen, mask False). **`run_loop_snapshots`** returns fixed **`T = max_unroll`** (or **`T=0`** if **`max_unroll==0`**): **`(seq, seq_mask)`** shapes **`(B, T, D)`** and **`(B, T)`**. **`InterpretedLiquidLoop`** passes **`mask=seq_mask`** into **`LiquidKANNode.forward_sequence_tensors`**. **`forward_sequence_tensors`**: **`h_cur = torch.where(m_t, h_next, h_prev)`** on padded timesteps. Tests: **`tests/test_simt_padding_drift.py`**, **`tests/test_vectorized_interpreter.py`**, **`tests/test_ssm.py`**, **`tests/test_phase12_fullgraph_loops.py`**.

**Phase 10 (complete):** **Functional forward** — **`ConditionalSinkhornBlock.forward`** returned **`(out, shadows)`**; superseded by Phase 11 three-tuple API below.

**Phase 11 (complete):** **Graph purity & signal bubbling** — Removed **`engine/signals.py`** / **`MutationSignal`**. **`SinkhornRouter.forward`** returns **`(weights, normalized_entropy)`** (0-dim entropy tensor, no **`.item()`** / no **`last_mutation_signal`**). Dummy-mask Sinkhorn path; **`ConditionalSinkhornBlock`** / **`ExecutionGraph`** three-tuple forward; **`react_to_signals`**; trainer gates shadow MSE on **`is_shadow`**. **`compile_graph=True`**: **`capture_dynamic_output_shape_ops`**.

**Phase 12 (complete):** **Static SIMT loop unroll** — No **`if not entering.any(): break`** in **`run_while_loop`**; **`run_loop_snapshots`** always stacks **`max_unroll`** steps (or empty if **`max_unroll==0`**). **`EvolutionaryTrainer`**: **`torch.compile(..., fullgraph=True)`** for all graphs (loops included). **`tests/test_jit_compile.py`** mixed cond+loop uses **`fullgraph=True`**. Proof: **`tests/test_phase12_fullgraph_loops.py`**.

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

Distributed dataloader (see `readme.md`). Bundles saved before Phase 7 lack `supernet_config` / expert fields — reload requires a freshly saved topology JSON. Optional: **`backend="inductor"`** on Linux+CUDA; further Dynamo hardening if new IR ops add Python breaks.
