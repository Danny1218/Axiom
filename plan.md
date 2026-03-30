# Axiom — project state (source of truth)

## Current phase

**Phase 1–5:** Parser/IR, supernet + topology + Sinkhorn + shadow meta + Liquid KAN / `OP_LOOP`, dataloader, evolutionary trainer, serializer, `main.py`.

**Phase 6 (complete):** **`engine/interpreter.py`** — stack IR eval, `run_while_loop` / `run_loop_snapshots` (prelude stmts + seed map + padded state vectors). **`InterpretedLiquidLoop`** — runs interpreted snapshots per batch row, **`LiquidKANNode.forward_sequence_tensors`** for sequence memory, falls back to **`forward`** when the loop body never runs (T=0). **`build_execution_graph_from_ir`** — `OP_LOOP` → `InterpretedLiquidLoop`; contiguous **`OP_ASSIGN` / `OP_EXPR_STMT`** before a loop are absorbed as prelude (no duplicate Identity nodes). IR remains on graph nodes for tooling.

**Phase 7 (complete):** **`compiler/serializer.py`** — `execution_topology_to_dict` adds **`supernet_config`** (dim, adapter_names, rank), **`router_config`** (Sinkhorn iters/eps/mutation threshold from the first conditional block, else defaults), **`loop_config`** (num_basis/max_unroll from the first loop), plus per-node **`expert_then` / `expert_else`** and **`loop_num_basis` / `loop_max_unroll`**. **`compiler/deserializer.py`** — **`load_execution_bundle(path_prefix)`** rebuilds `nx.DiGraph`, `LatentSupernet`, `ConditionalSinkhornBlock` / `InterpretedLiquidLoop` / `Identity`, then **`load_state_dict`** from `*.pt`. IR lists from JSON are tuple-normalized via **`_ir_from_json`**; loop **`seed_map`** is recomputed with **`make_seed_map`**.

**Phase 8 (complete):** Differentiable IR (tensor stack, safe **`OP_DIV`**, **`torch.where`** compares). **`truthy`** remains for scalar/B=1 checks only.

**Phase 9 (complete):** **`engine/interpreter.py`** — batched SIMT: env values are **`(B,)`** tensors; **`eval_expr(..., B=...)`**; **`OP_CONST`** → **`torch.full((B,), ...)`**; **`OP_CONDITIONAL`** → clone env, run then/else under **`active_mask`**, merge with **`torch.where(cond, then, else)`** then **`torch.where(active_mask, ...)`**; **`run_while_loop`** uses **`entering = scope & (cond_val != 0)`** each iter (**`scope`** = all ones or **`parent_active`** for nested loops), body updates under **`entering`**, **`snapshot_env`** → **`(B, D)`**, snapshots **`torch.stack(..., dim=1)`** → **`(B, T, D)`**. **`run_loop_snapshots`** accepts **`(B, D)`** or **`(D,)`** (treated as **`B=1`**); returns **`(seq, seq_mask)`** with **`seq_mask[b,t] = entering[b]`** for that step ( **`(B, T)`** bool). **`InterpretedLiquidLoop`** passes **`mask=seq_mask`** into **`LiquidKANNode.forward_sequence_tensors`**. **`forward_sequence_tensors`**: when **`mask`** set, **`h_cur = torch.where(m_t, h_next, h_prev)`** so padded timesteps do not drift liquid state. Tests: **`tests/test_simt_padding_drift.py`**, **`tests/test_vectorized_interpreter.py`**, **`tests/test_ssm.py`** (mask), updated interpreter tests.

**Phase 10 (complete):** **Functional forward** — **`ConditionalSinkhornBlock.forward`** returns **`(out, Dict[str, Tensor])`** shadow locals (no **`self.last_shadow_outputs`** mutation). **`InterpretedLiquidLoop.forward`** returns **`(y, {})`**. **`ExecutionGraph.forward(x)`** returns **`(h, all_shadows)`**; **`shadow_locals()`** removed. **`EvolutionaryTrainer`**: **`compile_graph=False`** default; if **`True`**, sets **`torch._dynamo.config.capture_scalar_outputs = True`** (Sinkhorn mutation uses **`.item()`**), then **`step_fn = torch.compile(graph, backend="aot_eager")`**. **`train_epoch`**: **`out, locs = self.step_fn(x)`**. Tests: **`tests/test_jit_compile.py`**, **`tests/test_trainer_evolutionary.py`**, call sites updated. **`SinkhornRouter`** still assigns **`last_mutation_signal`** (meta-compiler); full single-graph Inductor may need a follow-up pure router API.

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
