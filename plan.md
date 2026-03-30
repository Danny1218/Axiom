# Axiom — project state (source of truth)

## Current phase

**Phase 1–5:** Parser/IR, supernet + topology + Sinkhorn + shadow meta + Liquid KAN / `OP_LOOP`, dataloader, evolutionary trainer, serializer, `main.py`.

**Phase 6 (complete):** **`engine/interpreter.py`** — stack IR eval, `run_while_loop` / `run_loop_snapshots` (prelude stmts + seed map + padded state vectors). **`InterpretedLiquidLoop`** — runs interpreted snapshots per batch row, **`LiquidKANNode.forward_sequence_tensors`** for sequence memory, falls back to **`forward`** when **`max_unroll==0`** (empty sequence). **`build_execution_graph_from_ir`** — `OP_LOOP` → `InterpretedLiquidLoop`; contiguous **`OP_ASSIGN` / `OP_EXPR_STMT`** before a loop are absorbed as prelude (no duplicate Identity nodes). IR remains on graph nodes for tooling.

**Phase 7 (complete):** **`compiler/serializer.py`** — topology JSON + weights. **Phase 14** adds **`abi`** to JSON. **`compiler/deserializer.py`** — **`load_execution_bundle`**; IR tuple-normalized via **`_ir_from_json`**; loops receive the saved or IR-rebuilt global ABI.

**Phase 8 (complete):** Differentiable IR (tensor stack, safe **`OP_DIV`**, **`torch.where`** compares). **`truthy`** remains for scalar/B=1 checks only.

**Phase 9 (complete):** **`engine/interpreter.py`** — batched SIMT: env values are **`(B,)`** tensors; **`eval_expr(..., B=...)`**; **`OP_CONST`** → **`torch.full((B,), ...)`**; **`OP_CONDITIONAL`** → clone env, run then/else under **`active_mask`**, merge with **`torch.where(cond, then, else)`** then **`torch.where(active_mask, ...)`**; **`run_while_loop`** uses **`entering = scope & (cond_val != 0)`** each iter; body updates under **`entering`**. **`Phase 12`** removed the early **`break`** when **`~entering.any()`** so every run performs exactly **`max_unroll`** iterations (phantom steps: env frozen, mask False). **`run_loop_snapshots`** returns fixed **`T = max_unroll`** (or **`T=0`** if **`max_unroll==0`**): **`(seq, seq_mask)`** shapes **`(B, T, D)`** and **`(B, T)`**. **`InterpretedLiquidLoop`** passes **`mask=seq_mask`** into **`LiquidKANNode.forward_sequence_tensors`**. **`forward_sequence_tensors`**: **`h_cur = torch.where(m_t, h_next, h_prev)`** on padded timesteps. Tests: **`tests/test_simt_padding_drift.py`**, **`tests/test_vectorized_interpreter.py`**, **`tests/test_ssm.py`**, **`tests/test_phase12_fullgraph_loops.py`**.

**Phase 10 (complete):** **Functional forward** — **`ConditionalSinkhornBlock.forward`** returned **`(out, shadows)`**; superseded by Phase 11 three-tuple API below.

**Phase 11 (complete):** **Graph purity & signal bubbling** — Removed **`engine/signals.py`** / **`MutationSignal`**. **`SinkhornRouter.forward`** returns **`(weights, normalized_entropy)`** (0-dim entropy tensor, no **`.item()`** / no **`last_mutation_signal`**). Dummy-mask Sinkhorn path; **`ConditionalSinkhornBlock`** / **`ExecutionGraph`** three-tuple forward; **`react_to_signals`**; trainer gates shadow MSE on **`is_shadow`**. **`compile_graph=True`**: **`capture_dynamic_output_shape_ops`**.

**Phase 12 (complete):** **Static SIMT loop unroll** — No **`if not entering.any(): break`** in **`run_while_loop`**; **`run_loop_snapshots`** always stacks **`max_unroll`** steps (or empty if **`max_unroll==0`**). **`EvolutionaryTrainer`**: **`torch.compile(..., fullgraph=True)`** for all graphs (loops included). **`tests/test_jit_compile.py`** mixed cond+loop uses **`fullgraph=True`**. Proof: **`tests/test_phase12_fullgraph_loops.py`**.

**Phase 13 (complete):** **`engine/inference.py`** — **`AxiomRunner`** loads a deserialized **`ExecutionGraph`**; **`predict`** / **`predict_batch`** under **`torch.no_grad()`** / **`.eval()`**. **`predict_with_signals`** for CLI. **`EvolutionaryTrainer`** compile path: inductor warmup with **`aot_eager`** fallback. **`main.py`**: **`--mode train|inference`**.

**Phase 14 (complete):** **Global feature ABI** — **`compiler.ir.extract_global_abi(ir, max_vars=dim)`** walks the program in document order; first-seen variable names map to trunk columns **`0..min(n_vars,dim)-1`**. **`ExecutionGraph.abi`**: **`Dict[str, int]`**. All **`InterpretedLiquidLoop`** instances share that map (no per-loop **`make_seed_map`**). **`run_loop_snapshots`** / **`InterpretedLiquidLoop`**: **`seed_map`** is **`name -> column`**. **`execution_topology_to_dict`** writes **`"abi"`**; **`load_execution_bundle`** reads **`abi`** or rebuilds from embedded **`ir`**. **`AxiomRunner`** fills columns from **`graph.abi`**, default **`0.0`** for missing names, ignores unknown keys; empty **`abi`** keeps legacy sorted/broadcast behavior for ancient bundles. Tests: **`tests/test_inference_abi.py`**, **`tests/test_inference_api.py`**, **`tests/test_flow.py`**, **`tests/test_deserializer.py`**.

**Phase 15 (complete):** **Latent channel padding** — **`run_loop_snapshots(..., trunk_dim=...)`** zero-pads the sequence last dim up to the trunk width when the stacked snapshot width is smaller, so **`LiquidKANNode`** always sees **`(B, T, D_trunk)`** even if the IR layout width **`dim`** is narrower (ABI/script-only width vs supernet capacity). **`InterpretedLiquidLoop`** passes **`trunk_dim=flat.shape[-1]`**. **`engine/dataloader.py`**: **`AxiomDataset`** maps **`List[Dict[str, float]]`** into **`(trunk_dim,)`** inputs via **`abi`** and scalar **`target_key`** → **`y`** shape **`(1,)`**. Tests: **`tests/test_latent_channel_padding.py`**, **`tests/test_real_world_training.py`**, **`tests/test_dataloader_phase5.py`**.

**Phase 16 (complete):** **Targeted objective routing** — **`EvolutionaryTrainer(..., target_col=None)`**; when set, main and shadow MSE use **`out[:, c:c+1]`** (and the same slice on **`loc`**) vs **`y.view(-1, 1)`**, so only the ABI column for the prediction is supervised; no full-trunk broadcast of a scalar target (avoids latent space collapse). **`AxiomDataset`** no longer has **`broadcast_target`**. Tests: **`tests/test_target_column_loss.py`**, updated **`tests/test_real_world_training.py`** (**`target_col=graph.abi["x"]`**).

## Layout

- `main.py` — CLI entry
- `train.ax` — default training sketch
- `compiler/serializer.py`, `compiler/deserializer.py` — bundle save / reload
- `engine/dataloader.py`, `engine/trainer.py`, `engine/inference.py`, `engine/interpreter.py`, `engine/loop_executor.py`
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

Distributed dataloader (see `readme.md`). Bundles without **`abi`** or embedded **`ir`** deserialize with empty ABI (legacy inference layout). Further Dynamo hardening if new IR ops add Python breaks. Optional: CSV/JSON loaders that build **`AxiomDataset`** rows.
