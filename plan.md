# Axiom — project state (source of truth)

## Current phase

**Phase 1–5:** Parser/IR, supernet + topology + Sinkhorn + shadow meta + Liquid KAN / `OP_LOOP`, dataloader, evolutionary trainer, serializer, CLI (now **`axiom train`**).

**Phase 6 (complete):** **`engine/interpreter.py`** — stack IR eval, `run_while_loop` / `run_loop_snapshots` (prelude stmts + seed map + padded state vectors). **`InterpretedLiquidLoop`** — runs interpreted snapshots per batch row, **`LiquidKANNode.forward_sequence_tensors`** for sequence memory, falls back to **`forward`** when **`max_unroll==0`** (empty sequence). **`build_execution_graph_from_ir`** — `OP_LOOP` → `InterpretedLiquidLoop`; contiguous **`OP_ASSIGN` / `OP_EXPR_STMT`** before a loop are absorbed as prelude (no duplicate Identity nodes). IR remains on graph nodes for tooling.

**Phase 7 (complete):** **`compiler/serializer.py`** — topology JSON + weights. **Phase 14** adds **`abi`** to JSON. **`compiler/deserializer.py`** — **`load_execution_bundle`**; IR tuple-normalized via **`_ir_from_json`**; loops receive the saved or IR-rebuilt global ABI.

**Phase 8 (complete):** Differentiable IR (tensor stack, safe **`OP_DIV`**, **`torch.where`** compares). **`truthy`** remains for scalar/B=1 checks only.

**Phase 9 (complete):** **`engine/interpreter.py`** — batched SIMT: env values are **`(B,)`** tensors; **`eval_expr(..., B=...)`**; **`OP_CONST`** → **`torch.full((B,), ...)`**; **`OP_CONDITIONAL`** → clone env, run then/else under **`active_mask`**, merge with **`torch.where(cond, then, else)`** then **`torch.where(active_mask, ...)`**; **`run_while_loop`** uses **`entering = scope & (cond_val != 0)`** each iter; body updates under **`entering`**. **`Phase 12`** removed the early **`break`** when **`~entering.any()`** so every run performs exactly **`max_unroll`** iterations (phantom steps: env frozen, mask False). **`run_loop_snapshots`** returns fixed **`T = max_unroll`** (or **`T=0`** if **`max_unroll==0`**): **`(seq, seq_mask)`** shapes **`(B, T, D)`** and **`(B, T)`**. **`InterpretedLiquidLoop`** passes **`mask=seq_mask`** into **`LiquidKANNode.forward_sequence_tensors`**. **`forward_sequence_tensors`**: **`h_cur = torch.where(m_t, h_next, h_prev)`** on padded timesteps. Tests: **`tests/test_simt_padding_drift.py`**, **`tests/test_vectorized_interpreter.py`**, **`tests/test_ssm.py`**, **`tests/test_phase12_fullgraph_loops.py`**.

**Phase 10 (complete):** **Functional forward** — **`ConditionalSinkhornBlock.forward`** returned **`(out, shadows)`**; superseded by Phase 11 three-tuple API below.

**Phase 11 (complete):** **Graph purity & signal bubbling** — Removed **`engine/signals.py`** / **`MutationSignal`**. **`SinkhornRouter.forward`** returns **`(weights, normalized_entropy)`** (0-dim entropy tensor, no **`.item()`** / no **`last_mutation_signal`**). Dummy-mask Sinkhorn path; **`ConditionalSinkhornBlock`** / **`ExecutionGraph`** three-tuple forward; **`react_to_signals`**; trainer gates shadow MSE on **`is_shadow`**. **`compile_graph=True`**: **`capture_dynamic_output_shape_ops`**.

**Phase 12 (complete):** **Static SIMT loop unroll** — No **`if not entering.any(): break`** in **`run_while_loop`**; **`run_loop_snapshots`** always stacks **`max_unroll`** steps (or empty if **`max_unroll==0`**). **`EvolutionaryTrainer`**: **`torch.compile(..., fullgraph=True)`** for all graphs (loops included). **`tests/test_jit_compile.py`** mixed cond+loop uses **`fullgraph=True`**. Proof: **`tests/test_phase12_fullgraph_loops.py`**.

**Phase 13 (complete):** **`engine/inference.py`** — **`AxiomRunner`** loads a deserialized **`ExecutionGraph`**; **`predict`** / **`predict_batch`** under **`torch.no_grad()`** / **`.eval()`**. **`predict_with_signals`** for CLI. **`EvolutionaryTrainer`** compile path: inductor warmup with **`aot_eager`** fallback. **`axiom train`**: **`--mode train|inference`**.

**Phase 14 (complete):** **Global feature ABI** — **`compiler.ir.extract_global_abi(ir, max_vars=dim)`** walks the program in document order; first-seen variable names map to trunk columns **`0..min(n_vars,dim)-1`**. **`ExecutionGraph.abi`**: **`Dict[str, int]`**. All **`InterpretedLiquidLoop`** instances share that map (no per-loop **`make_seed_map`**). **`run_loop_snapshots`** / **`InterpretedLiquidLoop`**: **`seed_map`** is **`name -> column`**. **`execution_topology_to_dict`** writes **`"abi"`**; **`load_execution_bundle`** reads **`abi`** or rebuilds from embedded **`ir`**. **`AxiomRunner`** fills columns from **`graph.abi`**, default **`0.0`** for missing names, ignores unknown keys; empty **`abi`** keeps legacy sorted/broadcast behavior for ancient bundles. Tests: **`tests/test_inference_abi.py`**, **`tests/test_inference_api.py`**, **`tests/test_flow.py`**, **`tests/test_deserializer.py`**.

**Phase 15 (complete):** **Latent channel padding** — **`run_loop_snapshots(..., trunk_dim=...)`** zero-pads the sequence last dim up to the trunk width when the stacked snapshot width is smaller, so **`LiquidKANNode`** always sees **`(B, T, D_trunk)`** even if the IR layout width **`dim`** is narrower (ABI/script-only width vs supernet capacity). **`InterpretedLiquidLoop`** passes **`trunk_dim=flat.shape[-1]`**. **`engine/dataloader.py`**: **`AxiomDataset`** maps **`List[Dict[str, float]]`** into **`(trunk_dim,)`** inputs via **`abi`** and scalar **`target_key`** → **`y`** shape **`(1,)`**. Tests: **`tests/test_latent_channel_padding.py`**, **`tests/test_real_world_training.py`**, **`tests/test_dataloader_phase5.py`**.

**Phase 16 (complete):** **Targeted objective routing** — **`EvolutionaryTrainer(..., target_col=None)`**; when set, main and shadow MSE use **`out[:, c:c+1]`** (and the same slice on **`loc`**) vs **`y.view(-1, 1)`**, so only the ABI column for the prediction is supervised; no full-trunk broadcast of a scalar target (avoids latent space collapse). **`AxiomDataset`** no longer has **`broadcast_target`**. Tests: **`tests/test_target_column_loss.py`**, updated **`tests/test_real_world_training.py`** (**`target_col=graph.abi["x"]`**).

**Phase 17 (complete):** **Target blinding & dict outputs** — **`AxiomDataset`**: **`target_col = abi.get(target_key)`**; after filling **`x`**, **`x[target_col] = 0`** when that column is the supervised target so labels are not leaked into inputs. **`AxiomRunner`**: **`predict_dict`** / **`predict_dict_batch`** decode trunk tensors with **`graph.abi`**; **`predict`** / **`predict_batch`** / **`predict_with_signals`** set **`self.device`** and **`x = x.to(self.device)`** before the graph. **`axiom train --mode inference`** prints **`predict_dict(...)`**. Tests: **`tests/test_target_leakage.py`**, **`tests/test_inference_api.py`** (**`predict_dict`**, **`predict_dict_batch`**).

**Phase 18 (complete):** **Hybrid symbolic–neural execution** — **`engine/block_executor.py`**: **`InterpretedBlock`** runs IR stmts with **`exec_stmt`**, seeds **`env`** from **`h`** via **`abi`**, repacks trunk columns. Root **`OP_ASSIGN` / `OP_EXPR_STMT`** use **`InterpretedBlock`** (DAG node carries **`ir`** for bundles). **`ConditionalSinkhornBlock`** runs symbolic **`then_ir` / `else_ir`** per branch, then **`out = w0*(h_then+y0) + w1*(h_else+y1)`** (same shadow semantics on raw adapter outputs). **`load_execution_bundle`** rebuilds stmt/conditional IR from JSON. Tests: **`tests/test_hybrid_execution.py`**. **`torch._dynamo.reset()`** in **`test_evolutionary_trainer_compile_fullgraph_with_loop_one_epoch`** avoids Dynamo cache exhaustion after other compile tests.

**Phase 19 (complete):** **Domain tooling — Titanic** — **`load_csv_to_dicts`** / **`_cell_to_float`** in **`engine/dataloader.py`** ( **`csv.DictReader`**, numeric parse, **`female`/`male`**, empty → **0** ). **`examples/titanic.ax`** (current): **sabotage** rule **`Fare > 100000`** → useless symbolic **`survived_prob`**; hybrid **`TT-LoRA`** + optional **`MetaCompiler`** still fit **`Survived`**. **`examples/run_titanic.py`**: **`MetaCompiler` on by default** (**`--no-meta`** to disable); saves **`axiom_bundle`** (**`.pt` + `_topology.json`**) unless **`--no-save`**; **`--out`** prefix. Tests: **`tests/test_csv_titanic.py`**. **`examples/titanic.csv`** / **`axiom_bundle*`** gitignored.

**Phase 20 (complete):** **Glass Box visualizer** — **`src/axiom/tools/inspector.py`**: Streamlit UI loads a bundle via path prefix or uploaded **`upload_topology.json` + `upload.pt`**, builds ABI **`st.number_input`**s, **`Run inference`** → **`AxiomRunner.predict_with_signals`**, large output metric + routing expander. **`glass_box.py`**: **`execution_graph_to_graphviz`**, **`routing_trace_entries`**, **`tensor_preview_dict`**. **`ConditionalSinkhornBlock`** signals add **`{block}_weights`**; **`MetaCompiler.react_to_signals`** skips non-scalar tensors. Run: **`axiom inspect`** or **`streamlit run …/inspector.py`**. Tests: **`tests/test_glass_box.py`**.

**Phase 21 (complete):** **Deep Liquid-KAN expressivity** — **`engine/ssm.py`**: **`_hat_basis` → `_rbf_basis`**, Gaussian bumps **`exp(-(diff²))`** on normalized time coordinate (centers on **[0,1]**). **`LiquidKANNode`**: **`fuse_proj`**: **`Linear(2D, D)`** on **`cat(h_cur, x_t)`**, **`F.layer_norm`**, RBF coefficients mix, **`w_gate`**: **`Linear(3D, 1)`** on **`cat(h_cur, x_t, h0)`** → sigmoid scales KAN output. **`forward_sequence` / `forward_sequence_tensors`**: proposal **`_kan_update(h_cur, x_t, h0, tn)`** only (no **`0.1 * x_t`**). **`forward`**: zero dummy **`x_t`**. **`t_norm`** kept in signature for API stability (unused). Tests: **`tests/test_ssm.py`** (RBF, grads on fusion, **`x_t`** sensitivity); **`tests/test_hybrid_execution.py`** symbolic test uses asymmetric branch constants so Sinkhorn blending is not exactly zero at **`b`**.

**Phase 22 (complete):** **High-dimensional KAN splines** — **`_rbf_basis`**: **`sigmoid(fused_norm)`** per channel **`(B, D)`**, RBFs broadcast to **`(B, D, K)`** (no mean-pool). **`coeffs`**: **`(dim, num_basis)`**, init **`randn / sqrt(K)`**; readout **`(phi * coeffs).sum(-1)`** → **`(B, D)`**. **`tests/test_kan_mean_blindness.py`**: **`[1,-1]` vs `[-1,1]`** same mean, distinct **`phi`**; reference helper shows old pooled path collapses; **`LiquidKANNode`** with **`fuse_proj`** copying **`h`** from **`cat([h,0])`** + fixed gate proves full forward separates permutations. Re-saved bundles need retrain (**`coeffs`** shape vs Phase 21).

**Phase 23 (complete):** **Packaging** — Installable **`axiom-engine`** (**`pyproject.toml`**, **`src/axiom/`**): **`compiler/`**, **`engine/`**, **`primitives/`**, **`tools/`**, **`cli.py`**. Imports are **`axiom.*`**. Global CLI: **`axiom train …`**, **`axiom inspect`** (subprocess **`streamlit run`** on packaged **`inspector.py`**). **`pip install -e .`** for dev; **`grammar.lark`** in **`package-data`**. **`tests/`**, **`examples/`** stay at repo root; examples assume editable install.

## Layout

- `pyproject.toml` — **`axiom-engine`**, script **`axiom` → `axiom.cli:main`**
- `src/axiom/cli.py` — train / inspect subcommands
- `src/axiom/tools/inspector.py`, `glass_box.py` — Glass Box
- `examples/titanic.ax`, `examples/run_titanic.py` — Titanic pipeline (**`pip install -e .`** first)
- `train.ax` — default **`axiom train`** sketch (cwd)
- `src/axiom/compiler/`, `src/axiom/engine/`, `src/axiom/primitives/`
- `tests/`

## IR opcodes

`OP_CONST`, `OP_LOAD`, `OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_DIV`, `OP_NEG`, `OP_CMP_*`, `OP_ASSIGN`, `OP_EXPR_STMT`, `OP_CONDITIONAL`, `OP_LOOP`.

## Run training (PowerShell)

```powershell
cd "...\Axiom"
pip install -e .
python -m pytest tests -q
axiom train train.ax --epochs 10 --out axiom_bundle
axiom inspect
```

## Next

More domain examples, feature-rich Titanic encodings (Age/Fare in ABI), distributed dataloader (see `readme.md`). Further Dynamo hardening if new IR ops add Python breaks. Optional: embed Graphviz WASM fallback if `dot` is missing on Windows.
