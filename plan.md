# Axiom — project state (source of truth)

## Current phase

**Phase 1 (complete):** Grammar (`.ax`), Lark parser → AST, IR bridge, `LatentSupernet` + `TTLoRAAdapter`, `test.ax`, pytest.

**Phase 2 (complete):** `SinkhornRouter`, `ExecutionGraph` / `ConditionalSinkhornBlock`, `wire_execution_graph()` for `OP_CONDITIONAL`.

**Phase 3 (complete):** `MutationSignal`, `MetaCompiler`, shadow mode + `fitness` utilities.

**Phase 4 (complete):** `LiquidStateTensor` (`primitives/liquid_tensor.py`), `LiquidKANNode` (`engine/ssm.py`) with hat (linear B-spline-style) basis + liquid mixing, **`while`** in grammar → **`OP_LOOP`** in IR → **`LiquidKANNode`** in execution graph; `loop.ax` sample; `wire_execution_graph(..., loop_max_unroll=, loop_num_basis=)`.

## Layout

- `compiler/` — `grammar.lark`, `parser.py`, `ir.py`, `flow.py`
- `primitives/` — `liquid_tensor.py`
- `engine/` — `supernet`, `router`, `topology`, `ssm`, `signals`, `meta_compiler`, `fitness`
- `tests/` — phases 1–4
- `requirements.txt` — `torch`, `lark`, `networkx`, `pytest`

## IR opcodes

`OP_CONST`, `OP_LOAD`, `OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_DIV`, `OP_NEG`, `OP_CMP_*`, `OP_ASSIGN`, `OP_EXPR_STMT`, `OP_CONDITIONAL`, **`OP_LOOP`** `(cond_expr, body_ir)`.

## IR → topology

- `OP_LOOP` → `LiquidKANNode` (body IR stored on NX node `body_ir` for future interpretation).
- `forward(h)` on the node = fixed unroll recurrence; `forward_sequence([LiquidStateTensor,...])` uses per-step τ and payloads.

## Next (not started)

Execute `body_ir` inside the loop module, full KAN grids, true cubic B-splines, sequence batching (see `readme.md`).

## Verify locally (Windows PowerShell)

```powershell
cd "...\Axiom"
pip install -r requirements.txt
python -m pytest tests -q
```
