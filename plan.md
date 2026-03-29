# Axiom — project state (source of truth)

## Current phase

**Phase 1 (complete):** Grammar (`.ax`), Lark parser → AST, IR bridge (opcode tuples + `ir_to_digraph` chain), `LatentSupernet` + `TTLoRAAdapter`, `test.ax`, pytest.

**Phase 2 (complete):** `SinkhornRouter` (balanced optimal-transport routing over masked experts), `ExecutionGraph` (NetworkX DAG of IR steps → `nn.Module` nodes), `ConditionalSinkhornBlock` at each `OP_CONDITIONAL`, compiler `wire_execution_graph()` bridge. Forward: trunk → topo order through graph; routing uses differentiable Sinkhorn; autograd verified in tests.

## Layout

- `compiler/` — `grammar.lark`, `parser.py`, `ir.py`, `flow.py` (`wire_execution_graph`)
- `engine/` — `supernet.py`, `router.py`, `topology.py`
- `tests/` — parser, IR, supernet, router, topology, flow, phase2 integration
- `requirements.txt` — `torch`, `lark`, `networkx`, `pytest`

## IR → topology

- Each IR instruction is a DAG node (`stmt` = `Identity`, `OP_CONDITIONAL` = `ConditionalSinkhornBlock` with `SinkhornRouter` + two named LoRA experts).
- Call `wire_execution_graph(ir, supernet, [(then_name, else_name), ...])` with one pair per `OP_CONDITIONAL` in program order.

## IR opcodes (Phase 1)

`OP_CONST`, `OP_LOAD`, `OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_DIV`, `OP_NEG`, `OP_CMP_*`, `OP_ASSIGN`, `OP_EXPR_STMT`, `OP_CONDITIONAL`.

## Next (not started)

Phase 3+: evaluate IR expr stack, richer merge nodes, Sinkhorn-Knopp in log domain for scale, MoE-scale graphs (see `readme.md`).

## Verify locally (Windows PowerShell)

```powershell
cd "...\Axiom"
pip install -r requirements.txt
python -m pytest tests -q
```
