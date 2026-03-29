# Axiom — project state (source of truth)

## Current phase

**Phase 1 (complete):** Grammar (`.ax`), Lark parser → AST, IR bridge (opcode tuples + optional NetworkX chain graph), `LatentSupernet` with frozen trunk and masked TT-style LoRA adapters, `test.ax` + pytest coverage.

## Layout

- `compiler/` — `grammar.lark`, `parser.py`, `ir.py`
- `engine/` — `supernet.py` (`TTLoRAAdapter`, `LatentSupernet`)
- `tests/` — parser, IR, supernet, integration (`test.ax`)
- `requirements.txt` — `torch`, `lark`, `networkx`, `pytest`

## IR opcodes (Phase 1)

`OP_CONST`, `OP_LOAD`, `OP_ADD`, `OP_SUB`, `OP_MUL`, `OP_DIV`, `OP_NEG`, `OP_CMP_*`, `OP_ASSIGN`, `OP_EXPR_STMT`, `OP_CONDITIONAL` (cond list, then list, else list).

## Next (not started)

Phase 2+: topology mapper (IR → NetworkX module graph), execution, Sinkhorn routing, etc. (see `readme.md` vision).

## Verify locally (Windows PowerShell)

```powershell
cd "...\Axiom"
pip install -r requirements.txt
python -m pytest tests -q
```
