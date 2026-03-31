"""Named runtime handlers for in-program ``expert("name", …)`` (``OP_EXPERT``).

Separate from :mod:`axiom.experts` (semantic copilot / LLM backends). These are in-process
``(name, feature_row) -> float`` callables; they are never serialized in ``.axb`` files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from axiom.engine.expert_call import ExpertHandler

Stmt = Tuple[Any, ...]


class ExpertRuntimeRegistry:
    """Register handlers by the string literal used in source: ``expert("my_key", …)``."""

    __slots__ = ("_handlers",)

    def __init__(self) -> None:
        self._handlers: Dict[str, ExpertHandler] = {}

    def register(self, name: str, handler: ExpertHandler) -> None:
        self._handlers[str(name)] = handler

    def resolve(self, name: str) -> Optional[ExpertHandler]:
        return self._handlers.get(str(name))

    def clear(self) -> None:
        self._handlers.clear()

    def copy(self) -> ExpertRuntimeRegistry:
        r = ExpertRuntimeRegistry()
        r._handlers = dict(self._handlers)
        return r


def _expr_collect_expert_names(expr: List[Any], into: Set[str]) -> None:
    for tup in expr:
        if not isinstance(tup, tuple) or not tup:
            continue
        op = tup[0]
        if op == "OP_EXPERT" and len(tup) >= 3:
            into.add(str(tup[1]))
            _expr_collect_expert_names(list(tup[2]), into)
        elif op == "OP_NEURAL" and len(tup) >= 3:
            _expr_collect_expert_names(list(tup[2]), into)
        elif op == "OP_CALL":
            for a in tup[2]:
                _expr_collect_expert_names(list(a), into)


def _stmt_collect_expert_names(stmt: Any, into: Set[str]) -> None:
    if not isinstance(stmt, tuple) or not stmt:
        return
    op = stmt[0]
    if op == "OP_ASSIGN":
        _expr_collect_expert_names(list(stmt[2]), into)
    elif op == "OP_BLEND_ASSIGN":
        _expr_collect_expert_names(list(stmt[2]), into)
        _expr_collect_expert_names(list(stmt[3]), into)
    elif op == "OP_EXPR_STMT":
        _expr_collect_expert_names(list(stmt[1]), into)
    elif op == "OP_CONDITIONAL":
        _expr_collect_expert_names(list(stmt[1]), into)
        for s in stmt[2]:
            _stmt_collect_expert_names(tuple(s) if isinstance(s, list) else s, into)
        for s in stmt[3]:
            _stmt_collect_expert_names(tuple(s) if isinstance(s, list) else s, into)
    elif op == "OP_LOOP":
        _expr_collect_expert_names(list(stmt[1]), into)
        for s in stmt[2]:
            _stmt_collect_expert_names(tuple(s) if isinstance(s, list) else s, into)


def collect_expert_backend_names_from_stmts(ir_stmts: List[Any]) -> frozenset[str]:
    names: Set[str] = set()
    for st in ir_stmts:
        _stmt_collect_expert_names(tuple(st) if isinstance(st, list) else st, names)
    return frozenset(names)


def interpreted_block_ir_contains_expert(block: Any) -> bool:
    return bool(collect_expert_backend_names_from_stmts(block.ir_stmts))


def expert_runtime_wiring_sufficient(block: Any) -> bool:
    """True if every ``OP_EXPERT`` backend name can be resolved at runtime."""
    names = collect_expert_backend_names_from_stmts(block.ir_stmts)
    if not names:
        return True
    if block.expert_fallback is not None:
        return True
    if block.expert_handler is not None:
        return True
    reg = getattr(block, "expert_registry", None)
    if reg is None:
        return False
    for n in names:
        if reg.resolve(n) is None:
            return False
    return True


__all__ = [
    "ExpertRuntimeRegistry",
    "collect_expert_backend_names_from_stmts",
    "expert_runtime_wiring_sufficient",
    "interpreted_block_ir_contains_expert",
]
