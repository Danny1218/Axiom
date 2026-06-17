"""Strict execution mode for the interpreted IR (fail-fast diagnostics)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional, Set


class StrictInferenceError(ValueError):
    """Deterministic runtime error raised when strict mode rejects lenient behavior."""


_strict_ctx: ContextVar[bool] = ContextVar("axiom_strict_mode", default=False)
_env_defined_ctx: ContextVar[Optional[Set[str]]] = ContextVar("axiom_env_defined", default=None)


def strict_mode_enabled() -> bool:
    return bool(_strict_ctx.get())


def env_defined_set() -> Optional[Set[str]]:
    return _env_defined_ctx.get()


@contextmanager
def strict_execution(enabled: bool, env_defined: Optional[Set[str]] = None) -> Iterator[None]:
    t1 = _strict_ctx.set(bool(enabled))
    t2 = _env_defined_ctx.set(env_defined)
    try:
        yield
    finally:
        _strict_ctx.reset(t1)
        _env_defined_ctx.reset(t2)


def validate_predict_inputs_strict(
    row: Dict[str, object],
    abi: Dict[str, int],
    *,
    abi_widths: Dict[str, int] | None = None,
) -> None:
    """Require all ABI keys present and reject unknown feature keys."""
    aw = dict(abi_widths or {})
    missing = [name for name in abi if name not in row]
    if missing:
        raise StrictInferenceError(f"missing required ABI input(s): {', '.join(sorted(missing))}")
    allowed = set(abi.keys())
    unknown = [k for k in row.keys() if str(k) not in allowed]
    if unknown:
        raise StrictInferenceError(f"unknown input key(s): {', '.join(sorted(str(k) for k in unknown))}")
    for name in abi:
        val = row[name]
        w = max(1, int(aw.get(name, 1)))
        if w > 1 and not isinstance(val, (list, tuple)):
            raise StrictInferenceError(f"ABI column {name!r} expects a length-{w} vector")


def mark_defined(env_defined: Set[str], name: str) -> None:
    env_defined.add(str(name))
