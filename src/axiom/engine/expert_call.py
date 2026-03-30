"""Runtime support for ``expert()`` (Phase 66): non-differentiable external semantic calls.

``OP_EXPERT`` is separate from ``OP_NEURAL``; handlers are optional Python callables, not ``nn.Module``.
"""

from __future__ import annotations

from typing import Callable, Sequence

# ``name`` is the backend key from source (first string literal); ``features`` is one batch row.
ExpertHandler = Callable[[str, Sequence[float]], float]


class ExpertRuntimeError(RuntimeError):
    """Raised when ``expert()`` runs with no ``expert_handler`` and no ``expert_fallback``."""
