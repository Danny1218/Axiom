"""Universal LLM-output normalizer: canonicalize almost-valid ``.ax`` before parsing."""

from __future__ import annotations

from typing import Dict, Tuple

from axiom.compiler import normalizer_rewrites as nr

_REWRITE_STEPS = (
    nr.strip_line_comments_and_trailing_prose,
    nr.rewrite_shorthand_assignments,
    nr.rewrite_three_arg_extrema,
    nr.rewrite_clip_calls,
    nr.rewrite_else_if,
    nr.rewrite_logical_operators,
    nr.rewrite_inline_ternaries,
    nr.normalize_conservative,
)


def normalize_ax_source(ax: str) -> Tuple[str, Dict[str, bool]]:
    """Run deterministic source-to-source rewrites; metadata flags note what changed."""
    out = ax
    meta: Dict[str, bool] = {}
    for step in _REWRITE_STEPS:
        out, step_meta = step(out)
        meta.update(step_meta)
    return out.strip(), meta


__all__ = ["normalize_ax_source"]
