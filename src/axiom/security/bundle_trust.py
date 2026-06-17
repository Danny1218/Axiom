"""Trust boundary helpers for ``.axb`` loading and HTTP report writes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class BundleTrustError(RuntimeError):
    """Refused to unpickle an untrusted ``.axb`` bundle."""


def bundle_trust_from_env() -> bool:
    v = os.environ.get("AXIOM_TRUST_BUNDLE", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def resolve_trusted(trusted: Optional[bool]) -> bool:
    if trusted is None:
        return bundle_trust_from_env()
    return bool(trusted)


def report_output_dir_from_env() -> Optional[Path]:
    raw = os.environ.get("AXIOM_REPORT_OUTPUT_DIR", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def resolve_report_output_path(requested: str, sandbox: Path) -> Path:
    """Resolve ``requested`` under ``sandbox``; reject path escape."""
    if not requested or not str(requested).strip():
        raise ValueError("output_path must be a non-empty relative path")
    sandbox_res = sandbox.expanduser().resolve()
    sandbox_res.mkdir(parents=True, exist_ok=True)
    req = Path(str(requested).strip())
    if req.is_absolute():
        raise ValueError("output_path must be relative to the configured report sandbox")
    out = (sandbox_res / req).resolve()
    try:
        out.relative_to(sandbox_res)
    except ValueError as e:
        raise ValueError(f"output_path escapes report sandbox: {requested!r}") from e
    return out
