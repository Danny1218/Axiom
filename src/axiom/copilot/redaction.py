"""Redact sensitive fields before persisting copilot / Onyx artifacts."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, MutableMapping

_REDACT_KEY = re.compile(
    r"(api[_-]?key|authorization|password|secret|bearer|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)
_MAX_PROMPT_CHARS = 512


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def redact_mapping(obj: Mapping[str, Any], *, redact_prompts: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        k = str(key)
        if _REDACT_KEY.search(k):
            out[k] = "<redacted>"
            continue
        if redact_prompts and k in {"system_prompt", "user_prompt", "payload"}:
            if k == "payload" and isinstance(value, Mapping):
                out[k] = redact_mapping(value, redact_prompts=True)
                continue
            if isinstance(value, str):
                if len(value) <= _MAX_PROMPT_CHARS:
                    out[k] = value
                else:
                    out[k] = {
                        "redacted": True,
                        "sha256_prefix": _hash_text(value),
                        "char_count": len(value),
                        "preview": value[:120],
                    }
                continue
        if isinstance(value, Mapping):
            out[k] = redact_mapping(value, redact_prompts=redact_prompts)
        elif isinstance(value, list):
            out[k] = [
                redact_mapping(v, redact_prompts=redact_prompts) if isinstance(v, Mapping) else v
                for v in value
            ]
        else:
            out[k] = value
    return out


def capture_mode_from_env() -> str:
    import os

    mode = os.environ.get("AXIOM_ONYX_CAPTURE_MODE", "").strip().lower()
    if mode in ("full", "unsafe", "debug"):
        return "full"
    return "redacted"
