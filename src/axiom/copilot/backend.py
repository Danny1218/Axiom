"""Construct :class:`~axiom.experts.base.SemanticExpert` instances for copilot CLI, Studio, and HTTP server."""

from __future__ import annotations

from typing import Optional, Tuple

from axiom.experts.base import SemanticExpert
from axiom.experts.onyx_qwen import LMSTUDIO_DEFAULT_MODEL, LMSTUDIO_DEFAULT_URL

_HTTP_BACKENDS = frozenset({"onyx-qwen", "lmstudio"})


def build_onyx_qwen_expert(
    *,
    url: str,
    model: str,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> SemanticExpert:
    """Return :class:`~axiom.experts.onyx_qwen.OnyxQwenBackend` (requires ``[copilot]`` / ``requests``)."""
    try:
        import requests  # noqa: F401
    except ImportError as e:
        raise ImportError(
            'Onyx/Qwen expert requires requests. Install with: pip install -e ".[copilot]"'
        ) from e
    from axiom.experts.onyx_qwen import OnyxQwenBackend

    u, m = url.strip(), model.strip()
    if not u:
        raise ValueError("expert url is empty")
    if not m:
        raise ValueError("expert model is empty")
    key = api_key.strip() if api_key and str(api_key).strip() else None
    to = float(timeout) if timeout is not None else 120.0
    return OnyxQwenBackend(u, m, api_key=key, timeout=to)


def build_benchmark_dispatch_expert() -> SemanticExpert:
    """Return the deterministic offline benchmark expert used by tests and CI."""
    from axiom.copilot.benchmarks import BenchmarkDispatchExpert

    return BenchmarkDispatchExpert()


def resolve_copilot_http_settings(
    backend: str,
    *,
    expert_url: str,
    expert_model: str,
    expert_api_key: Optional[str] = None,
) -> Tuple[str, str, str, Optional[str]]:
    """Normalize backend alias and apply default URL/model for ``lmstudio``."""
    b = backend.strip().lower().replace("_", "-")
    url = (expert_url or "").strip()
    model = (expert_model or "").strip()
    key = expert_api_key.strip() if expert_api_key and str(expert_api_key).strip() else None
    if b == "lmstudio":
        if not url:
            url = LMSTUDIO_DEFAULT_URL
        if not model:
            model = LMSTUDIO_DEFAULT_MODEL
        return "onyx-qwen", url, model, key
    if b == "onyx-qwen":
        if not url:
            raise ValueError("expert url is empty")
        if not model:
            raise ValueError("expert model is empty")
        return b, url, model, key
    raise ValueError(f"unsupported copilot backend {backend!r} (expected onyx-qwen, lmstudio, or benchmark-dispatch)")


def build_copilot_expert(
    backend: str,
    *,
    expert_url: str,
    expert_model: str,
    expert_api_key: Optional[str] = None,
    timeout: Optional[float] = None,
) -> SemanticExpert:
    """Dispatch by ``backend`` name."""
    b = backend.strip().lower().replace("_", "-")
    if b == "benchmark-dispatch":
        return build_benchmark_dispatch_expert()
    if b not in _HTTP_BACKENDS:
        raise ValueError(
            f"unsupported copilot backend {backend!r} (expected onyx-qwen, lmstudio, or benchmark-dispatch)"
        )
    _, url, model, key = resolve_copilot_http_settings(
        b, expert_url=expert_url, expert_model=expert_model, expert_api_key=expert_api_key
    )
    return build_onyx_qwen_expert(url=url, model=model, api_key=key, timeout=timeout)


__all__ = [
    "build_benchmark_dispatch_expert",
    "build_copilot_expert",
    "build_onyx_qwen_expert",
    "resolve_copilot_http_settings",
]
