"""Construct :class:`~axiom.experts.base.SemanticExpert` instances for copilot CLI, Studio, and HTTP server."""

from __future__ import annotations

from typing import Optional

from axiom.experts.base import SemanticExpert


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
    if b != "onyx-qwen":
        raise ValueError(f"unsupported copilot backend {backend!r} (expected onyx-qwen or benchmark-dispatch)")
    return build_onyx_qwen_expert(
        url=expert_url, model=expert_model, api_key=expert_api_key, timeout=timeout
    )


__all__ = ["build_benchmark_dispatch_expert", "build_copilot_expert", "build_onyx_qwen_expert"]
