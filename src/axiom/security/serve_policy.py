"""Startup policy for ``axiom serve`` (auth required on public binds / production)."""

from __future__ import annotations

import os


class InsecureServeError(RuntimeError):
    """Refused to bind a production-style server without API authentication."""


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def is_public_bind(host: str) -> bool:
    h = host.strip().lower()
    return h in ("0.0.0.0", "::", "[::]", "")


def api_key_configured() -> bool:
    return bool(os.environ.get("AXIOM_API_KEY", "").strip())


def verify_serve_startup(host: str, *, allow_insecure: bool = False) -> None:
    """Fail closed when serving on a public interface without ``AXIOM_API_KEY``.

  Set ``AXIOM_ALLOW_INSECURE_SERVE=1`` or pass ``allow_insecure=True`` for local dev only.
  Docker / compose should set ``AXIOM_REQUIRE_API_KEY=1`` and a non-empty ``AXIOM_API_KEY``.
    """
    if allow_insecure or _truthy_env("AXIOM_ALLOW_INSECURE_SERVE"):
        return
    require_key = _truthy_env("AXIOM_REQUIRE_API_KEY") or is_public_bind(host)
    if require_key and not api_key_configured():
        raise InsecureServeError(
            "Refusing to start axiom serve without AXIOM_API_KEY while bound to "
            f"{host!r} (or AXIOM_REQUIRE_API_KEY is set). "
            "Set AXIOM_API_KEY for production serving, or AXIOM_ALLOW_INSECURE_SERVE=1 for "
            "local development only."
        )
