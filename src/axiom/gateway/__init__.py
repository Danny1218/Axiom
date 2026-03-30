"""Policy gateway (optional): scan → explain → approve/block → audit / downstream forward.

``create_gateway_app`` / ``gateway_app_from_env`` implement ``POST /gateway/chat``. The symbol
``create_app`` here is the **gateway** uvicorn factory (env-based), not ``axiom.serve.create_app``.
Semantic-copilot designs can treat the gateway as a policy shell with a pluggable downstream URL.
"""

from axiom.gateway.core import (
    build_block_audit,
    default_scan_text,
    forward_to_downstream,
    is_approved,
    policy_explain,
    resolve_signals,
)
from axiom.gateway.server import create_app, create_gateway_app, gateway_app_from_env

__all__ = [
    "build_block_audit",
    "create_app",
    "create_gateway_app",
    "default_scan_text",
    "forward_to_downstream",
    "gateway_app_from_env",
    "is_approved",
    "policy_explain",
    "resolve_signals",
]
