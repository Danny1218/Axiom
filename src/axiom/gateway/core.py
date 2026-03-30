"""Reusable policy gate + optional downstream forward (no vendor-specific URLs in core)."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from axiom.api import AxiomModel
from axiom.tools.html_exporter import render_html_report

_JSON_BODY = Callable[[str], Dict[str, Any]]

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def default_scan_text(message: str, *, rng: random.Random | None = None) -> Dict[str, float]:
    """Lightweight demo scanner: SSN-shaped PII, competitor keywords, random toxicity [0, 0.3).

    Override by passing pre-computed ``signals`` into the gateway HTTP handler instead of relying on this.
    """
    r = rng or random.Random()
    has_pii = 1.0 if _SSN_RE.search(message) else 0.0
    low = message.lower()
    comp = 1.0 if ("openai" in low or "anthropic" in low) else 0.0
    return {
        "has_pii_data": has_pii,
        "mentions_competitor": comp,
        "text_toxicity": r.uniform(0.0, 0.3),
    }


def resolve_signals(
    message: str,
    signals: Optional[Dict[str, float]],
    *,
    scan_fn: Callable[[str], Dict[str, float]] | None = None,
) -> Dict[str, float]:
    """Use ``signals`` when provided; else ``scan_fn`` (default :func:`default_scan_text`)."""
    if signals is not None:
        return dict(signals)
    fn = scan_fn or default_scan_text
    return fn(message)


def policy_explain(model: AxiomModel, signals: Dict[str, float]) -> Dict[str, Any]:
    """Run :meth:`AxiomModel.explain` on the feature row (policy trunk)."""
    return model.explain(signals)


def is_approved(trace: Dict[str, Any], *, threshold: float = 0.5) -> bool:
    return float(trace.get("is_approved", 0.0)) >= float(threshold)


def build_block_audit(
    model: AxiomModel,
    signals: Dict[str, float],
    *,
    source_code: str | None,
    audit_path: str | Path | None = None,
) -> Tuple[str, str | None]:
    """Return Glass Box HTML (same content as :meth:`AxiomModel.export_report`) and optional filesystem path.

    If ``audit_path`` is set, also writes via :meth:`AxiomModel.export_report`.
    """
    html = render_html_report(model, signals, source_code)
    written: str | None = None
    if audit_path is not None:
        p = Path(audit_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        model.export_report(signals, str(p.resolve()), source_code=source_code)
        written = str(p.resolve())
    return html, written


def forward_to_downstream(
    url: str,
    message: str,
    *,
    post_fn: Callable[[str, Dict[str, Any]], Any] | None = None,
    json_body: Dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> Any:
    """POST JSON to ``url``. ``post_fn(url, body)`` overrides HTTP (tests / custom transports)."""
    body = json_body if json_body is not None else {"message": message}
    if post_fn is not None:
        return post_fn(url, body)
    try:
        import requests
    except ImportError as e:
        raise RuntimeError('downstream forward requires: pip install -e ".[gateway]"') from e
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    ct = r.headers.get("content-type", "")
    if "application/json" in ct:
        return r.json()
    return {"raw": r.text}
