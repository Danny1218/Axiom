"""HTTP expert: OpenAI-style chat completions against an Onyx / Qwen-compatible server.

Install: ``pip install -e ".[copilot]"`` (pulls ``requests``). Not imported from ``axiom`` root.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urljoin

from axiom.experts.base import (
    ExpertDraftRequest,
    ExpertDraftResponse,
    ExpertRepairRequest,
    ExpertTraceSummaryRequest,
)

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

BACKEND_NAME = "onyx_qwen"

# --- Deterministic prompt templates (edit here only) ---

SYSTEM_DRAFT = (
    "You write programs in the Axiom .ax language (JavaScript-like syntax: assignments, if/else, "
    "while, neural([...]) calls). Output executable .ax source. When you use a markdown fence, "
    "use the info string `ax` so the block is ```ax ... ```."
)

SYSTEM_REPAIR = (
    "You fix Axiom .ax programs. Preserve intent; address the error report. "
    "Prefer a single fenced block ```ax ... ``` containing the full corrected program."
)

SYSTEM_SUMMARY = (
    "You summarize symbolic execution traces for engineers. Be concise and factual; "
    "do not invent variables absent from the trace."
)


def _context_json(context: Mapping[str, Any]) -> str:
    return json.dumps(dict(context), sort_keys=True, separators=(",", ":"), default=str)


def user_prompt_draft(goal: str, context: Mapping[str, Any]) -> str:
    return (
        f"Goal:\n{goal}\n\n"
        f"Context (JSON, sorted keys):\n{_context_json(context)}\n\n"
        "Respond with the complete .ax program. Use a fenced code block with tag `ax` if possible."
    )


def user_prompt_repair(goal: str, current_program: str, error_report: str, context: Mapping[str, Any]) -> str:
    return (
        f"Goal:\n{goal}\n\n"
        f"Error report:\n{error_report}\n\n"
        f"Context (JSON, sorted keys):\n{_context_json(context)}\n\n"
        "Current program:\n```ax\n"
        f"{current_program.rstrip()}\n"
        "```\n\n"
        "Return the corrected full .ax program (fenced with `ax` if possible)."
    )


def user_prompt_trace_summary(
    goal: str, program: str, trace: Mapping[str, Any], metrics: Mapping[str, Any], context: Mapping[str, Any]
) -> str:
    return (
        f"Goal:\n{goal}\n\n"
        f"Program (.ax):\n```ax\n{program.rstrip()}\n```\n\n"
        f"Trace (JSON):\n{_context_json(trace)}\n\n"
        f"Metrics (JSON):\n{_context_json(metrics)}\n\n"
        f"Extra context (JSON):\n{_context_json(context)}\n\n"
        "Summarize what happened in plain English for a developer (no code fence required)."
    )


# First ```ax ... ``` or generic ``` ... ``` block
_AX_FENCE = re.compile(r"```(?:ax)?\s*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL)


def split_ax_and_prose(raw: str) -> tuple[str, Optional[str]]:
    """Prefer fenced ``ax`` (or bare ```) body; else use full text as program."""
    text = raw.strip()
    m = _AX_FENCE.search(text)
    if not m:
        return text, None
    ax = m.group(1).strip()
    before, after = text[: m.start()].strip(), text[m.end() :].strip()
    prose_parts = [p for p in (before, after) if p]
    explanation = "\n\n".join(prose_parts) if prose_parts else None
    return ax, explanation


class OnyxQwenError(Exception):
    """Base for Onyx/Qwen HTTP expert failures."""


class OnyxQwenTimeoutError(OnyxQwenError):
    """Request exceeded ``timeout``."""


class OnyxQwenTransportError(OnyxQwenError):
    """Network / connection failure before a response."""


class OnyxQwenHTTPError(OnyxQwenError):
    def __init__(self, status_code: int, body_snippet: str) -> None:
        self.status_code = status_code
        self.body_snippet = body_snippet
        super().__init__(f"HTTP {status_code}: {body_snippet[:200]}")


class OnyxQwenParseError(OnyxQwenError):
    """Invalid JSON or unexpected response shape."""


PostFn = Callable[..., Any]


def _assistant_content(data: Any) -> str:
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as e:
        raise OnyxQwenParseError("response missing choices[0].message.content") from e


class OnyxQwenBackend:
    """``SemanticExpert`` over an OpenAI-compatible ``/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 120.0,
        api_key: Optional[str] = None,
        chat_path: str = "/v1/chat/completions",
        _post: Optional[PostFn] = None,
    ) -> None:
        self._base = base_url.rstrip("/") + "/"
        self._chat_url = urljoin(self._base, chat_path.lstrip("/"))
        self._model = model
        self._timeout = float(timeout)
        self._api_key = api_key.strip() if api_key else None
        self._post = _post

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _resolve_post(self) -> PostFn:
        if self._post is not None:
            return self._post
        if requests is None:
            raise ImportError(
                'OnyxQwenBackend requires requests. Install with: pip install -e ".[copilot]"'
            )
        return requests.post

    def _chat(self, system: str, user: str) -> str:
        post = self._resolve_post()

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            r = post(
                self._chat_url,
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except Exception as e:
            if requests is not None and isinstance(e, requests.exceptions.Timeout):
                raise OnyxQwenTimeoutError(str(e)) from e
            if requests is not None and isinstance(e, requests.exceptions.RequestException):
                raise OnyxQwenTransportError(str(e)) from e
            raise

        if r.status_code >= 400:
            snippet = r.text if isinstance(r.text, str) else ""
            raise OnyxQwenHTTPError(r.status_code, snippet[:2000])

        try:
            data = r.json()
        except ValueError as e:
            raise OnyxQwenParseError("response body is not valid JSON") from e

        return _assistant_content(data)

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        raw = self._chat(SYSTEM_DRAFT, user_prompt_draft(request.goal, request.context))
        ax, expl = split_ax_and_prose(raw)
        return ExpertDraftResponse(
            ax_source=ax,
            backend_name=BACKEND_NAME,
            explanation=expl,
            metadata={"model": self._model, "raw_chars": len(raw)},
        )

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        raw = self._chat(
            SYSTEM_REPAIR,
            user_prompt_repair(request.goal, request.current_program, request.error_report, request.context),
        )
        ax, expl = split_ax_and_prose(raw)
        return ExpertDraftResponse(
            ax_source=ax,
            backend_name=BACKEND_NAME,
            explanation=expl,
            metadata={"model": self._model, "raw_chars": len(raw)},
        )

    def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
        return self._chat(
            SYSTEM_SUMMARY,
            user_prompt_trace_summary(
                request.goal, request.program, request.trace, request.metrics, request.context
            ),
        )


__all__ = [
    "BACKEND_NAME",
    "OnyxQwenBackend",
    "OnyxQwenError",
    "OnyxQwenHTTPError",
    "OnyxQwenParseError",
    "OnyxQwenTimeoutError",
    "OnyxQwenTransportError",
    "user_prompt_draft",
    "user_prompt_repair",
    "user_prompt_trace_summary",
    "split_ax_and_prose",
]
