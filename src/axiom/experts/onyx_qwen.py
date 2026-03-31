"""HTTP expert: OpenAI-style chat completions against an Onyx / Qwen-compatible server.

Install: ``pip install -e ".[copilot]"`` (pulls ``requests``). Not imported from ``axiom`` root.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, NamedTuple, Optional
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
    "You write programs in THIS repository's custom `.ax` DSL (Axiom engine). "
    "It is NOT Macaulay2, NOT the Axiom computer algebra system, NOT a theorem prover, "
    "and NOT generic Python or pseudocode. "
    "Use JavaScript-like statements terminated with semicolons. "
    "Use `=` for assignment (never `:=`). "
    "Use `if (condition) { ... } else { ... }` and `while (condition) { ... }`. "
    "Do not use `print`. Do not emit prose, commentary, or explanations unless the user explicitly asks for them. "
    "When you use a markdown fence, use the info string `ax` so the block is ```ax ... ```.\n"
    "Canonical valid examples:\n"
    "  y = x * 2.0;\n"
    "  risk_score = max(0.0, min(1.0, 0.7 * risk_a + 0.3 * risk_b));"
)

SYSTEM_REPAIR = (
    "You fix programs in THIS repository's `.ax` DSL only (not Macaulay2, not Axiom CAS). "
    "Return ONLY the corrected full `.ax` program — no explanation, no preamble, no bullet points. "
    "Do not wrap in markdown unless you must; if you fence, use ```ax ... ```. "
    "Match syntax to this repo: `=` assignment, semicolon-terminated statements, `if`/`while` with braces, "
    "`neural(features)` or `neural(features, \"liquid\")`. Never use `:=` or `print`.\n"
    "Repair hint — bad → good: `x := 1` → `x = 1.0;` ; `print(y);` → delete or assign to an output variable instead."
)

SYSTEM_SUMMARY = (
    "You summarize symbolic execution traces for engineers. Be concise and factual; "
    "do not invent variables absent from the trace."
)

SYNTAX_SUMMARY = """Syntax summary (this repo's `.ax` DSL):
- Assignment with `=` (not `:=`). Statements end with `;`.
- Vectors like `[x, y]`. Use `neural(features)` or `neural(features, "liquid")`.
- Control flow: `if (cond) { ... } else { ... }`, `while (cond) { ... }`.
- No `print`."""

DRAFT_FEWSHOT = """Tiny example (valid `.ax`):
```ax
y = x * 2.0;
risk_score = max(0.0, min(1.0, 0.7 * risk_a + 0.3 * risk_b));
```"""

REPAIR_FEWSHOT = """Few-shot repair:
Bad: `score := max(a, b); print(score);`
Good:
```ax
score = max(a, b);
```"""


def _context_json(context: Mapping[str, Any]) -> str:
    return json.dumps(dict(context), sort_keys=True, separators=(",", ":"), default=str)


def user_prompt_draft(goal: str, context: Mapping[str, Any]) -> str:
    return (
        f"Goal:\n{goal}\n\n"
        f"Context (JSON, sorted keys):\n{_context_json(context)}\n\n"
        f"{SYNTAX_SUMMARY}\n\n"
        f"{DRAFT_FEWSHOT}\n\n"
        "Respond with the complete `.ax` program only (fenced with `ax` if you use a fence)."
    )


def user_prompt_repair(goal: str, current_program: str, error_report: str, context: Mapping[str, Any]) -> str:
    return (
        f"Goal:\n{goal}\n\n"
        f"Error report:\n{error_report}\n\n"
        f"Context (JSON, sorted keys):\n{_context_json(context)}\n\n"
        f"{SYNTAX_SUMMARY}\n\n"
        f"{REPAIR_FEWSHOT}\n\n"
        f"Current program:\n```ax\n{current_program.rstrip()}\n```\n\n"
        "Return the corrected full `.ax` program only (fenced with `ax` if you must)."
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


# Fenced blocks: explicit `ax` first; then generic ```lang
_AX_FENCE = re.compile(r"```ax\s*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL)
_ALL_FENCES = re.compile(r"```([^\n]*)\r?\n(.*?)```", re.DOTALL)
_SKIP_FENCE_LANGS = frozenset({"m2", "macaulay2", "macaulay"})


def _remove_skipped_fence_blocks(text: str) -> str:
    """Drop ```m2``` / Macaulay2-style fences so line heuristics do not treat them as ``.ax``."""
    out: list[str] = []
    last = 0
    for m in _ALL_FENCES.finditer(text):
        lang = (m.group(1) or "").strip().lower()
        if lang in _SKIP_FENCE_LANGS:
            out.append(text[last : m.start()])
            out.append("\n")
            last = m.end()
    out.append(text[last:])
    return "".join(out)

_PRINT_CALL = re.compile(r"\bprint\s*\(", re.IGNORECASE)


class AxSplitResult(NamedTuple):
    ax_source: str
    prose: Optional[str]
    extraction: dict[str, Any]


def _forbidden_flags(ax: str, raw: str) -> dict[str, Any]:
    flags: list[str] = []
    for _, blob in (("ax_source", ax), ("raw_response", raw)):
        if ":=" in blob:
            flags.append("assign_colon_eq")
        if _PRINT_CALL.search(blob):
            flags.append("print_call")
    if not flags:
        return {}
    return {"forbidden_tokens_detected": sorted(set(flags))}


def _line_code_score(line: str) -> int:
    s = line.strip()
    if not s:
        return 0
    score = 0
    if s.endswith(";"):
        score += 3
    if "if " in s or "if(" in s or "while " in s or "while(" in s:
        score += 2
    if "neural(" in s:
        score += 2
    if "max(" in s or "min(" in s:
        score += 1
    if re.search(r"[A-Za-z_][\w.]*\s*=\s*[^=]", s) and ":=" not in s:
        score += 2
    if len(s) > 220:
        score -= 3
    if s.count(" ") > 14 and not s.endswith(";"):
        score -= 2
    return score


def _best_code_run(lines: list[str]) -> tuple[int, int] | None:
    """Return (start, end) line indices of best contiguous run of code-like lines (end exclusive)."""
    scores = [_line_code_score(l) for l in lines]
    best_start, best_end, best_sum = 0, 0, -1
    n = len(lines)
    for i in range(n):
        if scores[i] <= 0:
            continue
        total = 0
        for j in range(i, n):
            if scores[j] <= 0:
                break
            total += scores[j]
            if total > best_sum:
                best_sum = total
                best_start, best_end = i, j + 1
    if best_sum <= 0:
        return None
    return best_start, best_end


def _score_fence_body(body: str) -> int:
    ls = body.strip().splitlines()
    if not ls:
        return 0
    run = _best_code_run(ls)
    if run is None:
        return sum(_line_code_score(l) for l in ls)
    a, b = run
    return sum(_line_code_score(l) for l in ls[a:b])


def split_ax_and_prose(raw: str) -> AxSplitResult:
    """Extract `.ax` from model output: prefer fenced ``ax``, then best non-Macaulay fence, then code-like lines."""
    text = raw.strip()
    extraction: dict[str, Any] = {"extraction_mode": "plain_fallback"}

    m_ax = _AX_FENCE.search(text)
    if m_ax:
        ax = m_ax.group(1).strip()
        before, after = text[: m_ax.start()].strip(), text[m_ax.end() :].strip()
        prose_parts = [p for p in (before, after) if p]
        explanation = "\n\n".join(prose_parts) if prose_parts else None
        extraction["extraction_mode"] = "fenced_ax"
        extraction.update(_forbidden_flags(ax, raw))
        return AxSplitResult(ax, explanation, extraction)

    best_body: str | None = None
    best_score = -1
    best_span: tuple[int, int] | None = None
    for m in _ALL_FENCES.finditer(text):
        lang = (m.group(1) or "").strip().lower()
        if lang in _SKIP_FENCE_LANGS:
            continue
        body = m.group(2).strip()
        sc = _score_fence_body(body)
        if sc > best_score:
            best_score = sc
            best_body = body
            best_span = (m.start(), m.end())

    if best_body is not None and best_score > 0:
        start, end = best_span  # type: ignore[misc]
        before, after = text[:start].strip(), text[end:].strip()
        prose_parts = [p for p in (before, after) if p]
        explanation = "\n\n".join(prose_parts) if prose_parts else None
        extraction["extraction_mode"] = "fenced_non_ax"
        extraction.update(_forbidden_flags(best_body, raw))
        return AxSplitResult(best_body, explanation, extraction)

    heur_text = _remove_skipped_fence_blocks(text)
    lines = heur_text.splitlines()
    run = _best_code_run(lines)
    if run is None:
        extraction.update(_forbidden_flags(text, raw))
        return AxSplitResult(text.strip(), None, extraction)

    a, b = run
    ax = "\n".join(lines[a:b]).strip()
    prose_before = "\n".join(lines[:a]).strip()
    prose_after = "\n".join(lines[b:]).strip()
    prose_parts = [p for p in (prose_before, prose_after) if p]
    explanation = "\n\n".join(prose_parts) if prose_parts else None
    extraction["extraction_mode"] = "heuristic_lines"
    extraction.update(_forbidden_flags(ax, raw))
    return AxSplitResult(ax, explanation, extraction)


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

    def _metadata(self, raw: str, split: AxSplitResult) -> dict[str, Any]:
        return {"model": self._model, "raw_chars": len(raw), **split.extraction}

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        raw = self._chat(SYSTEM_DRAFT, user_prompt_draft(request.goal, request.context))
        split = split_ax_and_prose(raw)
        return ExpertDraftResponse(
            ax_source=split.ax_source,
            backend_name=BACKEND_NAME,
            explanation=split.prose,
            metadata=self._metadata(raw, split),
        )

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        raw = self._chat(
            SYSTEM_REPAIR,
            user_prompt_repair(request.goal, request.current_program, request.error_report, request.context),
        )
        split = split_ax_and_prose(raw)
        return ExpertDraftResponse(
            ax_source=split.ax_source,
            backend_name=BACKEND_NAME,
            explanation=split.prose,
            metadata=self._metadata(raw, split),
        )

    def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
        return self._chat(
            SYSTEM_SUMMARY,
            user_prompt_trace_summary(
                request.goal, request.program, request.trace, request.metrics, request.context
            ),
        )


__all__ = [
    "AxSplitResult",
    "BACKEND_NAME",
    "DRAFT_FEWSHOT",
    "OnyxQwenBackend",
    "OnyxQwenError",
    "OnyxQwenHTTPError",
    "OnyxQwenParseError",
    "OnyxQwenTimeoutError",
    "OnyxQwenTransportError",
    "REPAIR_FEWSHOT",
    "SYSTEM_DRAFT",
    "SYSTEM_REPAIR",
    "SYNTAX_SUMMARY",
    "user_prompt_draft",
    "user_prompt_repair",
    "user_prompt_trace_summary",
    "split_ax_and_prose",
]
