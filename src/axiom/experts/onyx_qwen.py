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

# Merged into OpenAI-style chat ``payload`` for ``draft_program`` only; stripped from user context JSON.
COMPLETION_OVERRIDES_CONTEXT_KEY = "_onyx_completion_overrides"

# --- Deterministic prompt templates (edit here only) ---

RETURN_VALID_AX_SEMICOLON_LINE = (
    "Return only valid .ax source. Every assignment statement must end with a semicolon."
)

_CANONICAL_SYMBOLIC_FAMILY_DRAFTS_BLOCK = """Positive anchors - canonical symbolic family drafts:
If the goal matches one of these families, emit that canonical form directly and do not add validation/range-guard branches unless the goal explicitly asks for them.

- `double_x`
```ax
y = x * 2.0;
```

- `risk_score` clamped blend
```ax
risk_score = max(0.0, min(1.0, 0.7 * risk_a + 0.3 * risk_b));
```

- `three_input_affine`
```ax
score = 0.5 * a + 0.3 * b + 0.2 * c;
```

- `minmax_blend`
```ax
score = max(0.0, min(a + b, 1.0));
```

- `quadratic_with_cross_term`
```ax
y = a * b + a + 1.0;
```

- `three_way_maxmin`
```ax
score = max(min(a, b), c);
```

- `nested_piecewise`
```ax
if (x < 0.0) {
    y = 0.0;
} else {
    if (x < 1.0) {
        y = x;
    } else {
        y = 1.0;
    }
}
```

- `max_of_two`
```ax
if (a > b) {
    score = a;
} else {
    score = b;
}
```"""

_CANONICAL_SYMBOLIC_FAMILY_IDS = frozenset(
    {
        "double_x",
        "risk_score",
        "three_input_affine",
        "minmax_blend",
        "quadratic_with_cross_term",
        "three_way_maxmin",
        "nested_piecewise",
        "max_of_two",
    }
)

_CANONICAL_SYMBOLIC_FAMILY_HINTS = (
    re.compile(r"\bdouble_x\b|\bdouble of x\b|\by\s*=\s*x\s*\*\s*2(?:\.0)?\b", re.I | re.S),
    re.compile(r"\brisk_score\b|\b0\.7\s*\*\s*risk_a\b[\s\S]*\b0\.3\s*\*\s*risk_b\b", re.I | re.S),
    re.compile(r"\bthree_input_affine\b|\b0\.5\s*\*\s*a\b[\s\S]*\b0\.3\s*\*\s*b\b[\s\S]*\b0\.2\s*\*\s*c\b", re.I | re.S),
    re.compile(r"\bminmax_blend\b|max\s*\(\s*0(?:\.0)?\s*,\s*min\(\s*a\s*\+\s*b\s*,\s*1(?:\.0)?\s*\)\s*\)", re.I | re.S),
    re.compile(r"\bquadratic_with_cross_term\b|\ba\s*\*\s*b\s*\+\s*a\s*\+\s*1(?:\.0)?\b", re.I | re.S),
    re.compile(r"\bthree_way_maxmin\b|max\s*\(\s*min\(\s*a\s*,\s*b\s*\)\s*,\s*c\s*\)", re.I | re.S),
    re.compile(
        r"\bnested_piecewise\b|\bif\b[\s\S]*\bx\s*<\s*0(?:\.0)?\b[\s\S]*\belse\b[\s\S]*\bif\b[\s\S]*\bx\s*<\s*1(?:\.0)?\b[\s\S]*\by\s*=\s*x\b[\s\S]*\belse\b[\s\S]*\by\s*=\s*1(?:\.0)?\b",
        re.I | re.S,
    ),
    re.compile(r"\bmax_of_two\b|\bscore\s*=\s*max\(\s*a\s*,\s*b\s*\)|\bmax\(\s*a\s*,\s*b\s*\)", re.I | re.S),
)

SYSTEM_DRAFT = (
    "You write programs in THIS repository's custom `.ax` DSL (Axiom engine). "
    "It is NOT Macaulay2, NOT the Axiom computer algebra system, NOT a theorem prover, "
    "and NOT generic Python or pseudocode. "
    "Use JavaScript-like statements terminated with semicolons. "
    "Use `=` for assignment (never `:=`). "
    "Use `if (condition) { ... } else { ... }` and `while (condition) { ... }`. "
    "Nested control-flow (draft + search): explicitly forbid `else if`, `&&`, `||`, "
    "chained comparisons such as `a < b < c` or `0.9999<x<1`, `==` in assignment position, and missing semicolons. "
    "Rewrite `else if` as `else { if (...) { ... } else { ... } }`; split chained bounds with nested `if`/`else`; "
    "use `y = x;` for assignment, never `y == x`. "
    "For nested piecewise tasks, copy the canonical nested `if` / `else` structure from the prompt exactly: "
    "one comparison per `if`, nested under `else`, never `else if`, `&&`, `||`, or chained comparisons. "
    "If a piecewise program is needed, always use nested `else { if (...) { ... } else { ... } }`. "
    "Forbidden tokens: `:=`, `>=`, `<=`, `&&`, `||`, `then`, `else if`; float literals must be canonical (e.g. `2.0`, never bare `2.`). "
    "Do not use dotted variable access like `input.a` or `obj.value`; use direct variables only (e.g. `a`, `b`, `c`, `x`, `y`, `score`). "
    "Use only well-formed `if`/`else` branches with braces; do not invent invalid branch structure. "
    "Do not emit malformed literals like `0.0 0` or `1.0 0`. "
    "Comparisons allowed: `>`, `<`, `==`, `!=` only. "
    "Do not use `print`. Do not emit prose, commentary, or explanations unless the user explicitly asks for them. "
    "If the user prompt includes a canonical symbolic-family anchor, follow it exactly when relevant. "
    "When you use a markdown fence, use the info string `ax` so the block is ```ax ... ```.\n"
    + RETURN_VALID_AX_SEMICOLON_LINE
)

SYSTEM_REPAIR = (
    "You fix programs in THIS repository's `.ax` DSL only (not Macaulay2, not Axiom CAS). "
    "Return ONLY the corrected full `.ax` program — no explanation, no preamble, no bullet points. "
    "Do not wrap in markdown unless you must; if you fence, use ```ax ... ```. "
    "Match syntax to this repo: `=` assignment, semicolon-terminated statements, `if`/`while` with braces, "
    "`neural(features)` or `neural(features, \"liquid\")`. Never use `:=` or `print`. "
    "Nested control-flow (repair + search): explicitly forbid `else if`, `&&`, `||`, "
    "chained comparisons such as `a < b < c` or `0.9999<x<1`, `==` in assignment position, and missing semicolons. "
    "Rewrite `else if` as `else { if (...) { ... } else { ... } }`; split chained bounds with nested `if`/`else`; "
    "use `y = x;` for assignment, never `y == x`. "
    "For nested piecewise tasks, copy the canonical nested `if` / `else` structure from the prompt exactly: "
    "one comparison per `if`, nested under `else`, never `else if`, `&&`, `||`, or chained comparisons. "
    "If a piecewise program is needed, always use nested `else { if (...) { ... } else { ... } }`. "
    "Forbidden: `:=`, `>=`, `<=`, `&&`, `||`, `then`, `else if`; no bare float `2.` — use `2.0`. "
    "Do not use dotted variable access like `input.a` or `obj.value`; use direct variables only (e.g. `a`, `b`, `c`, `x`, `y`, `score`). "
    "Use only well-formed `if`/`else` branches with braces; do not invent invalid branch structure. "
    "Do not emit malformed literals like `0.0 0` or `1.0 0`. "
    "Comparisons: `>`, `<`, `==`, `!=` only; if parse errors mention stray `=`, `|`, or `.`, rewrite to supported operators and canonical floats.\n"
    "Repair hint — bad → good: `x := 1` → `x = 1.0;` ; `print(y);` → delete or assign to an output variable instead.\n"
    + RETURN_VALID_AX_SEMICOLON_LINE
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

FORBIDDEN_SYNTAX_BLOCK = """Forbidden syntax (not in this grammar — do not emit):
- Tokens: `:=`, `>=`, `<=`, `&&`, `||`, and the keyword `then`
- Unsupported control-flow patterns: `if (cond) then ...`, `if (cond) then { ... }`, `then (...)`
- Dotted variable access: `input.a`, `foo.bar`, `obj.value`
- Nested search stability — never emit: `else if` (as one keyword), `&&`, `||`, chained comparisons, `==` as assignment, or missing `;`
- `else if` as a single construct (nest explicitly with `else { if (...) { ... } else { ... } }`)
- Chained comparisons like `a < b < c`, `0.0 < x < 1.0`, `0.0<x<1.0`, or `0.9999<x<1` — use nested `if`/`else` only (never chained comparisons)
- `==` in assignment position (bad: `y == x` — use `y = x;`)
- Missing semicolons after assignments/statements (every `y = ...` line ends with `;`)
- Malformed numeric literals like `0.0 0` or `1.0 0`
- Float format: do not use a lone trailing dot (e.g. `2.`) — use canonical decimals like `2.0`"""

ALLOWED_SYNTAX_BLOCK = """Allowed syntax:
- Assignment: `y = x * 2.0;`
- `if`: `if (x > 0) { y = x; } else { y = 0.0; }`
- Use the real variable names from examples directly (`a`, `b`, `c`, `x`, `y`, `score`) — no dotted prefixes
- Nested branching form: `else { if (...) { ... } else { ... } }`
- Comparisons only: `>`, `<`, `==`, `!=` (express range checks with `min`/`max` and/or nested `if`/`else`, not `&&`/`||`/`>=`/`<=`)"""

REPAIR_PARSE_ERROR_RULE = """Repair-specific rule: if parse errors mention unexpected `=`, `|`, or `.`, rewrite using only grammar-supported operators and canonical floats like `2.0` (not `2.`)."""

SYNTAX_BAD_GOOD_FEWSHOT = """Few-shot bad → good rewrites:
1. `y := x * 2` → `y = x * 2.0;`
2. `if (x >= 0 && x <= 1) { ... }` → use only `>`, `<`, `==`, `!=`, and/or `min`/`max` with nested `if`/`else` — not `&&`, `||`, `>=`, or `<=`
3. `y = x + 2.;` → `y = x + 2.0;`
4. `score = max(min(input.a, input.b), input.c);` → `score = max(min(a, b), c);`
5. `x = 1.0 0;` → `x = 1.0;`"""

CONTROL_FLOW_FEWSHOT = """Control-flow few-shot rewrites (grammar-supported only):

Canonical nested piecewise example — copy this structure exactly for nested piecewise tasks; only change variable names, constants, and assigned expressions:
```ax
if (x < 0.0) {
    y = 0.0;
} else {
    if (x < 1.0) {
        y = x;
    } else {
        y = 1.0;
    }
}
```

If a piecewise program is needed, always use nested `else { if (...) { ... } else { ... } }`.

Bad → good (nested piecewise — apply these first):
1. bad: `else if (x < 1.0) { ... }` — good: `else { if (x < 1.0) { ... } else { ... } }` (never `else if`)
2. bad: `0.9999<x<1` — good: never use chained comparisons; copy the canonical nested piecewise structure above with one comparison per `if`
3. bad: `y == x;` — good: `y = x;`
4. bad: `if (x != 0.0 && x < 2.0) { ... }`
   good: `if (x != 0.0) { if (x < 2.0) { ... } else { ... } } else { ... }` (use nested `if`s, never `&&`)

More bad → good:
5. bad: `if (0.0<x<1.0) { y = x; }` or `if (0.0 < x < 1.0) { ... }`
   good: nested `if`/`else` as in the canonical example — never chained comparisons
6. `if (x == 0) then y = 0.0 else y = x;` →
```ax
if (x == 0) { y = 0.0; } else { y = x; }
```
7. `if (x < 0) then (y = 0.0) else (y = x)` →
```ax
if (x < 0) { y = 0.0; } else { y = x; }
```
8. bad: `if (x <= 0) { ... }`
   good: use only `<`, `>`, `==`, `!=` (no `<=`), with strict comparisons and nested `if`/`else` as needed.
9. Simple two-branch `if` (valid when you do not need a third region): `if (x < 0.0) { y = 0.0; } else { y = x; }`
10. bad: `if (x <= 0) { y = 0.0; } else { y = x; }` when you need an extra boundary
   good:
```ax
if (x < 0.0) {
    y = 0.0;
} else {
    if (x == 0.0) {
        y = 0.0;
    } else {
        y = x;
    }
}
```"""

DRAFT_FEWSHOT = _CANONICAL_SYMBOLIC_FAMILY_DRAFTS_BLOCK

REPAIR_FEWSHOT = """Few-shot repair:
Bad: `score := max(a, b); print(score);`
Good:
```ax
score = max(a, b);
```"""

EXAMPLES_SEMANTICS_BLOCK = """Example-driven semantics (when `example_input_rows` / `expected_outputs` are present):
- Return a SINGLE general `.ax` program — not one statement block per example row.
- Prefer direct symbolic arithmetic over `neural(...)` when the mapping is exact and simple.
- Use the actual variable names from the examples (e.g. `x`, `y`).
- Do NOT emit row-indexed variables like `x_0`, `x_1`, `y_0`, `y_1`.
- Do NOT unroll one line per example row.
- Do NOT use `output(...)` — it is not valid in this DSL.
- The program must work for arbitrary future inputs, not only the provided examples.
- The program must satisfy every provided example: matching inputs must produce the expected outputs.
- Do not ignore input variables; each output must follow from the inputs for that row.
- Do not return a constant placeholder unless it matches all examples simultaneously.

Unrolling bad → good:
Bad:
x_0 = 1.0;
y_0 = x_0 * 2.0;
x_1 = 2.5;
y_1 = x_1 * 2.0;
Good:
y = x * 2.0;

Constant shortcut bad → good:
Bad: `x = 5.0; y = x;`
Good: `y = x * 2.0;`"""

REPAIR_UNROLL_COLLAPSE_BLOCK = """Repair focus — unrolled or invalid I/O:
The current program uses row-indexed names (`x_0`, `y_1`, …) and/or `output(...)`. Collapse it into ONE reusable program over the real input/output names from the goal and examples (e.g. `x`, `y`). `output(...)` is invalid in this DSL — use assignments to named variables only.
Bad: `y_0 = x_0 * 2; ... output(y_0, y_1)`
Good: `y = x * 2.0;`"""

EXACT_SYMBOLIC_MATH_BLOCK = """Exact symbolic mapping (small example-driven math / affine / clamp tasks):
- Prefer **direct symbolic arithmetic** using `+`, `-`, `*`, `/`, `min`, `max`, and numeric literals.
- **Do NOT** use `neural(...)` unless the mapping truly cannot be expressed symbolically from the examples.
- If the goal/examples imply a **single closed-form arithmetic expression**, do **NOT** introduce `if` / `else` / `while`.
- For pure algebraic mappings, never introduce `if` / `else` / `while`.
- For a pure algebraic mapping, return **one direct assignment expression only** (for example, `y = ...;`).
- Never introduce boolean guard logic for pure algebraic mappings.
- Do not use `||` or `&&` under any circumstance.
- For affine blends and clamp-to-[0,1] style behavior, write explicit `max`, `min`, and arithmetic — not a learned head.
- Avoid malformed numerics: use `0.3` not `03` or ambiguous multi-dot literals.

Bad → good:
Bad: `if (a < 0 || b < 0) { y = ... } else { y = a*b + a + 1.0; }`
Good: `y = a * b + a + 1.0;`"""

REPAIR_NEURAL_TO_SYMBOLIC_BLOCK = """Repair focus — replace `neural(...)` with symbolic arithmetic (examples suggest an exact formula):
The current program uses `neural(...)`. When the goal and examples define a closed-form mapping (blend, clamp, affine), **replace** the `neural(...)` call with explicit `max` / `min` / arithmetic on the input variables.
Bad: `risk_score = neural([0.7*risk_a, 0.3*risk_b], "liquid");`
Good: `risk_score = max(min(0.7 * risk_a + 0.3 * risk_b, 1.0), 0.0);`"""

_INDEXED_VAR_PATTERN = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9]*_\d+\b")
_OUTPUT_CALL_PATTERN = re.compile(r"\boutput\s*\(", re.IGNORECASE)
_NEURAL_CALL_PATTERN = re.compile(r"\bneural\s*\(", re.IGNORECASE)
# Leading-zero integer literals like ``03`` (often invalid / typo); not ``0.3``.
_LEADING_ZERO_INT_LITERAL = re.compile(r"(?<![0-9.])0[0-9]+\b")
# Malformed multi-dot numbers (e.g. ``1.0.0``).
_MALFORMED_DECIMAL_PATTERN = re.compile(r"\d+\.\d+\.\d+")


def _program_has_neural(source: str) -> bool:
    return bool(_NEURAL_CALL_PATTERN.search(source))


def _suspicious_numeric_literals(source: str) -> bool:
    """Heuristic: likely-invalid numeric tokens (do not rewrite source here)."""
    if _LEADING_ZERO_INT_LITERAL.search(source):
        return True
    if _MALFORMED_DECIMAL_PATTERN.search(source):
        return True
    return False


def _program_has_indexed_variables(source: str) -> bool:
    return bool(_INDEXED_VAR_PATTERN.search(source))


def _program_has_output_call(source: str) -> bool:
    return bool(_OUTPUT_CALL_PATTERN.search(source))


def _pattern_warnings(ax: str) -> dict[str, Any]:
    """Metadata flags for likely per-row unrolling or invalid ``output(...)`` (source kept intact)."""
    out: dict[str, Any] = {}
    if _program_has_indexed_variables(ax):
        out["indexed_variable_warning"] = True
    if _program_has_output_call(ax):
        out["output_call_warning"] = True
    if _suspicious_numeric_literals(ax):
        out["suspicious_numeric_literal_warning"] = True
    return out


def ax_source_metadata_flags(source: str) -> dict[str, Any]:
    """Public: pattern warnings + ``uses_neural`` for copilot ranking (JSON-serializable flags)."""
    m = dict(_pattern_warnings(source))
    if _program_has_neural(source):
        m["uses_neural"] = True
    return m


def _append_repair_unroll_hints_if_needed(parts: list[str], current_program: str) -> None:
    if _program_has_indexed_variables(current_program) or _program_has_output_call(current_program):
        parts.append(REPAIR_UNROLL_COLLAPSE_BLOCK + "\n\n")


def _append_repair_neural_to_symbolic_if_needed(parts: list[str], current_program: str, context: Mapping[str, Any]) -> None:
    if not _context_has_examples_driven_semantics(context):
        return
    if not _program_has_neural(current_program):
        return
    parts.append(REPAIR_NEURAL_TO_SYMBOLIC_BLOCK + "\n\n")


def _append_exact_symbolic_math_if_needed(parts: list[str], context: Mapping[str, Any]) -> None:
    if not context.get("exact_symbolic_examples_task"):
        return
    if not _context_has_examples_driven_semantics(context):
        return
    parts.append(EXACT_SYMBOLIC_MATH_BLOCK + "\n\n")


def _goal_matches_known_canonical_symbolic_family(goal: str) -> bool:
    g = (goal or "").strip()
    if not g:
        return False
    return any(p.search(g) for p in _CANONICAL_SYMBOLIC_FAMILY_HINTS)


def _context_hints_known_canonical_symbolic_family(context: Mapping[str, Any]) -> bool:
    for key in ("benchmark_task_id", "task_id", "task_family", "family"):
        value = context.get(key)
        if isinstance(value, str) and value.strip().lower() in _CANONICAL_SYMBOLIC_FAMILY_IDS:
            return True
    return False


def _append_canonical_symbolic_family_drafts_if_needed(
    parts: list[str], goal: str, context: Mapping[str, Any]
) -> None:
    if context.get("exact_symbolic_examples_task"):
        parts.append(DRAFT_FEWSHOT + "\n\n")
        return
    if _context_hints_known_canonical_symbolic_family(context):
        parts.append(DRAFT_FEWSHOT + "\n\n")
        return
    if _goal_matches_known_canonical_symbolic_family(goal):
        parts.append(DRAFT_FEWSHOT + "\n\n")


def _context_has_examples_driven_semantics(context: Mapping[str, Any]) -> bool:
    eo = context.get("expected_outputs")
    ei = context.get("example_input_rows")
    return (isinstance(eo, list) and len(eo) > 0) or (isinstance(ei, list) and len(ei) > 0)


def _append_examples_semantics_if_needed(parts: list[str], context: Mapping[str, Any]) -> None:
    if _context_has_examples_driven_semantics(context):
        parts.append(EXAMPLES_SEMANTICS_BLOCK)


def _context_json(context: Mapping[str, Any]) -> str:
    return json.dumps(dict(context), sort_keys=True, separators=(",", ":"), default=str)


def user_prompt_draft(goal: str, context: Mapping[str, Any]) -> str:
    parts = [
        f"Goal:\n{goal}\n\n",
        f"Context (JSON, sorted keys):\n{_context_json(context)}\n\n",
    ]
    _append_examples_semantics_if_needed(parts, context)
    _append_exact_symbolic_math_if_needed(parts, context)
    _append_canonical_symbolic_family_drafts_if_needed(parts, goal, context)
    parts.extend(
        [
            f"{SYNTAX_SUMMARY}\n\n",
            f"{FORBIDDEN_SYNTAX_BLOCK}\n\n",
            f"{ALLOWED_SYNTAX_BLOCK}\n\n",
            f"{SYNTAX_BAD_GOOD_FEWSHOT}\n\n",
            f"{CONTROL_FLOW_FEWSHOT}\n\n",
            "Respond with the complete `.ax` program only (fenced with `ax` if you use a fence).\n"
            + RETURN_VALID_AX_SEMICOLON_LINE,
        ]
    )
    return "".join(parts)


def user_prompt_repair(goal: str, current_program: str, error_report: str, context: Mapping[str, Any]) -> str:
    parts = [
        f"Goal:\n{goal}\n\n",
        f"Error report:\n{error_report}\n\n",
        f"Context (JSON, sorted keys):\n{_context_json(context)}\n\n",
    ]
    _append_examples_semantics_if_needed(parts, context)
    _append_exact_symbolic_math_if_needed(parts, context)
    parts.extend(
        [
            f"{SYNTAX_SUMMARY}\n\n",
            f"{FORBIDDEN_SYNTAX_BLOCK}\n\n",
            f"{ALLOWED_SYNTAX_BLOCK}\n\n",
            f"{REPAIR_PARSE_ERROR_RULE}\n\n",
            f"{SYNTAX_BAD_GOOD_FEWSHOT}\n\n",
            f"{CONTROL_FLOW_FEWSHOT}\n\n",
            f"{REPAIR_FEWSHOT}\n\n",
        ]
    )
    _append_repair_unroll_hints_if_needed(parts, current_program)
    _append_repair_neural_to_symbolic_if_needed(parts, current_program, context)
    parts.extend(
        [
            f"Current program:\n```ax\n{current_program.rstrip()}\n```\n\n",
            "Return the corrected full `.ax` program only (fenced with `ax` if you must).\n"
            + RETURN_VALID_AX_SEMICOLON_LINE,
        ]
    )
    return "".join(parts)


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

_STRAY_LANG_TAGS = frozenset({"ax", "axiom", "javascript", "js"})


def _is_stray_lang_tag_line(line: str) -> bool:
    return line.strip().lower() in _STRAY_LANG_TAGS


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


def _rest_looks_code_like(rest: list[str]) -> bool:
    if not rest:
        return False
    for ln in rest:
        if _line_code_score(ln) > 0:
            return True
    blob = "\n".join(rest)
    return "=" in blob and ";" in blob


def _finalize_extracted_source(ax: str, extraction: dict[str, Any]) -> str:
    """Strip a lone first-line language tag (``ax`` / ``js`` / …) when the remainder is code-like; set counts."""
    lines = ax.splitlines()
    if lines and _is_stray_lang_tag_line(lines[0]) and len(lines) > 1 and _rest_looks_code_like(lines[1:]):
        extraction["stripped_language_tag"] = lines[0].strip().lower()
        lines = lines[1:]
        ax = "\n".join(lines).strip()
    extraction["code_line_count"] = len([ln for ln in ax.splitlines() if ln.strip()])
    return ax


# Lone `2.` → `2.0` without touching `2.0`, `3.14`, or the fractional part after `.` in `2.0.`
_TRAILING_DOT_FLOAT = re.compile(r"(?<!\.\d)(\d+)\.(?![0-9.])")
_STATEMENT_EQ_EQ_ASSIGN = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*)==(\s*)(.+;)\s*$")


def _normalize_ax_source_conservative(ax: str) -> tuple[str, dict[str, bool]]:
    """Small deterministic fixes on extracted source; sets flags only when something changed."""
    meta: dict[str, bool] = {}
    out = ax
    if ":=" in out:
        out = out.replace(":=", "=")
        meta["normalized_colon_eq"] = True
    lines = out.splitlines(keepends=True)
    rewritten_lines: list[str] = []
    changed_eqeq_assign = False
    for line in lines:
        line_body = line.rstrip("\r\n")
        newline = line[len(line_body) :]
        m = _STATEMENT_EQ_EQ_ASSIGN.match(line_body)
        if m:
            line = f"{m.group(1)}{m.group(2)}{m.group(3)}={m.group(4)}{m.group(5)}{newline}"
            changed_eqeq_assign = True
        rewritten_lines.append(line)
    if changed_eqeq_assign:
        out = "".join(rewritten_lines)
        meta["normalized_statement_eq_eq_assignment"] = True
    out2, n = _TRAILING_DOT_FLOAT.subn(r"\1.0", out)
    if n:
        out = out2
        meta["normalized_trailing_dot_float"] = True
    return out, meta


def _split_ax_result(ax: str, explanation: Optional[str], extraction: dict[str, Any], raw: str) -> AxSplitResult:
    extraction.update(_forbidden_flags(ax, raw))
    ax = _finalize_extracted_source(ax, extraction)
    ax, norm_meta = _normalize_ax_source_conservative(ax)
    extraction.update(norm_meta)
    extraction.update(_pattern_warnings(ax))
    return AxSplitResult(ax, explanation, extraction)


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
        return _split_ax_result(ax, explanation, extraction, raw)

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
        return _split_ax_result(best_body, explanation, extraction, raw)

    heur_text = _remove_skipped_fence_blocks(text)
    lines0 = heur_text.splitlines()
    prose_prefix = ""
    heur_lines = lines0
    if len(lines0) >= 2 and _is_stray_lang_tag_line(lines0[0]) and _rest_looks_code_like(lines0[1:]):
        prose_prefix = lines0[0].strip()
        heur_lines = lines0[1:]
        extraction["stripped_language_tag"] = prose_prefix

    run = _best_code_run(heur_lines)
    if run is None:
        ax_plain = text.strip()
        extraction["extraction_mode"] = "plain_fallback"
        return _split_ax_result(ax_plain, None, extraction, raw)

    a, b = run
    ax = "\n".join(heur_lines[a:b]).strip()
    prose_before = "\n".join(heur_lines[:a]).strip()
    prose_after = "\n".join(heur_lines[b:]).strip()
    if prose_prefix:
        prose_before = "\n\n".join(p for p in (prose_prefix, prose_before) if p)
    prose_parts = [p for p in (prose_before, prose_after) if p]
    explanation = "\n\n".join(prose_parts) if prose_parts else None
    extraction["extraction_mode"] = "heuristic_lines"
    return _split_ax_result(ax, explanation, extraction, raw)


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


def normalize_onyx_chat_completion_payload(payload: dict[str, Any]) -> None:
    """Onyx rejects ``temperature: 0``; map non-positive temperature to greedy decoding in-place.

    If ``temperature`` is present and ``<= 0`` (after ``float()``), drops ``temperature`` and
    ``top_p``, and sets ``do_sample`` to ``False``. Positive temperatures are unchanged.
    """
    if "temperature" not in payload:
        return
    try:
        t = float(payload["temperature"])
    except (TypeError, ValueError):
        return
    if t > 0:
        return
    payload.pop("temperature", None)
    payload["do_sample"] = False
    payload.pop("top_p", None)


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

    def _chat(self, system: str, user: str, *, completion_overrides: Optional[dict[str, Any]] = None) -> str:
        post = self._resolve_post()

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if completion_overrides:
            for k, v in completion_overrides.items():
                if k != "messages" and k != "model":
                    payload[k] = v
            normalize_onyx_chat_completion_payload(payload)
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
        ctx = dict(request.context) if isinstance(request.context, Mapping) else {}
        co = ctx.pop(COMPLETION_OVERRIDES_CONTEXT_KEY, None)
        overrides = co if isinstance(co, dict) else None
        raw = self._chat(
            SYSTEM_DRAFT,
            user_prompt_draft(request.goal, ctx),
            completion_overrides=overrides,
        )
        split = split_ax_and_prose(raw)
        return ExpertDraftResponse(
            ax_source=split.ax_source,
            backend_name=BACKEND_NAME,
            explanation=split.prose,
            metadata=self._metadata(raw, split),
        )

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        ctx = dict(request.context) if isinstance(request.context, Mapping) else {}
        co = ctx.pop(COMPLETION_OVERRIDES_CONTEXT_KEY, None)
        overrides = co if isinstance(co, dict) else None
        raw = self._chat(
            SYSTEM_REPAIR,
            user_prompt_repair(request.goal, request.current_program, request.error_report, ctx),
            completion_overrides=overrides,
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
    "COMPLETION_OVERRIDES_CONTEXT_KEY",
    "DRAFT_FEWSHOT",
    "EXAMPLES_SEMANTICS_BLOCK",
    "EXACT_SYMBOLIC_MATH_BLOCK",
    "REPAIR_NEURAL_TO_SYMBOLIC_BLOCK",
    "REPAIR_UNROLL_COLLAPSE_BLOCK",
    "ax_source_metadata_flags",
    "OnyxQwenBackend",
    "OnyxQwenError",
    "OnyxQwenHTTPError",
    "OnyxQwenParseError",
    "OnyxQwenTimeoutError",
    "OnyxQwenTransportError",
    "normalize_onyx_chat_completion_payload",
    "REPAIR_FEWSHOT",
    "SYSTEM_DRAFT",
    "SYSTEM_REPAIR",
    "SYNTAX_SUMMARY",
    "user_prompt_draft",
    "user_prompt_repair",
    "user_prompt_trace_summary",
    "split_ax_and_prose",
]
