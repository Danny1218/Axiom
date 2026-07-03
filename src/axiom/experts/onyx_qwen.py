"""HTTP expert: OpenAI-style chat completions against an Onyx / Qwen-compatible server.

Install: ``pip install -e ".[copilot]"`` (pulls ``requests``). Not imported from ``axiom`` root.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional
from urllib.parse import urljoin, urlparse

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
LMSTUDIO_DEFAULT_URL = "http://127.0.0.1:1234/v1/"
LMSTUDIO_DEFAULT_MODEL = "qwen/qwen3-8b"

_THINKING_BLOCK_RE = re.compile(
    r"<think>[\s\S]*?</think>",
    re.IGNORECASE,
)

# Merged into OpenAI-style chat ``payload`` for ``draft_program`` only; stripped from user context JSON.
COMPLETION_OVERRIDES_CONTEXT_KEY = "_onyx_completion_overrides"
REQUEST_CAPTURE_DIR_CONTEXT_KEY = "_onyx_request_capture_dir"
REQUEST_CAPTURE_ENABLED_CONTEXT_KEY = "_onyx_request_capture"
REQUEST_CAPTURE_DIR_ENV_VAR = "AXIOM_ONYX_REQUEST_CAPTURE_DIR"
DEFAULT_REQUEST_CAPTURE_DIR = "debug_onyx_request_capture"

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

DRAFT_ALWAYS_ON_SYNTAX_CORE = """Always-on syntax core (valid `.ax` examples - copy this style exactly):
```ax
y = x * 2.0;
if (x > 0.0) { y = x; } else { y = 0.0; }
if (x < 0.0) {
    y = 0.0;
} else {
    if (x < 1.0) {
        y = x;
    } else {
        y = 1.0;
    }
}
score = max(0.0, min(a + b, 1.0));
```
Return only `.ax` source, no prose."""

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

_ROBUSTNESS_AMBIGUITY_FALLBACK_TASK_IDS = frozenset(
    {
        "noisy_affine_thermometer",
        "sparse_quadratic_story",
        "sparse_three_way_peak",
        "adversarial_clean_reply_clip",
        "near_abs_with_bias",
        "weighted_floor_then_ramp",
        "signed_cross_term_noisy",
        "soft_cap_prefer_signal",
    }
)

_ROBUSTNESS_AMBIGUITY_FALLBACK_HINT = re.compile(
    r"(noise|noisy|underdetermined|adversarial|near[- ]miss|fallback|fall[- ]back|"
    r"should[- ]fall[- ]back|fallback[- ]only|expert[_ ]backend|robust(?:ness)?|ambigu(?:ity|ous))",
    re.I,
)

ROBUSTNESS_AMBIGUITY_FALLBACK_BLOCK = """Robustness / ambiguity fallback mode:
- Prefer the simplest symbolic program consistent with the goal.
- Do not overfit minor row noise with extra branches, thresholds, or constants.
- For underdetermined examples, choose the simplest formula implied by the goal.
- Never use `neural(...)`.
- Never use `clip(...)`.
- Never use comments or prose lines.
- Never use `else if` or `elseif`.
- Never use shorthand operators such as `*=`, `+=`, or `-=`.
- Never use inline if-expression forms such as `a if cond else b` or `cond ? a : b`.
- Never emit empty branches.
- Use canonical nested `if` blocks only.
- Use nested `max` / `min` only; never 3-arg `max` or `min`."""

ROBUSTNESS_AMBIGUITY_FALLBACK_EXAMPLES_BLOCK = """Positive anchors for robustness / ambiguity fallback:
- Affine + bias
```ax
adjusted = 1.25 * thermometer_reading - 0.2;
```

- Signed cross-term + bias
```ax
response = exposure * hedge - 0.5 * hedge + 0.25;
```

- Preference / cap with nested `max` / `min`
```ax
decision = min(max(primary, backup + 0.2), cap);
```

- Shifted piecewise with clean nested `if`
```ax
if (offset < -1.0) {
    band = -0.5 * offset;
} else {
    if (offset < 2.0) {
        band = offset + 1.5;
    } else {
        band = 4.0;
    }
}
```"""

ROBUSTNESS_AMBIGUITY_REPAIR_CLEANUP_BLOCK = """Fallback cleanup rules (repair these into canonical `.ax`):
- `else if (...) { ... }` or `elseif (...) { ... }` -> `else { if (...) { ... } else { ... } }`
- `clip(expr, low, high)` -> `max(low, min(expr, high))`
- `max(a, b, c)` -> `max(max(a, b), c)` and the same idea for `min(a, b, c)`
- `x *= y;`, `x += y;`, `x -= y;` -> explicit assignments such as `x = x * y;`
- Remove comments and prose entirely; return only code.
- Replace inline conditionals such as `a if cond else b` or `cond ? a : b` with full `if` / `else` blocks.
- Never leave an empty branch; every emitted branch must contain a real assignment."""

ROBUSTNESS_AMBIGUITY_SYNTAX_REPAIR_ONLY_BLOCK = """Fallback-only syntax repair mode:
- Preserve the intended math and variable names.
- Rewrite only into valid canonical `.ax`; do not add new behavior.
- Never add `neural(...)`.
- Never add comments or prose.
- Never use `else if` or `elseif`.
- Never use 3-arg `max` or `min`; use nested binary `max` / `min` only."""

SYSTEM_DRAFT = (
    "You write programs in THIS repository's `.ax` DSL, not Macaulay2, not Axiom CAS, and not generic pseudocode. "
    "Use `=` assignments with semicolons, direct variable names, `if (...) { ... } else { ... }`, and `while (...) { ... }`. "
    "Never use `:=`, `else if`, `&&`, `||`, `>=`, `<=`, chained comparisons, dotted access like `input.a`, malformed numerics like `0.0 0`, or prose. "
    "Use only `>`, `<`, `==`, `!=` for comparisons. "
    "Rewrite `else if` as nested `else { if (...) { ... } else { ... } }`. "
    "If the user prompt includes a canonical symbolic-family anchor, follow it exactly when relevant.\n"
    + DRAFT_ALWAYS_ON_SYNTAX_CORE
    + "\n"
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

SYSTEM_DRAFT_BENCHMARK_COMPACT = (
    "Write only this repository's `.ax` DSL. Use `=` assignments with semicolons, direct variable names, "
    "`if (...) { ... } else { ... }`, and `while (...) { ... }`. Use only `>`, `<`, `==`, `!=` for comparisons. "
    "Never emit `:=`, `print`, dotted access, `else if`, `elseif`, `otherwise`, `&&`, `||`, `>=`, `<=`, "
    "chained comparisons, inline conditionals, comments/prose, `clip(...)`, flat 3-arg `max`/`min`, or malformed numerics. "
    "Use nested `else { if (...) { ... } else { ... } }` when piecewise logic is needed.\n"
    + RETURN_VALID_AX_SEMICOLON_LINE
)

SYSTEM_REPAIR_BENCHMARK_COMPACT = (
    "Repair only into this repository's `.ax` DSL. Return ONLY the corrected full `.ax` program. "
    "Use `=` assignments with semicolons, direct variable names, and canonical nested `if` / `else` blocks. "
    "Use only `>`, `<`, `==`, `!=` for comparisons. Never emit `:=`, `print`, dotted access, `else if`, `elseif`, "
    "`otherwise`, `&&`, `||`, `>=`, `<=`, chained comparisons, inline conditionals, comments/prose, `clip(...)`, "
    "flat 3-arg `max`/`min`, or malformed numerics.\n"
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

BENCHMARK_COMPACT_SYNTAX_BLOCK = """Benchmark compact syntax guardrails:
- Return only valid `.ax` source with semicolon-terminated assignments.
- Use direct variable names only; no dotted access or prose/comments.
- Control flow only as `if (...) { ... } else { ... }` / `while (...) { ... }`.
- Comparisons only: `>`, `<`, `==`, `!=`.
- Never emit `:=`, `print`, `else if`, `elseif`, `otherwise`, `&&`, `||`, `>=`, `<=`, chained comparisons, inline conditionals, `clip(...)`, flat 3-arg `max`/`min`, malformed numerics, or missing semicolons.
- Use nested `else { if (...) { ... } else { ... } }` when piecewise logic is needed."""

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
_CLIP_CALL_PATTERN = re.compile(r"\bclip\s*\(", re.IGNORECASE)
_LOGICAL_OPERATOR_PATTERN = re.compile(r"&&|\|\|")
_UNSUPPORTED_BRANCH_SURFACE_PATTERN = re.compile(r"\belse\s+if\b|\belseif\b|\botherwise\b", re.IGNORECASE)
_SHORTHAND_ASSIGN_PATTERN = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*)([+\-*/])=(\s*)(.+;)\s*$")
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


def _has_inline_if_expression(source: str) -> bool:
    for line in source.splitlines():
        s = line.strip()
        if not s:
            continue
        if "?" in s and ":" in s:
            return True
        if " if " in f" {s} " and " else " in f" {s} " and not s.startswith("if ") and not s.startswith("if("):
            return True
    return False


def _unsupported_pattern_warnings(source: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if _UNSUPPORTED_BRANCH_SURFACE_PATTERN.search(source):
        out["unsupported_branch_surface_warning"] = True
    if _has_inline_if_expression(source):
        out["inline_if_expression_warning"] = True
    if _CLIP_CALL_PATTERN.search(source):
        out["clip_call_warning"] = True
    if _LOGICAL_OPERATOR_PATTERN.search(source):
        out["logical_operator_warning"] = True
    return out


def _pattern_warnings(ax: str) -> dict[str, Any]:
    """Metadata flags for likely per-row unrolling or invalid ``output(...)`` (source kept intact)."""
    out: dict[str, Any] = dict(_unsupported_pattern_warnings(ax))
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


def _context_flag_true(context: Mapping[str, Any], key: str) -> bool:
    value = context.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _robustness_ambiguity_fallback_text(goal: str, context: Mapping[str, Any]) -> str:
    parts = [(goal or "").strip()]
    for key in ("domain_context", "category", "backend_expected"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return " ".join(p for p in parts if p)


def _is_robustness_ambiguity_fallback_context(goal: str, context: Mapping[str, Any]) -> bool:
    if _context_flag_true(context, "fallback_expected"):
        return True
    for key in ("benchmark_task_id", "task_id"):
        value = context.get(key)
        if isinstance(value, str) and value.strip().lower() in _ROBUSTNESS_AMBIGUITY_FALLBACK_TASK_IDS:
            return True
    return bool(_ROBUSTNESS_AMBIGUITY_FALLBACK_HINT.search(_robustness_ambiguity_fallback_text(goal, context)))


def _append_robustness_ambiguity_fallback_blocks_if_needed(
    parts: list[str], goal: str, context: Mapping[str, Any], *, include_repair_cleanup: bool
) -> None:
    if not _is_robustness_ambiguity_fallback_context(goal, context):
        return
    parts.append(ROBUSTNESS_AMBIGUITY_FALLBACK_BLOCK + "\n\n")
    parts.append(ROBUSTNESS_AMBIGUITY_FALLBACK_EXAMPLES_BLOCK + "\n\n")
    if include_repair_cleanup:
        parts.append(ROBUSTNESS_AMBIGUITY_SYNTAX_REPAIR_ONLY_BLOCK + "\n\n")
        parts.append(ROBUSTNESS_AMBIGUITY_REPAIR_CLEANUP_BLOCK + "\n\n")


def _append_exact_symbolic_math_if_needed(parts: list[str], goal: str, context: Mapping[str, Any]) -> None:
    if _is_robustness_ambiguity_fallback_context(goal, context):
        return
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
    if _is_robustness_ambiguity_fallback_context(goal, context):
        return
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


def _benchmark_task_id(context: Mapping[str, Any]) -> Optional[str]:
    value = context.get("benchmark_task_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _use_compact_benchmark_prompt(context: Mapping[str, Any]) -> bool:
    return _benchmark_task_id(context) is not None


def _append_examples_semantics_if_needed(parts: list[str], context: Mapping[str, Any]) -> None:
    if _context_has_examples_driven_semantics(context):
        parts.append(EXAMPLES_SEMANTICS_BLOCK)


def _context_json(context: Mapping[str, Any]) -> str:
    return json.dumps(dict(context), sort_keys=True, separators=(",", ":"), default=str)


def _compact_benchmark_context_text(context: Mapping[str, Any]) -> str:
    parts: list[str] = []
    bench_id = _benchmark_task_id(context)
    if bench_id:
        parts.append(f"Benchmark task id: {bench_id}")
    domain_context = context.get("domain_context")
    if isinstance(domain_context, str) and domain_context.strip():
        parts.append(f"Domain context: {domain_context.strip()}")
    example_rows = context.get("example_input_rows")
    if isinstance(example_rows, list) and example_rows:
        parts.append(
            "Example inputs (JSON):\n"
            + json.dumps(example_rows, sort_keys=True, separators=(",", ":"), default=str)
        )
    expected_outputs = context.get("expected_outputs")
    if isinstance(expected_outputs, list) and expected_outputs:
        parts.append(
            "Expected outputs (JSON):\n"
            + json.dumps(expected_outputs, sort_keys=True, separators=(",", ":"), default=str)
        )
    return "\n\n".join(parts)


def _draft_prompt_impl(goal: str, context: Mapping[str, Any], *, compact_benchmark_prompt: bool) -> str:
    parts = [f"Goal:\n{goal}\n\n"]
    if compact_benchmark_prompt:
        compact_context = _compact_benchmark_context_text(context)
        if compact_context:
            parts.append(compact_context + "\n\n")
    else:
        parts.append(f"Context (JSON, sorted keys):\n{_context_json(context)}\n\n")
    _append_examples_semantics_if_needed(parts, context)
    _append_robustness_ambiguity_fallback_blocks_if_needed(parts, goal, context, include_repair_cleanup=False)
    _append_exact_symbolic_math_if_needed(parts, goal, context)
    _append_canonical_symbolic_family_drafts_if_needed(parts, goal, context)
    if compact_benchmark_prompt:
        parts.extend(
            [
                f"{BENCHMARK_COMPACT_SYNTAX_BLOCK}\n\n",
                "Respond with the complete `.ax` program only.\n" + RETURN_VALID_AX_SEMICOLON_LINE,
            ]
        )
    else:
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


def _repair_prompt_impl(
    goal: str,
    current_program: str,
    error_report: str,
    context: Mapping[str, Any],
    *,
    compact_benchmark_prompt: bool,
) -> str:
    parts = [f"Goal:\n{goal}\n\n", f"Error report:\n{error_report}\n\n"]
    if compact_benchmark_prompt:
        compact_context = _compact_benchmark_context_text(context)
        if compact_context:
            parts.append(compact_context + "\n\n")
    else:
        parts.append(f"Context (JSON, sorted keys):\n{_context_json(context)}\n\n")
    _append_examples_semantics_if_needed(parts, context)
    _append_robustness_ambiguity_fallback_blocks_if_needed(parts, goal, context, include_repair_cleanup=True)
    _append_exact_symbolic_math_if_needed(parts, goal, context)
    if compact_benchmark_prompt:
        parts.extend(
            [
                f"{BENCHMARK_COMPACT_SYNTAX_BLOCK}\n\n",
                f"{REPAIR_PARSE_ERROR_RULE}\n\n",
            ]
        )
    else:
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


def user_prompt_draft(goal: str, context: Mapping[str, Any]) -> str:
    return _draft_prompt_impl(goal, context, compact_benchmark_prompt=_use_compact_benchmark_prompt(context))


def user_prompt_repair(goal: str, current_program: str, error_report: str, context: Mapping[str, Any]) -> str:
    return _repair_prompt_impl(
        goal,
        current_program,
        error_report,
        context,
        compact_benchmark_prompt=_use_compact_benchmark_prompt(context),
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
    """Strip a lone first-line language tag (``ax`` / ``js`` / …) when the remainder is code-like."""
    lines = ax.splitlines()
    if lines and _is_stray_lang_tag_line(lines[0]) and len(lines) > 1 and _rest_looks_code_like(lines[1:]):
        extraction["stripped_language_tag"] = lines[0].strip().lower()
        lines = lines[1:]
        ax = "\n".join(lines).strip()
    return ax


def _line_looks_like_ax_code(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _line_code_score(s) > 0:
        return True
    if "{" in s or "}" in s:
        return True
    if s.startswith("else"):
        return True
    return False


def _split_ax_result(ax: str, explanation: Optional[str], extraction: dict[str, Any], raw: str) -> AxSplitResult:
    from axiom.compiler.normalizer import normalize_ax_source

    extraction.update(_forbidden_flags(ax, raw))
    ax = _finalize_extracted_source(ax, extraction)
    ax, norm_meta = normalize_ax_source(ax)
    extraction.update(norm_meta)
    extraction.update(_pattern_warnings(ax))
    extraction["code_line_count"] = len([ln for ln in ax.splitlines() if ln.strip()])
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
    text, stripped_think = strip_thinking_blocks(raw.strip())
    extraction: dict[str, Any] = {"extraction_mode": "plain_fallback"}
    if stripped_think:
        extraction["stripped_think_block"] = True

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

    def __init__(self, message: str, *, metadata: Optional[Mapping[str, Any]] = None) -> None:
        self.metadata = dict(metadata or {})
        super().__init__(message)


class OnyxQwenTransportError(OnyxQwenError):
    """Network / connection failure before a response."""

    def __init__(self, message: str, *, metadata: Optional[Mapping[str, Any]] = None) -> None:
        self.metadata = dict(metadata or {})
        super().__init__(message)


class OnyxQwenHTTPError(OnyxQwenError):
    def __init__(self, status_code: int, body_snippet: str, *, metadata: Optional[Mapping[str, Any]] = None) -> None:
        self.status_code = status_code
        self.body_snippet = body_snippet
        self.metadata = dict(metadata or {})
        summary_parts: list[str] = []
        for key in (
            "benchmark_task_id",
            "prompt_char_count",
            "system_prompt_char_count",
            "user_prompt_char_count",
            "compact_benchmark_prompt_used",
        ):
            if key in self.metadata:
                summary_parts.append(f"{key}={self.metadata[key]}")
        if "completion_overrides_applied" in self.metadata:
            summary_parts.append(f"completion_overrides_applied={self.metadata['completion_overrides_applied']}")
        summary = f" [{' ; '.join(summary_parts)}]" if summary_parts else ""
        super().__init__(f"HTTP {status_code}{summary}: {body_snippet[:200]}")


class OnyxQwenParseError(OnyxQwenError):
    """Invalid JSON or unexpected response shape."""


PostFn = Callable[..., Any]


def strip_thinking_blocks(raw: str) -> tuple[str, bool]:
    """Remove Qwen3 ``...`` spans before code extraction."""
    if not raw or not _THINKING_BLOCK_RE.search(raw):
        return raw, False
    stripped = _THINKING_BLOCK_RE.sub("", raw).strip()
    return stripped, True


def normalize_onyx_chat_completion_payload(payload: dict[str, Any]) -> None:
    """Onyx rejects ``temperature: 0``; map non-positive temperature to greedy decoding in-place.

    If ``temperature`` is present and ``<= 0`` (after ``float()``), drops ``temperature`` and
    ``top_p``, and sets ``do_sample`` to ``False``. Positive temperatures are unchanged.
    Also disables Qwen3 thinking mode when the API supports ``enable_thinking``.
    """
    payload.setdefault("enable_thinking", False)
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
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise OnyxQwenParseError("response missing choices[0].message") from e
    if not isinstance(msg, dict):
        raise OnyxQwenParseError("response choices[0].message is not an object")
    for key in ("content", "reasoning_content", "text"):
        value = msg.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise OnyxQwenParseError("response missing choices[0].message.content")


def _completion_overrides_applied(overrides: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": "m", "messages": []}
    if overrides:
        for k, v in overrides.items():
            if k != "messages" and k != "model":
                payload[k] = v
        normalize_onyx_chat_completion_payload(payload)
    return {k: payload[k] for k in sorted(payload) if k not in {"model", "messages", "enable_thinking"}}


def _request_diagnostics(
    context: Mapping[str, Any],
    *,
    system_prompt: str,
    user_prompt: str,
    completion_overrides: Optional[Mapping[str, Any]],
    compact_benchmark_prompt_used: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prompt_char_count": len(system_prompt) + len(user_prompt),
        "system_prompt_char_count": len(system_prompt),
        "user_prompt_char_count": len(user_prompt),
        "completion_overrides_applied": _completion_overrides_applied(completion_overrides),
        "compact_benchmark_prompt_used": bool(compact_benchmark_prompt_used),
        "http_failure_detail": None,
    }
    bench_id = _benchmark_task_id(context)
    if bench_id is not None:
        out["benchmark_task_id"] = bench_id
    return out


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _request_timing_metadata(
    *,
    request_started_at: str,
    elapsed_seconds: float,
    timeout_seconds: float,
    max_tokens: Optional[Any] = None,
    response_received_at: Optional[str] = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "request_started_at": request_started_at,
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "timeout_seconds": float(timeout_seconds),
    }
    if response_received_at is not None:
        out["response_received_at"] = response_received_at
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    return out


def _request_capture_dir_from_context(context: dict[str, Any]) -> Optional[Path]:
    capture_dir = context.pop(REQUEST_CAPTURE_DIR_CONTEXT_KEY, None)
    capture_enabled = bool(context.pop(REQUEST_CAPTURE_ENABLED_CONTEXT_KEY, False))
    if isinstance(capture_dir, str) and capture_dir.strip():
        return Path(capture_dir.strip())
    env_dir = os.environ.get(REQUEST_CAPTURE_DIR_ENV_VAR, "").strip()
    if env_dir:
        return Path(env_dir)
    if capture_enabled:
        return Path(DEFAULT_REQUEST_CAPTURE_DIR)
    return None


def _stable_json_text(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json_text(dict(payload)).encode("utf-8")).hexdigest()


def _diagnostic_response_headers(response: Any) -> dict[str, str]:
    raw = getattr(response, "headers", None)
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        lname = name.lower()
        if lname in {"content-type", "date", "server", "via"} or "request-id" in lname or "trace" in lname:
            out[name] = str(value)
    return out


def _request_id_from_response_headers(headers: Mapping[str, str]) -> Optional[str]:
    for key, value in headers.items():
        lname = str(key).strip().lower()
        if "request-id" not in lname:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _response_id_from_body(data: Any) -> Optional[str]:
    if not isinstance(data, Mapping):
        return None
    response_id = data.get("id")
    text = str(response_id).strip() if response_id is not None else ""
    return text or None


def _request_capture_filename(
    request_kind: str, benchmark_task_id: Optional[str], payload_sha256: str, status_code: Optional[int]
) -> str:
    stem = benchmark_task_id or "request"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "request"
    suffix = f"_http{status_code}" if status_code is not None else ""
    return f"{request_kind}_{stem}_{payload_sha256[:12]}{suffix}.json"


def _write_request_capture(
    capture_dir: Path,
    *,
    request_kind: str,
    benchmark_task_id: Optional[str],
    chat_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    prompt_char_count: int,
    system_prompt_char_count: int,
    user_prompt_char_count: int,
    completion_overrides_applied: Mapping[str, Any],
    compact_benchmark_prompt_used: bool,
    payload: Mapping[str, Any],
    payload_sha256: str,
    status_code: Optional[int] = None,
    http_failure_detail: Optional[str] = None,
    failure_kind: Optional[str] = None,
    exception_class: Optional[str] = None,
    exception_message: Optional[str] = None,
    response_headers: Optional[Mapping[str, str]] = None,
    request_id: Optional[str] = None,
    response_id: Optional[str] = None,
    timing_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    capture_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "kind": "axiom.onyx_qwen.request_capture",
        "request_kind": request_kind,
        "benchmark_task_id": benchmark_task_id,
        "model": model,
        "chat_path": urlparse(chat_url).path,
        "chat_url": chat_url,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_char_count": prompt_char_count,
        "system_prompt_char_count": system_prompt_char_count,
        "user_prompt_char_count": user_prompt_char_count,
        "completion_overrides_applied": dict(completion_overrides_applied),
        "compact_benchmark_prompt_used": bool(compact_benchmark_prompt_used),
        "payload_sha256": payload_sha256,
        "payload": dict(payload),
    }
    if status_code is not None:
        out["status_code"] = int(status_code)
    if http_failure_detail is not None:
        out["http_failure_detail"] = http_failure_detail
    if failure_kind is not None:
        out["failure_kind"] = failure_kind
    if exception_class is not None:
        out["exception_class"] = exception_class
    if exception_message is not None:
        out["exception_message"] = exception_message
    if response_headers:
        out["response_headers"] = {str(k): str(v) for k, v in response_headers.items()}
    if request_id is not None:
        out["request_id"] = str(request_id)
    if response_id is not None:
        out["response_id"] = str(response_id)
    if timing_metadata:
        out.update({str(k): v for k, v in timing_metadata.items() if v is not None})
    from axiom.copilot.redaction import capture_mode_from_env, redact_mapping

    mode = capture_mode_from_env()
    if mode != "full":
        out = redact_mapping(out, redact_prompts=True)
        out["capture_mode"] = "redacted"
    else:
        out["capture_mode"] = "full"
    path = capture_dir / _request_capture_filename(request_kind, benchmark_task_id, payload_sha256, status_code)
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _capture_exception_artifact(
    *,
    capture_dir: Optional[Path],
    request_kind: str,
    request_diagnostics: Optional[dict[str, Any]],
    chat_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    payload: Mapping[str, Any],
    payload_sha256: str,
    failure_kind: str,
    exc: Exception,
    timing_metadata: Mapping[str, Any],
) -> Optional[str]:
    if capture_dir is None:
        return None
    meta = dict(request_diagnostics or {})
    capture_path = _write_request_capture(
        capture_dir,
        request_kind=request_kind,
        benchmark_task_id=meta.get("benchmark_task_id"),
        chat_url=chat_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_char_count=int(meta.get("prompt_char_count", len(system_prompt) + len(user_prompt))),
        system_prompt_char_count=int(meta.get("system_prompt_char_count", len(system_prompt))),
        user_prompt_char_count=int(meta.get("user_prompt_char_count", len(user_prompt))),
        completion_overrides_applied=meta.get("completion_overrides_applied") or {},
        compact_benchmark_prompt_used=bool(meta.get("compact_benchmark_prompt_used")),
        payload=payload,
        payload_sha256=payload_sha256,
        failure_kind=failure_kind,
        exception_class=type(exc).__name__,
        exception_message=str(exc),
        timing_metadata=timing_metadata,
    )
    if request_diagnostics is not None:
        request_diagnostics["request_capture_path"] = str(capture_path)
    return str(capture_path)


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
        base = base_url.rstrip("/") + "/"
        path = chat_path.lstrip("/")
        if path.startswith("v1/") and (base.endswith("/v1/") or base.endswith("/v1")):
            path = path[3:]
        self._base = base
        self._chat_url = urljoin(self._base, path)
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

    def _chat(
        self,
        system: str,
        user: str,
        *,
        completion_overrides: Optional[dict[str, Any]] = None,
        request_diagnostics: Optional[dict[str, Any]] = None,
        request_kind: str = "draft",
        capture_dir: Optional[Path] = None,
    ) -> str:
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
        else:
            normalize_onyx_chat_completion_payload(payload)
        user_content = str(payload["messages"][-1]["content"])
        if "qwen" in self._model.lower() and "/no_think" not in user_content.lower():
            payload["messages"][-1]["content"] = user_content.rstrip() + " /no_think"
        request_started_at = _utc_now_iso()
        started_perf = time.perf_counter()
        max_tokens = payload.get("max_tokens")
        payload_sha = _payload_sha256(payload)
        if request_diagnostics is not None:
            request_diagnostics["payload_sha256"] = payload_sha
        try:
            r = post(
                self._chat_url,
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except Exception as e:
            timing_metadata = _request_timing_metadata(
                request_started_at=request_started_at,
                elapsed_seconds=time.perf_counter() - started_perf,
                timeout_seconds=self._timeout,
                max_tokens=max_tokens,
            )
            if requests is not None and isinstance(e, requests.exceptions.Timeout):
                meta = dict(request_diagnostics or {})
                meta.update(timing_metadata)
                meta["failure_kind"] = "timeout"
                meta["exception_class"] = type(e).__name__
                meta["exception_message"] = str(e)
                capture_path = _capture_exception_artifact(
                    capture_dir=capture_dir,
                    request_kind=request_kind,
                    request_diagnostics=request_diagnostics,
                    chat_url=self._chat_url,
                    model=self._model,
                    system_prompt=system,
                    user_prompt=user,
                    payload=payload,
                    payload_sha256=payload_sha,
                    failure_kind="timeout",
                    exc=e,
                    timing_metadata=timing_metadata,
                )
                if capture_path is not None:
                    meta["request_capture_path"] = capture_path
                raise OnyxQwenTimeoutError(str(e), metadata=meta) from e
            if requests is not None and isinstance(e, requests.exceptions.RequestException):
                meta = dict(request_diagnostics or {})
                meta.update(timing_metadata)
                meta["failure_kind"] = "transport"
                meta["exception_class"] = type(e).__name__
                meta["exception_message"] = str(e)
                capture_path = _capture_exception_artifact(
                    capture_dir=capture_dir,
                    request_kind=request_kind,
                    request_diagnostics=request_diagnostics,
                    chat_url=self._chat_url,
                    model=self._model,
                    system_prompt=system,
                    user_prompt=user,
                    payload=payload,
                    payload_sha256=payload_sha,
                    failure_kind="transport",
                    exc=e,
                    timing_metadata=timing_metadata,
                )
                if capture_path is not None:
                    meta["request_capture_path"] = capture_path
                raise OnyxQwenTransportError(str(e), metadata=meta) from e
            raise

        response_received_at = _utc_now_iso()
        timing_metadata = _request_timing_metadata(
            request_started_at=request_started_at,
            response_received_at=response_received_at,
            elapsed_seconds=time.perf_counter() - started_perf,
            timeout_seconds=self._timeout,
            max_tokens=max_tokens,
        )
        if r.status_code >= 400:
            snippet = r.text if isinstance(r.text, str) else ""
            meta = dict(request_diagnostics or {})
            meta.update(timing_metadata)
            meta["http_failure_detail"] = snippet[:2000]
            meta["status_code"] = int(r.status_code)
            response_headers = _diagnostic_response_headers(r)
            request_id = _request_id_from_response_headers(response_headers)
            if request_id:
                meta["request_id"] = request_id
            if capture_dir is not None:
                capture_path = _write_request_capture(
                    capture_dir,
                    request_kind=request_kind,
                    benchmark_task_id=meta.get("benchmark_task_id"),
                    chat_url=self._chat_url,
                    model=self._model,
                    system_prompt=system,
                    user_prompt=user,
                    prompt_char_count=int(meta.get("prompt_char_count", len(system) + len(user))),
                    system_prompt_char_count=int(meta.get("system_prompt_char_count", len(system))),
                    user_prompt_char_count=int(meta.get("user_prompt_char_count", len(user))),
                    completion_overrides_applied=meta.get("completion_overrides_applied") or {},
                    compact_benchmark_prompt_used=bool(meta.get("compact_benchmark_prompt_used")),
                    payload=payload,
                    payload_sha256=payload_sha,
                    status_code=int(r.status_code),
                    http_failure_detail=snippet[:2000],
                    response_headers=response_headers,
                    request_id=request_id,
                    timing_metadata=timing_metadata,
                )
                meta["request_capture_path"] = str(capture_path)
                if request_diagnostics is not None:
                    request_diagnostics["request_capture_path"] = str(capture_path)
            raise OnyxQwenHTTPError(r.status_code, snippet[:2000], metadata=meta)

        try:
            data = r.json()
        except ValueError as e:
            raise OnyxQwenParseError("response body is not valid JSON") from e

        response_headers = _diagnostic_response_headers(r)
        request_id = _request_id_from_response_headers(response_headers)
        response_id = _response_id_from_body(data)
        if request_diagnostics is not None:
            request_diagnostics.update(timing_metadata)
            if request_id:
                request_diagnostics["request_id"] = request_id
            if response_id:
                request_diagnostics["response_id"] = response_id
            if capture_dir is not None:
                capture_path = _write_request_capture(
                    capture_dir,
                    request_kind=request_kind,
                    benchmark_task_id=request_diagnostics.get("benchmark_task_id"),
                    chat_url=self._chat_url,
                    model=self._model,
                    system_prompt=system,
                    user_prompt=user,
                    prompt_char_count=int(request_diagnostics.get("prompt_char_count", len(system) + len(user))),
                    system_prompt_char_count=int(request_diagnostics.get("system_prompt_char_count", len(system))),
                    user_prompt_char_count=int(request_diagnostics.get("user_prompt_char_count", len(user))),
                    completion_overrides_applied=request_diagnostics.get("completion_overrides_applied") or {},
                    compact_benchmark_prompt_used=bool(request_diagnostics.get("compact_benchmark_prompt_used")),
                    payload=payload,
                    payload_sha256=payload_sha,
                    response_headers=response_headers,
                    request_id=request_id or response_id,
                    response_id=response_id,
                    timing_metadata=timing_metadata,
                )
                request_diagnostics["request_capture_path"] = str(capture_path)

        return _assistant_content(data)

    def _metadata(self, raw: str, split: AxSplitResult, request_diagnostics: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        return {"model": self._model, "raw_chars": len(raw), **(request_diagnostics or {}), **split.extraction}

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        ctx = dict(request.context) if isinstance(request.context, Mapping) else {}
        co = ctx.pop(COMPLETION_OVERRIDES_CONTEXT_KEY, None)
        overrides = co if isinstance(co, dict) else None
        capture_dir = _request_capture_dir_from_context(ctx)
        compact_benchmark_prompt_used = _use_compact_benchmark_prompt(ctx)
        system_prompt = SYSTEM_DRAFT_BENCHMARK_COMPACT if compact_benchmark_prompt_used else SYSTEM_DRAFT
        user_prompt = _draft_prompt_impl(request.goal, ctx, compact_benchmark_prompt=compact_benchmark_prompt_used)
        request_diagnostics = _request_diagnostics(
            ctx,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            completion_overrides=overrides,
            compact_benchmark_prompt_used=compact_benchmark_prompt_used,
        )
        raw = self._chat(
            system_prompt,
            user_prompt,
            completion_overrides=overrides,
            request_diagnostics=request_diagnostics,
            request_kind="draft",
            capture_dir=capture_dir,
        )
        split = split_ax_and_prose(raw)
        return ExpertDraftResponse(
            ax_source=split.ax_source,
            backend_name=BACKEND_NAME,
            explanation=split.prose,
            metadata=self._metadata(raw, split, request_diagnostics),
        )

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        ctx = dict(request.context) if isinstance(request.context, Mapping) else {}
        co = ctx.pop(COMPLETION_OVERRIDES_CONTEXT_KEY, None)
        overrides = co if isinstance(co, dict) else None
        capture_dir = _request_capture_dir_from_context(ctx)
        compact_benchmark_prompt_used = _use_compact_benchmark_prompt(ctx)
        system_prompt = SYSTEM_REPAIR_BENCHMARK_COMPACT if compact_benchmark_prompt_used else SYSTEM_REPAIR
        user_prompt = _repair_prompt_impl(
            request.goal,
            request.current_program,
            request.error_report,
            ctx,
            compact_benchmark_prompt=compact_benchmark_prompt_used,
        )
        request_diagnostics = _request_diagnostics(
            ctx,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            completion_overrides=overrides,
            compact_benchmark_prompt_used=compact_benchmark_prompt_used,
        )
        raw = self._chat(
            system_prompt,
            user_prompt,
            completion_overrides=overrides,
            request_diagnostics=request_diagnostics,
            request_kind="repair",
            capture_dir=capture_dir,
        )
        split = split_ax_and_prose(raw)
        return ExpertDraftResponse(
            ax_source=split.ax_source,
            backend_name=BACKEND_NAME,
            explanation=split.prose,
            metadata=self._metadata(raw, split, request_diagnostics),
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
    "DEFAULT_REQUEST_CAPTURE_DIR",
    "DRAFT_FEWSHOT",
    "EXAMPLES_SEMANTICS_BLOCK",
    "EXACT_SYMBOLIC_MATH_BLOCK",
    "ROBUSTNESS_AMBIGUITY_FALLBACK_BLOCK",
    "ROBUSTNESS_AMBIGUITY_FALLBACK_EXAMPLES_BLOCK",
    "ROBUSTNESS_AMBIGUITY_REPAIR_CLEANUP_BLOCK",
    "REQUEST_CAPTURE_DIR_CONTEXT_KEY",
    "REQUEST_CAPTURE_DIR_ENV_VAR",
    "REQUEST_CAPTURE_ENABLED_CONTEXT_KEY",
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
    "strip_thinking_blocks",
    "split_ax_and_prose",
]
