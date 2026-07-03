"""Deterministic source-to-source rewrites for almost-valid ``.ax`` text."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

_TRAILING_DOT_FLOAT = re.compile(r"(?<!\.\d)(\d+)\.(?![0-9.])")
_STATEMENT_EQ_EQ_ASSIGN = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*)==(\s*)(.+;)\s*$")
_SHORTHAND_ASSIGN_PATTERN = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*)([+\-*/])=(\s*)(.+;)\s*$")
_ELSE_IF_PATTERN = re.compile(r"\}\s*else\s+if\s*\(", re.IGNORECASE)
_LEADING_ELSE_IF = re.compile(r"(^|\n)(\s*)else\s+if\s*\(", re.IGNORECASE)
_CLIP_PATTERN = re.compile(r"\bclip\s*\(", re.IGNORECASE)
_PY_TERNARY = re.compile(
    r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s+if\s+(.+?)\s+else\s+(.+?);\s*$"
)
_C_TERNARY = re.compile(
    r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*\?\s*(.+?)\s*:\s*(.+?);\s*$"
)


def _line_looks_like_ax_code(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if any(tok in s for tok in ("=", ";", "{", "}", "if ", "if(", "else", "while", "max(", "min(")):
        return True
    return s.startswith("else")


def strip_line_comments_and_trailing_prose(ax: str) -> Tuple[str, Dict[str, bool]]:
    meta: Dict[str, bool] = {}
    lines: List[str] = []
    stripped_comments = False
    for line in ax.splitlines():
        body, sep, _ = line.partition("//")
        if sep:
            line = body.rstrip()
            stripped_comments = True
        lines.append(line)
    if stripped_comments:
        meta["stripped_line_comments"] = True
    last_code_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if _line_looks_like_ax_code(line):
            last_code_idx = i
    if last_code_idx is not None:
        trailing = lines[last_code_idx + 1 :]
        if any(line.strip() for line in trailing):
            meta["stripped_trailing_prose"] = True
        lines = lines[: last_code_idx + 1]
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip(), meta


def rewrite_shorthand_assignments(ax: str) -> Tuple[str, Dict[str, bool]]:
    meta: Dict[str, bool] = {}
    out_lines: List[str] = []
    changed = False
    for line in ax.splitlines():
        m = _SHORTHAND_ASSIGN_PATTERN.match(line)
        if not m:
            out_lines.append(line)
            continue
        indent, lhs, _, op, _, rhs = m.groups()
        out_lines.append(f"{indent}{lhs} = {lhs} {op} {rhs.strip()}")
        changed = True
    if changed:
        meta["normalized_shorthand_assignment"] = True
    return "\n".join(out_lines).strip(), meta


def find_matching_paren(text: str, open_idx: int) -> Optional[int]:
    depth = 0
    in_string = False
    escaped = False
    for i in range(open_idx, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def find_matching_brace(text: str, open_idx: int) -> Optional[int]:
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def split_top_level_args(text: str) -> List[str]:
    parts: List[str] = []
    start = 0
    paren_depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "(":
            paren_depth += 1
            continue
        if ch == ")":
            paren_depth -= 1
            continue
        if ch == "," and paren_depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def rewrite_three_arg_extrema(ax: str) -> Tuple[str, Dict[str, bool]]:
    meta: Dict[str, bool] = {}
    out = ax
    for fn_name, meta_key in (("max", "normalized_three_arg_max"), ("min", "normalized_three_arg_min")):
        pattern = re.compile(rf"\b{fn_name}\s*\(")
        changed = False
        while True:
            match = pattern.search(out)
            if match is None:
                break
            open_idx = out.find("(", match.start(), match.end())
            close_idx = find_matching_paren(out, open_idx)
            if close_idx is None:
                break
            args = split_top_level_args(out[open_idx + 1 : close_idx])
            if len(args) != 3 or not all(args):
                break
            replacement = f"{fn_name}({fn_name}({args[0]}, {args[1]}), {args[2]})"
            out = out[: match.start()] + replacement + out[close_idx + 1 :]
            changed = True
        if changed:
            meta[meta_key] = True
    return out, meta


def rewrite_clip_calls(ax: str) -> Tuple[str, Dict[str, bool]]:
    meta: Dict[str, bool] = {}
    out = ax
    changed = False
    while True:
        match = _CLIP_PATTERN.search(out)
        if match is None:
            break
        open_idx = out.find("(", match.end() - 1)
        close_idx = find_matching_paren(out, open_idx)
        if close_idx is None:
            break
        args = split_top_level_args(out[open_idx + 1 : close_idx])
        if len(args) != 3 or not all(args):
            break
        expr, lo, hi = args
        replacement = f"max({lo}, min({expr}, {hi}))"
        out = out[: match.start()] + replacement + out[close_idx + 1 :]
        changed = True
    if changed:
        meta["normalized_clip_call"] = True
    return out, meta


def _split_top_level_logical(condition: str, op: str) -> Optional[List[str]]:
    parts: List[str] = []
    start = 0
    depth = 0
    i = 0
    while i < len(condition):
        ch = condition[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and condition.startswith(op, i):
            parts.append(condition[start:i].strip())
            start = i + len(op)
            i += len(op)
            continue
        i += 1
    parts.append(condition[start:].strip())
    return parts if len(parts) > 1 else None


def _nest_or_if(parts: List[str], body: str, indent: str) -> str:
    if len(parts) == 1:
        return f"{indent}if ({parts[0]}) {{\n{body}\n{indent}}}"
    head, *tail = parts
    inner = _nest_or_if(tail, body, indent + "  ")
    return f"{indent}if ({head}) {{\n{body}\n{indent}}} else {{\n{inner}\n{indent}}}"


def rewrite_logical_operators(ax: str) -> Tuple[str, Dict[str, bool]]:
    meta: Dict[str, bool] = {}
    out_lines: List[str] = []
    changed = False
    for line in ax.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("if (") or "&&" not in line and "||" not in line:
            out_lines.append(line)
            continue
        open_idx = line.find("(")
        close_idx = find_matching_paren(line, open_idx)
        if close_idx is None:
            out_lines.append(line)
            continue
        condition = line[open_idx + 1 : close_idx]
        body_start = line.find("{", close_idx)
        body_end = find_matching_brace(line, body_start) if body_start >= 0 else None
        if body_end is None:
            out_lines.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        body = line[body_start + 1 : body_end].strip()
        and_parts = _split_top_level_logical(condition, "&&")
        if and_parts:
            nested = body
            for part in reversed(and_parts):
                nested = f"{indent}if ({part}) {{\n{nested}\n{indent}}}"
            out_lines.append(nested.rstrip())
            changed = True
            meta["normalized_logical_and"] = True
            continue
        or_parts = _split_top_level_logical(condition, "||")
        if or_parts:
            nested = _nest_or_if(or_parts, body, indent)
            out_lines.append(nested)
            changed = True
            meta["normalized_logical_or"] = True
            continue
        out_lines.append(line)
    if changed:
        return "\n".join(out_lines).strip(), meta
    return ax, meta


def _close_else_if_wrappers(text: str) -> str:
    marker = "else { if ("
    idx = 0
    insert_at: List[int] = []
    while True:
        pos = text.find(marker, idx)
        if pos == -1:
            break
        open_paren = pos + len("else { if ")
        close_paren = find_matching_paren(text, open_paren)
        if close_paren is None:
            break
        body_open = text.find("{", close_paren)
        if body_open < 0:
            break
        block_end = find_matching_brace(text, body_open)
        if block_end is None:
            break
        tail = text[block_end + 1 :].lstrip()
        if tail.startswith("else"):
            else_body = text.find("{", block_end + 1)
            if else_body >= 0:
                block_end = find_matching_brace(text, else_body) or block_end
        insert_at.append(block_end + 1)
        idx = pos + len(marker)
    out = text
    for pos in reversed(insert_at):
        out = out[:pos] + "}" + out[pos:]
    return out


def rewrite_else_if(ax: str) -> Tuple[str, Dict[str, bool]]:
    meta: Dict[str, bool] = {}
    out = ax
    changed = False
    while _ELSE_IF_PATTERN.search(out):
        out = _ELSE_IF_PATTERN.sub("} else { if (", out, count=1)
        changed = True
    while _LEADING_ELSE_IF.search(out):
        out = _LEADING_ELSE_IF.sub(r"\1\2else { if (", out, count=1)
        changed = True
    if changed:
        out = _close_else_if_wrappers(out)
        meta["normalized_else_if"] = True
    return out, meta


def rewrite_inline_ternaries(ax: str) -> Tuple[str, Dict[str, bool]]:
    meta: Dict[str, bool] = {}
    out_lines: List[str] = []
    changed = False
    for line in ax.splitlines():
        m = _PY_TERNARY.match(line) or _C_TERNARY.match(line)
        if not m:
            out_lines.append(line)
            continue
        indent, target, when_true, cond, when_false = m.groups()
        out_lines.extend(
            [
                f"{indent}if ({cond.strip()}) {{",
                f"{indent}    {target} = {when_true.strip()};",
                f"{indent}}} else {{",
                f"{indent}    {target} = {when_false.strip()};",
                f"{indent}}}",
            ]
        )
        changed = True
    if changed:
        meta["normalized_inline_ternary"] = True
    return "\n".join(out_lines).strip(), meta


def normalize_conservative(ax: str) -> Tuple[str, Dict[str, bool]]:
    meta: Dict[str, bool] = {}
    out = ax
    if ":=" in out:
        out = out.replace(":=", "=")
        meta["normalized_colon_eq"] = True
    lines = out.splitlines(keepends=True)
    rewritten: List[str] = []
    changed_eqeq = False
    for line in lines:
        body = line.rstrip("\r\n")
        newline = line[len(body) :]
        m = _STATEMENT_EQ_EQ_ASSIGN.match(body)
        if m:
            line = f"{m.group(1)}{m.group(2)}{m.group(3)}={m.group(4)}{m.group(5)}{newline}"
            changed_eqeq = True
        rewritten.append(line)
    if changed_eqeq:
        out = "".join(rewritten)
        meta["normalized_statement_eq_eq_assignment"] = True
    out2, n = _TRAILING_DOT_FLOAT.subn(r"\1.0", out)
    if n:
        out = out2
        meta["normalized_trailing_dot_float"] = True
    return out, meta
