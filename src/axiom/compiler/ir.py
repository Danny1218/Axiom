from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from lark import Token, Tree


def _child_trees(t: Tree) -> list:
    return [c for c in t.children if isinstance(c, Tree)]

IRList = List[tuple]


_CMP = {"GT": "OP_CMP_GT", "LT": "OP_CMP_LT", "EQ": "OP_CMP_EQ", "NE": "OP_CMP_NE"}


def ast_to_ir(tree: Tree) -> IRList:
    assert tree.data == "start"
    out: IRList = []
    for child in tree.children:
        out.extend(_stmt(child))
    return out


def ir_to_digraph(ir: IRList) -> nx.DiGraph:
    """Linear opcode sequence as a chain of nodes (lightweight view; see `compiler.flow.wire_execution_graph` for module DAG)."""
    G = nx.DiGraph()
    for i, instr in enumerate(ir):
        op = instr[0] if isinstance(instr, tuple) else instr
        G.add_node(i, op=op)
        if i:
            G.add_edge(i - 1, i)
    return G


def _stmt(t: Tree) -> IRList:
    if t.data == "assign_stmt":
        name = str(t.children[0])
        return [("OP_ASSIGN", name, _expr(t.children[1]))]
    if t.data == "expr_stmt":
        return [("OP_EXPR_STMT", _expr(t.children[0]))]
    if t.data == "if_stmt":
        cond = _expr(t.children[0])
        then_ir = _inner(t.children[1])
        if len(t.children) > 2:
            eb = t.children[2]
            assert eb.data == "else_block"
            else_ir = _inner(eb.children[0])
        else:
            else_ir = []
        return [("OP_CONDITIONAL", cond, then_ir, else_ir)]
    if t.data == "while_stmt":
        trees = _child_trees(t)
        cond = _expr(trees[0])
        body_ir = _inner(trees[1])
        return [("OP_LOOP", cond, body_ir)]
    raise ValueError(f"unknown statement {t.data}")


def _inner(t: Tree) -> IRList:
    assert t.data == "inner"
    acc: IRList = []
    for c in t.children:
        acc.extend(_stmt(c))
    return acc


def _expr(t: Tree) -> List[tuple]:
    if t.data == "comparison":
        return _comparison(t)
    if t.data == "sum":
        return _sum(t)
    if t.data == "product":
        return _product(t)
    if t.data == "postfix_expr":
        return _postfix_expr(t)
    if t.data == "atom":
        return _atom(t)
    raise ValueError(f"unknown expr {t.data}")


def _comparison(t: Tree) -> List[tuple]:
    kids = t.children
    if len(kids) == 1:
        return _expr(kids[0])
    left, op_tree, right = kids[0], kids[1], kids[2]
    return _cmp_operand(left) + _cmp_operand(right) + [(_CMP[str(op_tree.children[0].type)],)]


def _cmp_operand(t: Tree) -> List[tuple]:
    if t.data == "atom":
        return _atom(t)
    if t.data == "postfix_expr":
        return _postfix_expr(t)
    return _expr(t)


def _sum(t: Tree) -> List[tuple]:
    terms = t.children[0::2]
    ops = t.children[1::2]
    acc = _mul_group(terms[0])
    for i, op in enumerate(ops):
        acc += _mul_group(terms[i + 1])
        acc += [("OP_ADD",) if op.type == "ADD" else ("OP_SUB",)]
    return acc


def _mul_group(t: Tree) -> List[tuple]:
    if t.data == "product":
        return _product(t)
    if t.data == "postfix_expr":
        return _postfix_expr(t)
    if t.data == "atom":
        return _atom(t)
    raise ValueError(f"expected product|postfix_expr|atom under sum, got {t.data}")


def _product(t: Tree) -> List[tuple]:
    terms = t.children[0::2]
    ops = t.children[1::2]
    acc = _postfix_expr(terms[0])
    for i, op in enumerate(ops):
        acc += _postfix_expr(terms[i + 1])
        acc += [("OP_MUL",) if op.type == "MUL" else ("OP_DIV",)]
    return acc


def _postfix_expr(t: Tree) -> List[tuple]:
    if t.data == "atom":
        return _atom(t)
    if t.data == "postfix_expr":
        ch = t.children
        if len(ch) == 1:
            c0 = ch[0]
            if not isinstance(c0, Tree):
                raise ValueError("postfix_expr expects Tree child")
            if c0.data == "atom":
                return _atom(c0)
            return _postfix_expr(c0)
        base, idx_tree = ch[0], ch[1]
        return _postfix_expr(base) + _expr(idx_tree) + [("OP_INDEX",)]
    raise ValueError(f"unknown postfix_expr {t.data}")


def _array_literal(t: Tree) -> List[tuple]:
    if not t.children:
        raise ValueError("empty array literal is not allowed")
    ir: List[tuple] = []
    for c in t.children:
        ir += _expr(c)
    ir.append(("OP_VEC_PACK", len(t.children)))
    return ir


def _atom(t: Tree) -> List[tuple]:
    ch = t.children
    if len(ch) == 1:
        x = ch[0]
        if isinstance(x, Token):
            if x.type == "NUMBER":
                return [("OP_CONST", _number(x))]
            if x.type == "NAME":
                return [("OP_LOAD", str(x))]
        if isinstance(x, Tree):
            if x.data == "atom":
                return _atom(x) + [("OP_NEG",)]
            if x.data == "array_literal":
                return _array_literal(x)
            if x.data == "comparison":
                return _comparison(x)
            if x.data == "expr":
                return _expr(x.children[0])
            if x.data in ("sum", "product", "comparison", "postfix_expr"):
                return _expr(x)
    raise ValueError(f"bad atom {ch!r}")


def _number(tok: Token) -> int | float:
    s = str(tok)
    if "." in s or "e" in s.lower():
        return float(s)
    return int(s)


Stmt = Tuple[Any, ...]
ExprIR = List[Tuple[Any, ...]]


def _infer_expr_output_width(expr: ExprIR, known: Dict[str, int]) -> int:
    """Infer trailing tensor width (1 = scalar/(B,), K = (B,K)) from stack IR + current name widths."""
    stack: List[int] = []
    for tup in expr:
        if not isinstance(tup, tuple) or not tup:
            continue
        op = tup[0]
        if op == "OP_CONST":
            stack.append(1)
        elif op == "OP_LOAD":
            stack.append(max(1, int(known.get(str(tup[1]), 1))))
        elif op == "OP_NEG":
            stack.append(stack.pop())
        elif op == "OP_VEC_PACK":
            n = int(tup[1])
            for _ in range(n):
                stack.pop()
            stack.append(n)
        elif op == "OP_INDEX":
            stack.pop()
            stack.pop()
            stack.append(1)
        elif op in ("OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV"):
            b, a = stack.pop(), stack.pop()
            stack.append(max(a, b))
        elif isinstance(op, str) and op.startswith("OP_CMP_"):
            stack.pop()
            stack.pop()
            stack.append(1)
        else:
            raise ValueError(f"unknown expr op for width inference: {op}")
    return stack[-1] if stack else 1


def _accum_widths_from_stmt(stmt: Stmt, widths: Dict[str, int]) -> None:
    if not isinstance(stmt, tuple) or not stmt:
        return
    op = stmt[0]
    if op == "OP_ASSIGN":
        nm = str(stmt[1])
        rhs_w = _infer_expr_output_width(list(stmt[2]), widths)
        widths[nm] = max(widths.get(nm, 1), rhs_w)
    elif op == "OP_CONDITIONAL":
        snap = dict(widths)
        wt = dict(widths)
        for s in stmt[2]:
            _accum_widths_from_stmt(tuple(s) if isinstance(s, list) else s, wt)
        we = dict(widths)
        for s in stmt[3]:
            _accum_widths_from_stmt(tuple(s) if isinstance(s, list) else s, we)
        for k in set(wt) | set(we):
            widths[k] = max(snap.get(k, 1), wt.get(k, 1), we.get(k, 1))
    elif op == "OP_LOOP":
        for _ in range(64):
            before = dict(widths)
            for s in stmt[2]:
                _accum_widths_from_stmt(tuple(s) if isinstance(s, list) else s, widths)
            if widths == before:
                break


def extract_abi_layout(
    ir: IRList, *, max_vars: Optional[int] = None
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Map each variable name to a **start** trunk column and optional width (default 1).

    Vector assignments (``OP_VEC_PACK``) reserve ``width`` consecutive columns for that name.
    """
    widths: Dict[str, int] = {}
    for instr in ir:
        _accum_widths_from_stmt(tuple(instr) if isinstance(instr, list) else instr, widths)

    order: List[str] = []
    seen: Set[str] = set()

    def add_name(name: str) -> bool:
        n = str(name)
        if n in seen:
            return True
        seen.add(n)
        order.append(n)
        return True

    def walk_expr(expr: ExprIR) -> None:
        for tup in expr:
            if not isinstance(tup, tuple) or not tup:
                continue
            if tup[0] == "OP_LOAD" and len(tup) > 1:
                add_name(str(tup[1]))

    def walk_stmt(stmt: Stmt) -> None:
        if not isinstance(stmt, tuple) or not stmt:
            return
        op = stmt[0]
        if op == "OP_ASSIGN":
            add_name(str(stmt[1]))
            walk_expr(list(stmt[2]))
        elif op == "OP_EXPR_STMT":
            walk_expr(list(stmt[1]))
        elif op == "OP_CONDITIONAL":
            walk_expr(list(stmt[1]))
            for s in stmt[2]:
                walk_stmt(tuple(s) if isinstance(s, list) else s)
            for s in stmt[3]:
                walk_stmt(tuple(s) if isinstance(s, list) else s)
        elif op == "OP_LOOP":
            walk_expr(list(stmt[1]))
            for s in stmt[2]:
                walk_stmt(tuple(s) if isinstance(s, list) else s)

    for instr in ir:
        walk_stmt(tuple(instr) if isinstance(instr, list) else instr)

    starts: Dict[str, int] = {}
    col = 0
    for name in order:
        w = max(1, int(widths.get(name, 1)))
        if max_vars is not None and col + w > max_vars:
            break
        starts[name] = col
        col += w
    return starts, widths


def extract_global_abi(ir: IRList, *, max_vars: Optional[int] = None) -> Dict[str, int]:
    """Variable name → **start** column (width may be >1; see ``extract_abi_layout``)."""
    starts, _ = extract_abi_layout(ir, max_vars=max_vars)
    return starts


def extract_abi_widths(ir: IRList, *, max_vars: Optional[int] = None) -> Dict[str, int]:
    """Per-name trunk column span (1 for scalars)."""
    starts, widths = extract_abi_layout(ir, max_vars=max_vars)
    return {n: max(1, int(widths.get(n, 1))) for n in starts}
