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
    # ?comparison/sum/product may collapse; assign/if pass the lowest kept rule.
    if t.data == "comparison":
        return _comparison(t)
    if t.data == "sum":
        return _sum(t)
    if t.data == "product":
        return _product(t)
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
    if t.data == "atom":
        return _atom(t)
    raise ValueError(f"expected product|atom under sum, got {t.data}")


def _product(t: Tree) -> List[tuple]:
    atoms = t.children[0::2]
    ops = t.children[1::2]
    acc = _atom(atoms[0])
    for i, op in enumerate(ops):
        acc += _atom(atoms[i + 1])
        acc += [("OP_MUL",) if op.type == "MUL" else ("OP_DIV",)]
    return acc


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
            if x.data == "comparison":
                return _comparison(x)
            if x.data == "expr":
                return _expr(x.children[0])
            if x.data in ("sum", "product", "comparison"):
                return _expr(x)
    raise ValueError(f"bad atom {ch!r}")


def _number(tok: Token) -> int | float:
    s = str(tok)
    if "." in s or "e" in s.lower():
        return float(s)
    return int(s)


Stmt = Tuple[Any, ...]
ExprIR = List[Tuple[Any, ...]]


def extract_global_abi(ir: IRList, *, max_vars: Optional[int] = None) -> Dict[str, int]:
    """First-seen variable order across the whole program (document order). Maps name -> trunk column."""
    order: List[str] = []
    seen: Set[str] = set()

    def add_name(name: str) -> None:
        n = str(name)
        if n in seen:
            return
        if max_vars is not None and len(order) >= max_vars:
            return
        seen.add(n)
        order.append(n)

    def walk_expr(expr: ExprIR) -> None:
        for tup in expr:
            if not isinstance(tup, tuple) or not tup:
                continue
            if tup[0] == "OP_LOAD" and len(tup) > 1:
                add_name(tup[1])

    def walk_stmt(stmt: Stmt) -> None:
        if not isinstance(stmt, tuple) or not stmt:
            return
        op = stmt[0]
        if op == "OP_ASSIGN":
            add_name(stmt[1])
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
    return {name: i for i, name in enumerate(order)}
