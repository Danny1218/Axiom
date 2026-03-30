from __future__ import annotations

from pathlib import Path

from lark import Lark, Tree

_GRAMMAR = Path(__file__).with_name("grammar.lark")
_parser: Lark | None = None


def reset_parser() -> None:
    global _parser
    _parser = None


def _get_parser() -> Lark:
    global _parser
    if _parser is None:
        _parser = Lark.open(str(_GRAMMAR), parser="lalr", start="start", propagate_positions=True)
    return _parser


def parse_ax(source: str) -> Tree:
    return _get_parser().parse(source)


def parse_ax_file(path: str | Path) -> Tree:
    return parse_ax(Path(path).read_text(encoding="utf-8"))


def parse_ax_program(source: str):
    """Return ``(function_defs, main_ir)`` with calls not yet expanded (see ``ast_to_ir``)."""
    from axiom.compiler.ir import parse_program as program_from_tree

    return program_from_tree(parse_ax(source))


# String literals (e.g. ``neural(x, "kan")``) are lowered in ``axiom.compiler.ir`` via ``StringLiteral`` / ``_postfix_expr``; there is no separate Lark Transformer class in this module.
