"""Measures the source-independence tests share: how alike a runtime function or file is to its pinned upstream source."""

from __future__ import annotations

import ast
import copy

from pathlib import Path


def function_node(path: Path, qualified_name: str) -> ast.FunctionDef:
    """Return the function ``Class.method`` or ``function`` of the module at ``path``, or fail when it is absent."""

    body: list[ast.stmt] = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    ).body
    selected: ast.AST | None = None
    for part in qualified_name.split("."):
        selected = next(
            (
                node
                for node in body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name == part
            ),
            None,
        )
        assert selected is not None, f"{qualified_name!r} is absent from {path}"
        body = selected.body
    assert isinstance(selected, ast.FunctionDef)
    return selected


def normalized_ast_lines(node: ast.FunctionDef) -> list[str]:
    """Return the unparsed lines of ``node`` with its name, decorators, annotations, and docstring removed."""

    normalized = copy.deepcopy(node)
    normalized.name = "function"
    normalized.decorator_list = []
    normalized.returns = None
    for argument in (
        *normalized.args.posonlyargs,
        *normalized.args.args,
        *normalized.args.kwonlyargs,
    ):
        argument.annotation = None
    if normalized.args.vararg is not None:
        normalized.args.vararg.annotation = None
    if normalized.args.kwarg is not None:
        normalized.args.kwarg.annotation = None
    if (
        normalized.body
        and isinstance(normalized.body[0], ast.Expr)
        and isinstance(normalized.body[0].value, ast.Constant)
        and isinstance(normalized.body[0].value.value, str)
    ):
        normalized.body.pop(0)
    ast.fix_missing_locations(normalized)
    return [
        " ".join(line.strip().split())
        for line in ast.unparse(normalized).splitlines()
        if line.strip()
    ]


def meaningful_lines(text: str) -> list[str]:
    """Return the non-blank, non-comment lines of ``text`` with whitespace collapsed."""

    return [
        " ".join(line.strip().split())
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
