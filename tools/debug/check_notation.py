"""Reject noncanonical tensor-shape notation in documentation and comments."""

from __future__ import annotations

import argparse
import ast
import io
import re
import tokenize
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


TEXT_SUFFIXES = frozenset(
    {".hcl", ".in", ".md", ".rst", ".toml", ".txt", ".yaml", ".yml"}
)
SKIP_PARTS = frozenset(
    {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "LICENSES",
        "artifacts",
        "dist",
        "vendor",
        "__pycache__",
    }
)
SQUARE_CANDIDATE = re.compile(
    r"\[\s*(?:[A-Za-z][A-Za-z0-9_]*|\d+|\.\.\.|\*)"
    r"(?:\s*,\s*(?:[A-Za-z][A-Za-z0-9_]*|\d+|\.\.\.|\*))*\s*\]"
)
UPPER_PAREN_CANDIDATE = re.compile(
    r"\(\s*(?:[A-Za-z][A-Za-z0-9_]*|\d+)(?:\s*,\s*(?:[A-Za-z][A-Za-z0-9_]*|\d+)){1,}\s*\)"
)
COMMON_DIMENSIONS = frozenset(
    {
        "b",
        "l",
        "d",
        "n",
        "h",
        "q",
        "k",
        "v",
        "s",
        "t",
        "c",
        "m",
        "r",
        "p",
        "batch",
        "batch_size",
        "length",
        "seq_len",
        "sequence_length",
        "hidden_size",
        "d_model",
        "d_head",
        "n_layers",
        "n_heads",
        "heads",
        "layers",
        "atoms",
        "channels",
        "items",
        "residues",
        "samples",
        "sequences",
        "tokens",
    }
)


@dataclass(frozen=True, slots=True)
class Violation:
    path: Path
    line: int
    column: int
    message: str
    excerpt: str

    def render(self, root: Path) -> str:
        relative = self.path.resolve().relative_to(root.resolve())
        return (
            f"{relative.as_posix()}:{self.line}:{self.column}: {self.message}: "
            f"{self.excerpt.strip()}"
        )


def _tokens(candidate: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in candidate[1:-1].split(","))


def _is_dimension_name(token: str) -> bool:
    lowered = token.lower()
    return (
        lowered in COMMON_DIMENSIONS
        or lowered.endswith(("_dim", "_dims", "_len", "_length", "_size"))
        or lowered.startswith(
            (
                "b_",
                "l_",
                "d_",
                "n_",
                "h_",
                "q_",
                "k_",
                "v_",
                "s_",
                "t_",
                "c_",
                "m_",
                "r_",
                "p_",
                "batch_",
                "seq_",
                "sequence_",
                "hidden_",
                "head_",
                "layer_",
            )
        )
    )


def violations_in_text(
    text: str,
    *,
    path: Path,
    first_line: int = 1,
) -> Iterator[Violation]:
    """Yield notation violations in one documentation string or comment."""

    for offset, line in enumerate(text.splitlines() or (text,)):
        line_number = first_line + offset
        for match in SQUARE_CANDIDATE.finditer(line):
            dimensions = _tokens(match.group())
            if any(_is_dimension_name(token) for token in dimensions):
                yield Violation(
                    path,
                    line_number,
                    match.start() + 1,
                    "shape signatures must use parentheses",
                    match.group(),
                )
        for match in UPPER_PAREN_CANDIDATE.finditer(line):
            dimensions = _tokens(match.group())
            uppercase_dimensions = [
                token
                for token in dimensions
                if token.isidentifier() and token.upper() == token and _is_dimension_name(token)
            ]
            if uppercase_dimensions:
                yield Violation(
                    path,
                    line_number,
                    match.start() + 1,
                    "shape dimensions must use lowercase symbols",
                    match.group(),
                )


def _python_documentation(path: Path) -> Iterator[tuple[str, int]]:
    """Yield comments and real docstrings with source line numbers."""

    source = path.read_text(encoding="utf-8")
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                yield token.string[1:], token.start[0]
    except (IndentationError, tokenize.TokenError) as error:
        raise ValueError(f"Cannot tokenize {path}: {error}") from error
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        raise ValueError(f"Cannot parse {path}: {error}") from error
    documentable = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, documentable) or not node.body:
            continue
        expression = node.body[0]
        if (
            isinstance(expression, ast.Expr)
            and isinstance(expression.value, ast.Constant)
            and isinstance(expression.value.value, str)
        ):
            yield expression.value.value, expression.lineno


def iter_repository_files(root: Path) -> Iterator[Path]:
    """Yield tracked documentation and Python paths inside the source boundary."""

    candidates = (
        root / "README.md",
        root / "AGENTS.md",
        root / "THIRD_PARTY_NOTICES.md",
        root / "LICENSES" / "README.md",
        root / "vendor" / "README.md",
        root / "requirements",
        root / "docs",
        root / "model_cards",
        root / "docker",
        root / "src",
        root / "tests",
        root / "benchmarks",
        root / "tools",
        root / "examples",
    )
    for candidate in candidates:
        if candidate.is_file():
            yield candidate
            continue
        if not candidate.is_dir():
            continue
        for path in sorted(candidate.rglob("*")):
            if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
                continue
            if path.suffix in TEXT_SUFFIXES or path.suffix == ".py" or path.name == "Dockerfile":
                yield path


def scan_repository(root: Path) -> list[Violation]:
    """Return every shape-notation violation in repository prose."""

    result: list[Violation] = []
    for path in iter_repository_files(root):
        if path.suffix == ".py":
            regions = _python_documentation(path)
        else:
            regions = ((path.read_text(encoding="utf-8"), 1),)
        for text, first_line in regions:
            result.extend(violations_in_text(text, path=path, first_line=first_line))
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    arguments = parser.parse_args(argv)
    root = arguments.source_root.resolve()
    violations = scan_repository(root)
    for violation in violations:
        print(violation.render(root))
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
