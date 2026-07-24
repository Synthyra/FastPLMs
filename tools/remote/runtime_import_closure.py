"""Statically attest the declared dependency closure of FastPLMs runtime source."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_DISTRIBUTION_IMPORT_NAMES = {
    "biopython": "Bio",
    "huggingface-hub": "huggingface_hub",
    "msgpack-numpy": "msgpack_numpy",
    "scikit-learn": "sklearn",
    "transformer-engine": "transformer_engine",
    "transformer-engine-cu13": "transformer_engine",
    "transformer-engine-torch": "transformer_engine",
}


class RuntimeImportClosureError(RuntimeError):
    """Runtime source contains an undeclared import-time dependency."""


@dataclass(frozen=True)
class _ImportRecord:
    module: str
    source: str
    line: int
    kind: str
    guarded: bool
    source_scope: str


def _normalized_distribution(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        raise RuntimeImportClosureError(f"Invalid dependency requirement: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _import_name(distribution: str) -> str:
    return _DISTRIBUTION_IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))


def _read_direct_requirements(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeImportClosureError(f"Missing direct dependency declaration: {path}")
    requirements: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        requirement = raw_line.partition("#")[0].strip()
        if not requirement:
            continue
        if requirement.startswith(("-", "--")):
            raise RuntimeImportClosureError(
                f"Direct dependency file may not compose other files: {path}:{line_number}"
            )
        requirements.append(requirement)
    return requirements


def _declared_import_scopes(requirements_root: Path) -> dict[str, list[str]]:
    scoped_requirements = {
        "core": _read_direct_requirements(requirements_root / "core.in"),
    }
    features_root = requirements_root / "features"
    if not features_root.is_dir():
        raise RuntimeImportClosureError(
            f"Missing feature dependency declarations: {features_root}"
        )
    for path in sorted(features_root.glob("*.in")):
        scoped_requirements[f"extra:{path.stem}"] = _read_direct_requirements(path)

    scopes: defaultdict[str, set[str]] = defaultdict(set)
    for scope, requirements in scoped_requirements.items():
        for requirement in requirements:
            distribution = _normalized_distribution(requirement)
            scopes[_import_name(distribution)].add(scope)
    return {name: sorted(values) for name, values in sorted(scopes.items())}


def _literal_dynamic_import(node: ast.Call) -> str | None:
    function = node.func
    is_import = (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "importlib"
        and function.attr == "import_module"
    ) or (isinstance(function, ast.Name) and function.id in {"__import__", "import_module"})
    if not is_import or not node.args:
        return None
    value = node.args[0]
    return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None


def _family_runtime_scopes(source_root: Path) -> dict[str, tuple[str, ...]]:
    """Return manifest-declared dependency scopes for runtime path prefixes."""

    resolved_source_root = source_root.resolve()
    manifest_path = source_root / "models.toml"
    if not manifest_path.is_file():
        return {}
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    families = manifest.get("families", {})
    if not isinstance(families, dict):
        raise RuntimeImportClosureError("models.toml [families] must be a table")

    scopes: defaultdict[str, set[str]] = defaultdict(set)
    for family_name, raw_family in families.items():
        if not isinstance(raw_family, dict):
            raise RuntimeImportClosureError(
                f"models.toml family {family_name!r} must be a table"
            )
        extra = raw_family.get("extra")
        runtime_paths = raw_family.get("runtime_paths")
        if not isinstance(extra, str) or not isinstance(runtime_paths, list):
            raise RuntimeImportClosureError(
                f"models.toml family {family_name!r} needs extra and runtime_paths"
            )
        scope = "core" if extra == "core" else f"extra:{extra}"
        for raw_path in runtime_paths:
            if not isinstance(raw_path, str):
                raise RuntimeImportClosureError(
                    f"models.toml family {family_name!r} has a non-string runtime path"
                )
            path = PurePosixPath(raw_path)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != raw_path
            ):
                raise RuntimeImportClosureError(
                    f"models.toml family {family_name!r} has a non-portable runtime path"
                )
            resolved_path = source_root.joinpath(*path.parts).resolve()
            if (
                not resolved_path.is_relative_to(resolved_source_root)
                or not resolved_path.exists()
            ):
                raise RuntimeImportClosureError(
                    f"models.toml family {family_name!r} has a runtime path "
                    "outside the runtime source root"
                )
            scopes[raw_path].add(scope)
    return {path: tuple(sorted(values)) for path, values in sorted(scopes.items())}


def _source_scope(
    relative: str,
    runtime_scopes: Mapping[str, tuple[str, ...]],
) -> str:
    matches = [
        (len(PurePosixPath(prefix).parts), scopes)
        for prefix, scopes in runtime_scopes.items()
        if relative == prefix or relative.startswith(f"{prefix}/")
    ]
    if not matches:
        return "core"
    specificity = max(length for length, _ in matches)
    scopes = {
        scope
        for length, declared in matches
        if length == specificity
        for scope in declared
    }
    if "core" in scopes:
        return "core"
    if len(scopes) != 1:
        raise RuntimeImportClosureError(
            f"Runtime source {relative!r} maps to ambiguous feature scopes: {sorted(scopes)}"
        )
    return next(iter(scopes))


def _caught_exception_names(node: ast.expr | None) -> set[str]:
    if node is None:
        return {"BaseException"}
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Tuple):
        return {
            name
            for element in node.elts
            for name in _caught_exception_names(element)
        }
    return set()


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    return bool(
        _caught_exception_names(handler.type)
        & {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}
    )


class _RuntimeImportVisitor(ast.NodeVisitor):
    """Collect imports while retaining whether execution is feature guarded."""

    def __init__(self, *, source: str, source_scope: str) -> None:
        self.source = source
        self.source_scope = source_scope
        self.records: list[_ImportRecord] = []
        self._guard_depth = 0

    def _record(self, module: str, line: int, kind: str) -> None:
        top_level = module.partition(".")[0]
        if top_level in {"fastplms"} or top_level in sys.stdlib_module_names:
            return
        self.records.append(
            _ImportRecord(
                module=top_level,
                source=self.source,
                line=line,
                kind=kind,
                guarded=self._guard_depth > 0,
                source_scope=self.source_scope,
            )
        )

    def _visit_statements(
        self,
        statements: Sequence[ast.stmt],
        *,
        guarded: bool,
    ) -> None:
        if guarded:
            self._guard_depth += 1
        try:
            for statement in statements:
                self.visit(statement)
        finally:
            if guarded:
                self._guard_depth -= 1

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if node.args.vararg is not None and node.args.vararg.annotation is not None:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
            self.visit(node.args.kwarg.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        for type_parameter in getattr(node, "type_params", ()):
            self.visit(type_parameter)
        self._visit_statements(node.body, guarded=True)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self._guard_depth += 1
        try:
            self.visit(node.body)
        finally:
            self._guard_depth -= 1

    def _visit_try(self, node: ast.Try | ast.TryStar) -> None:
        catches_import_error = any(_catches_import_error(handler) for handler in node.handlers)
        self._visit_statements(node.body, guarded=catches_import_error)
        for handler in node.handlers:
            if handler.type is not None:
                self.visit(handler.type)
            self._visit_statements(handler.body, guarded=False)
        self._visit_statements(node.orelse, guarded=False)
        self._visit_statements(node.finalbody, guarded=False)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_try(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._visit_try(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record(alias.name, node.lineno, "static")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            self._record(node.module, node.lineno, "static")

    def visit_Call(self, node: ast.Call) -> None:
        module = _literal_dynamic_import(node)
        if module is not None:
            self._record(module, node.lineno, "dynamic")
        self.generic_visit(node)


def _record_sort_key(record: Mapping[str, object]) -> tuple[str, str, int, str]:
    line = record["line"]
    if not isinstance(line, int):
        raise RuntimeImportClosureError("Import record line must be an integer")
    return (
        str(record["module"]),
        str(record["source"]),
        line,
        str(record["kind"]),
    )


def _resolved_record(
    record: _ImportRecord,
    declared_scopes: Sequence[str],
    required_scope: str,
) -> dict[str, object]:
    return {
        "module": record.module,
        "source": record.source,
        "line": record.line,
        "kind": record.kind,
        "source_scope": record.source_scope,
        "required_scope": required_scope,
        "declared_scopes": list(declared_scopes),
    }


def inspect_runtime_import_closure(
    source_root: Path,
    requirements_root: Path,
) -> dict[str, object]:
    """Return deterministic, scope-aware closure evidence or fail closed."""

    scopes = _declared_import_scopes(requirements_root)
    runtime_scopes = _family_runtime_scopes(source_root)
    records: list[_ImportRecord] = []
    source_files = sorted(source_root.rglob("*.py"))
    if not source_files:
        raise RuntimeImportClosureError(f"No Python runtime source found under {source_root}")

    for path in source_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(source_root).as_posix()
        visitor = _RuntimeImportVisitor(
            source=relative,
            source_scope=_source_scope(relative, runtime_scopes),
        )
        visitor.visit(tree)
        records.extend(visitor.records)

    undeclared_static = sorted(
        {
            record.module
            for record in records
            if record.kind == "static" and record.module not in scopes
        }
    )
    if undeclared_static:
        raise RuntimeImportClosureError(
            "Runtime source has undeclared import-time dependencies: "
            f"{undeclared_static}"
        )
    undeclared_dynamic = sorted({
        record.module
        for record in records
        if record.kind == "dynamic" and record.module not in scopes
    })
    if undeclared_dynamic:
        raise RuntimeImportClosureError(
            "Runtime source has undeclared literal dynamic dependencies: "
            f"{undeclared_dynamic}"
        )

    resolved: list[tuple[_ImportRecord, dict[str, object]]] = []
    unconditional_mismatches: list[str] = []
    guarded_mismatches: list[str] = []
    ambiguous_guarded: list[str] = []
    for record in records:
        declared = scopes[record.module]
        if not record.guarded:
            required_scope = (
                "core" if "core" in declared else record.source_scope
            )
        elif record.source_scope != "core" and record.source_scope in declared:
            required_scope = record.source_scope
        elif "core" in declared:
            required_scope = "core"
        elif len(declared) == 1:
            required_scope = declared[0]
        else:
            ambiguous_guarded.append(
                f"{record.module}@{record.source}:{record.line} declared={declared}"
            )
            continue

        if required_scope not in declared:
            mismatch = (
                f"{record.module}@{record.source}:{record.line} "
                f"requires={required_scope} declared={declared}"
            )
            if record.guarded:
                guarded_mismatches.append(mismatch)
            else:
                unconditional_mismatches.append(mismatch)
            continue
        resolved.append(
            (record, _resolved_record(record, declared, required_scope))
        )

    if ambiguous_guarded:
        raise RuntimeImportClosureError(
            "Guarded import does not map to one intended dependency scope: "
            f"{sorted(ambiguous_guarded)}"
        )
    if unconditional_mismatches:
        raise RuntimeImportClosureError(
            "Unconditional import dependency scope mismatch: "
            f"{sorted(unconditional_mismatches)}"
        )
    if guarded_mismatches:
        raise RuntimeImportClosureError(
            f"Guarded import dependency scope mismatch: {sorted(guarded_mismatches)}"
        )

    import_time = sorted(
        (payload for record, payload in resolved if not record.guarded),
        key=_record_sort_key,
    )
    feature_gated = sorted(
        (
            payload
            for record, payload in resolved
            if record.guarded or payload["required_scope"] != "core"
        ),
        key=_record_sort_key,
    )
    dynamic = sorted(
        (record for record in feature_gated if record["kind"] == "dynamic"),
        key=_record_sort_key,
    )
    return {
        "schema_version": 2,
        "source_files": len(source_files),
        "import_time_dependencies": import_time,
        "feature_gated_imports": feature_gated,
        "feature_gated_dynamic_imports": dynamic,
        "undeclared_import_time_dependencies": [],
        "undeclared_literal_dynamic_dependencies": [],
        "scope_mismatches": [],
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("src/fastplms"))
    parser.add_argument("--requirements-root", type=Path, default=Path("requirements"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runtime-import-closure.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    payload = inspect_runtime_import_closure(
        arguments.source_root.resolve(),
        arguments.requirements_root.resolve(),
    )
    _atomic_json(arguments.output, payload)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
