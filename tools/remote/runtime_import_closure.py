"""Statically attest the declared dependency closure of installable runtime source."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

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
    """Installable source contains an undeclared import-time dependency."""


def _normalized_distribution(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        raise RuntimeImportClosureError(f"Invalid dependency requirement: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _import_name(distribution: str) -> str:
    return _DISTRIBUTION_IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))


def _declared_import_scopes(project: Mapping[str, object]) -> dict[str, list[str]]:
    metadata = project.get("project")
    if not isinstance(metadata, dict):
        raise RuntimeImportClosureError("pyproject.toml is missing [project]")
    scoped_requirements: dict[str, object] = {"core": metadata.get("dependencies", [])}
    extras = metadata.get("optional-dependencies", {})
    if not isinstance(extras, dict):
        raise RuntimeImportClosureError("[project.optional-dependencies] must be a table")
    scoped_requirements.update({f"extra:{name}": value for name, value in extras.items()})

    scopes: defaultdict[str, set[str]] = defaultdict(set)
    for scope, requirements in scoped_requirements.items():
        if not isinstance(requirements, list):
            raise RuntimeImportClosureError(f"Dependency scope {scope!r} must be a string list")
        typed_requirements: list[str] = []
        for requirement in requirements:
            if not isinstance(requirement, str):
                raise RuntimeImportClosureError(
                    f"Dependency scope {scope!r} must be a string list"
                )
            typed_requirements.append(requirement)
        for requirement in typed_requirements:
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


def _dynamic_record_sort_key(record: Mapping[str, object]) -> tuple[str, str, int]:
    """Return a typed deterministic key for one dynamic-import record."""

    line = record["line"]
    if not isinstance(line, int):
        raise RuntimeImportClosureError("Dynamic-import record line must be an integer")
    return str(record["module"]), str(record["source"]), line


def inspect_runtime_import_closure(
    source_root: Path,
    pyproject: Path,
) -> dict[str, object]:
    """Return deterministic closure evidence or fail on an undeclared import."""

    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scopes = _declared_import_scopes(project)
    unconditional: set[str] = set()
    dynamic_records: list[dict[str, object]] = []
    source_files = sorted(source_root.rglob("*.py"))
    if not source_files:
        raise RuntimeImportClosureError(f"No installable Python source found under {source_root}")

    for path in source_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(source_root).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                unconditional.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                unconditional.add(node.module.partition(".")[0])
            elif isinstance(node, ast.Call):
                module = _literal_dynamic_import(node)
                if module is not None:
                    top_level = module.partition(".")[0]
                    if top_level not in {"fastplms"} and top_level not in sys.stdlib_module_names:
                        dynamic_records.append(
                            {
                                "module": top_level,
                                "source": relative,
                                "line": node.lineno,
                                "declared_scopes": scopes.get(top_level, []),
                            }
                        )

    external = sorted(
        name
        for name in unconditional
        if name not in {"fastplms"} and name not in sys.stdlib_module_names
    )
    undeclared = sorted(name for name in external if name not in scopes)
    if undeclared:
        raise RuntimeImportClosureError(
            f"Installable source has undeclared import-time dependencies: {undeclared}"
        )
    undeclared_dynamic = sorted(
        {
            str(record["module"])
            for record in dynamic_records
            if not record["declared_scopes"]
        }
    )
    if undeclared_dynamic:
        raise RuntimeImportClosureError(
            "Installable source has undeclared literal dynamic dependencies: "
            f"{undeclared_dynamic}"
        )
    return {
        "schema_version": 1,
        "source_files": len(source_files),
        "import_time_dependencies": [
            {"module": name, "declared_scopes": scopes[name]} for name in external
        ],
        "feature_gated_dynamic_imports": sorted(
            dynamic_records,
            key=_dynamic_record_sort_key,
        ),
        "undeclared_import_time_dependencies": [],
        "undeclared_literal_dynamic_dependencies": [],
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("src/fastplms"))
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
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
        arguments.pyproject.resolve(),
    )
    _atomic_json(arguments.output, payload)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
