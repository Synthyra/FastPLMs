"""Generate an apply-patch update from exported official Boltz conformers."""

from __future__ import annotations

import argparse
import ast
import json
import pprint
import textwrap
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path)
    parser.add_argument("conformers", type=Path)
    parser.add_argument("--name", action="append", dest="names")
    parser.add_argument(
        "--repair-chirality",
        action="store_true",
        help="Regenerate the canonical chiral-atom table from the export.",
    )
    parser.add_argument(
        "--patch-path-label",
        help="Path written into the patch when it differs from the readable target.",
    )
    return parser


def _existing_table(source: str) -> dict[str, dict[str, list[float]]]:
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_RDKIT_CONFORMERS"
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                return value
    raise RuntimeError("Target source omits _RDKIT_CONFORMERS.")


def _assignment_lines(source: str, variable: str) -> list[str]:
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign):
            target = node.targets[0] if len(node.targets) == 1 else None
        else:
            target = None
        if isinstance(target, ast.Name) and target.id == variable:
            assert node.end_lineno is not None
            return lines[node.lineno - 1 : node.end_lineno]
    raise RuntimeError(f"Target source omits {variable}.")


def _mapping_entry_lines(source: str, variable: str, entry: str) -> list[str]:
    """Return every source line occupied by one top-level mapping entry."""

    tree = ast.parse(source)
    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        elif isinstance(node, ast.Assign):
            target = node.targets[0] if len(node.targets) == 1 else None
            value = node.value
        else:
            continue
        if not (isinstance(target, ast.Name) and target.id == variable):
            continue
        if not isinstance(value, ast.Dict):
            raise TypeError(f"{variable} must be a dictionary literal.")
        for key_node, value_node in zip(value.keys, value.values, strict=True):
            if (
                isinstance(key_node, ast.Constant)
                and key_node.value == entry
                and value_node.end_lineno is not None
            ):
                return lines[key_node.lineno - 1 : value_node.end_lineno]
        raise KeyError(f"{variable} omits {entry!r}.")
    raise RuntimeError(f"Target source omits {variable}.")


def main(argv: Sequence[str] | None = None) -> int:
    """Print a patch retaining full-precision coordinates for selected names."""

    arguments = _parser().parse_args(argv)
    source = arguments.target.read_text(encoding="utf-8")
    existing = _existing_table(source)
    exported = json.loads(arguments.conformers.read_text(encoding="utf-8"))
    if not arguments.names and not arguments.repair_chirality:
        raise ValueError("Select at least one conformer name or --repair-chirality.")
    print("*** Begin Patch")
    print(f"*** Update File: {arguments.patch_path_label or arguments.target}")
    if arguments.repair_chirality:
        old_lines = _assignment_lines(source, "_CHIRAL_ATOMS")
        chiral_atoms = {
            name: frozenset(atom["name"] for atom in value["atoms"] if atom["chirality"] != 0)
            for name, value in exported.items()
            if any(atom["chirality"] != 0 for atom in value["atoms"])
        }
        rendered = pprint.pformat(chiral_atoms, width=88, sort_dicts=False)
        rendered_lines = rendered.splitlines()
        new_lines = [f"_CHIRAL_ATOMS = {rendered_lines[0]}", *rendered_lines[1:]]
        print("@@")
        for line in old_lines:
            print(f"-{line}")
        for line in new_lines:
            print(f"+{line}")
    for name in arguments.names or ():
        old_lines = _mapping_entry_lines(source, "_RDKIT_CONFORMERS", name)
        raw = exported[name]["conformers"][0]
        entry = {atom: raw[atom] for atom in existing[name]}
        rendered = pprint.pformat(entry, width=88, sort_dicts=False)
        lines = textwrap.indent(rendered, "    ").splitlines()
        new_lines = [f'    "{name}": {lines[0].lstrip()}', *lines[1:]]
        new_lines[-1] += ","
        print("@@")
        for line in old_lines:
            print(f"-{line}")
        for line in new_lines:
            print(f"+{line}")
    print("*** End Patch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
