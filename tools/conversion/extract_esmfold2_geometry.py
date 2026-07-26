"""Extract the pinned ESMFold2 protein geometry table as declarative JSON."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path


def _validate_geometry(residues: object) -> dict[str, dict[str, tuple[float, float, float]]]:
    if not isinstance(residues, dict) or not residues:
        raise ValueError("PROTEIN_REF_POS must be a non-empty dictionary.")

    validated: dict[str, dict[str, tuple[float, float, float]]] = {}
    for residue, atoms in residues.items():
        if not isinstance(residue, str) or not residue:
            raise ValueError("PROTEIN_REF_POS residue names must be non-empty strings.")
        if not isinstance(atoms, dict) or not atoms:
            raise ValueError(f"PROTEIN_REF_POS[{residue!r}] must be a non-empty dictionary.")
        validated_atoms: dict[str, tuple[float, float, float]] = {}
        for atom, coordinates in atoms.items():
            if not isinstance(atom, str) or not atom:
                raise ValueError(f"PROTEIN_REF_POS[{residue!r}] has an invalid atom name.")
            if not isinstance(coordinates, (tuple, list)) or len(coordinates) != 3:
                raise ValueError(
                    f"PROTEIN_REF_POS[{residue!r}][{atom!r}] must contain three coordinates."
                )
            values: list[float] = []
            for coordinate in coordinates:
                if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                    raise ValueError(
                        f"PROTEIN_REF_POS[{residue!r}][{atom!r}] contains a non-numeric coordinate."
                    )
                value = float(coordinate)
                if not math.isfinite(value):
                    raise ValueError(
                        f"PROTEIN_REF_POS[{residue!r}][{atom!r}] contains a non-finite coordinate."
                    )
                values.append(value)
            validated_atoms[atom] = (values[0], values[1], values[2])
        validated[residue] = validated_atoms
    return validated


def extract_geometry(source: Path) -> dict[str, object]:
    """Read ``PROTEIN_REF_POS`` without importing or executing upstream code."""

    module = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for statement in module.body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "PROTEIN_REF_POS"
            and statement.value is not None
        ):
            try:
                residues = ast.literal_eval(statement.value)
            except (SyntaxError, TypeError, ValueError) as error:
                raise ValueError("PROTEIN_REF_POS must be a Python literal.") from error
            return {
                "schema": "fastplms.esmfold2.reference_geometry.v1",
                "provenance": {
                    "manifest_family": "esmfold2",
                    "contract": "biohub_esmfold2_input_v1",
                },
                "dtype": "float32",
                "residues": _validate_geometry(residues),
            }
    raise ValueError(f"PROTEIN_REF_POS was not found in {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = extract_geometry(args.source)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
