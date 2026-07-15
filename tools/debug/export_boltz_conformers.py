"""Export pinned Boltz molecule metadata as dependency-free Python data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("molecule_dir", type=Path)
    parser.add_argument(
        "--name",
        action="append",
        dest="names",
        help="Residue name to export; repeat for more than one residue.",
    )
    parser.add_argument("--output", type=Path, help="Write JSON to this path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print atom metadata and conformer coordinates from official pickles."""

    from boltz.data import const
    from boltz.data.mol import load_molecules

    arguments = _parser().parse_args(argv)
    names = (
        sorted(arguments.names)
        if arguments.names
        else sorted(path.stem for path in arguments.molecule_dir.glob("*.pkl"))
    )
    molecules = load_molecules(arguments.molecule_dir, names)
    unknown_chirality = const.chirality_type_ids[const.unk_chirality_type]
    payload = {}
    for name in names:
        molecule = molecules[name]
        atoms = []
        for atom in molecule.GetAtoms():
            atoms.append(
                {
                    "name": atom.GetProp("name"),
                    "charge": atom.GetFormalCharge(),
                    "chirality": const.chirality_type_ids.get(
                        atom.GetChiralTag().name,
                        unknown_chirality,
                    ),
                }
            )
        conformers = []
        for conformer in molecule.GetConformers():
            conformers.append(
                {
                    atom["name"]: [
                        conformer.GetAtomPosition(index).x,
                        conformer.GetAtomPosition(index).y,
                        conformer.GetAtomPosition(index).z,
                    ]
                    for index, atom in enumerate(atoms)
                }
            )
        payload[name] = {"atoms": atoms, "conformers": conformers}
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
        print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
