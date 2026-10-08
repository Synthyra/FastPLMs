"""Small synthetic protein structures for the ESMFold2 structure-utility tests.

A backbone of idealized N, CA, and C atoms is placed along a helix, and every other atom of a residue sits at a seeded
random offset from its alpha carbon. The geometry is not physical; the tests need coordinates whose frames, distances,
and surface areas are well defined and repeatable, not a real protein.

Shapes: `l` residues in a chain.
"""

import io
import numpy as np
import torch

from dataclasses import replace
from pathlib import Path
from biotite.structure.io.pdbx import CIFCategory, CIFColumn, CIFData, CIFFile

from fastplms.models.esmfold2 import esmfold2_residue_constants as residue_constants
from fastplms.models.esmfold2.esmfold2_molecular_complex import MolecularComplex
from fastplms.models.esmfold2.esmfold2_protein_chain import ProteinChain
from fastplms.models.esmfold2.esmfold2_protein_complex import ProteinComplex


IDEAL_BACKBONE = [[-0.525, 1.363, 0.0], [0.0, 0.0, 0.0], [1.526, 0.0, 0.0]]  # N, CA, C of one residue, in angstroms
BACKBONE_ATOMS = ("N", "CA", "C")
HELIX_TURN = 1.7  # radians between consecutive residues
HELIX_RISE = 1.5  # angstroms between consecutive residues
SIDE_CHAIN_SPREAD = 1.8  # standard deviation of a non-backbone atom's offset from its alpha carbon, in angstroms


def z_rotation(angle: float) -> torch.Tensor:
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    return torch.tensor([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])  # (3, 3)


def helix_backbone(length: int) -> torch.Tensor:
    """N, CA, C of `length` residues, each turned by HELIX_TURN radians and HELIX_RISE angstroms higher than the one before."""
    angles = torch.arange(length) * HELIX_TURN  # (l,)
    rotations = torch.stack([z_rotation(float(angle)) for angle in angles])  # (l, 3, 3)
    shifts = torch.stack([2 * angles.cos(), 2 * angles.sin(), HELIX_RISE * torch.arange(length)], dim=-1)  # (l, 3)
    placed = torch.einsum("lij,aj->lai", rotations, torch.tensor(IDEAL_BACKBONE)) + shifts[:, None, :]  # (l, 3, 3)
    return placed.contiguous()  # (l, 3, 3), laid out so that it can be viewed as (l * 3, 3)


def atom37_of(backbone: torch.Tensor) -> torch.Tensor:
    # backbone: (l, 3, 3) holding N, CA, C; every other atom is unresolved (infinite).
    atom37 = torch.full((backbone.shape[0], 37, 3), torch.inf)  # (l, 37, 3)
    for position, name in enumerate(BACKBONE_ATOMS):
        atom37[:, residue_constants.atom_order[name]] = backbone[:, position]
    return atom37  # (l, 37, 3)


def moved(points: torch.Tensor) -> torch.Tensor:
    # points: (..., 3), moved by a fixed rotation about z followed by a shift.
    return (points @ z_rotation(0.7).T + torch.tensor([5.0, -3.0, 2.0])).contiguous()  # (..., 3)


def moved_chain(structure: ProteinChain, angle: float = 0.9, shift: tuple[float, float, float] = (3.0, -2.0, 7.0)) -> ProteinChain:
    """The same chain turned about z and shifted rigidly; atoms that are missing stay missing."""
    rotation = z_rotation(angle).numpy()  # (3, 3)
    positions = structure.atom37_positions @ rotation.T + np.asarray(shift, dtype=np.float32)  # (l, 37, 3)
    return replace(structure, atom37_positions=positions.astype(np.float32))


def synthetic_chain(
    sequence: str = "ACDEFGHIKL",
    chain_id: str = "A",
    entity_id: int | None = 1,
    shift: float = 0.0,
    seed: int = 0,
    structure_id: str = "syn1",
) -> ProteinChain:
    """A chain whose residues carry every heavy atom the residue type has, at helix-backbone and seeded positions."""
    generator = np.random.default_rng(seed)
    length = len(sequence)
    backbone = helix_backbone(length).numpy() + shift  # (l, 3, 3)
    positions = np.full((length, 37, 3), np.nan, dtype=np.float32)  # (l, 37, 3)
    present = np.zeros((length, 37), dtype=bool)  # (l, 37)
    for index, letter in enumerate(sequence):
        atom_names = residue_constants.restype_name_to_atom14_names[residue_constants.restype_1to3[letter]]
        for atom_name in filter(None, atom_names):
            slot = residue_constants.atom_order[atom_name]
            if atom_name in BACKBONE_ATOMS:
                positions[index, slot] = backbone[index, BACKBONE_ATOMS.index(atom_name)]
            else:
                positions[index, slot] = backbone[index, 1] + generator.normal(0.0, SIDE_CHAIN_SPREAD, 3)
            present[index, slot] = True
    return ProteinChain(
        id=structure_id,
        sequence=sequence,
        chain_id=chain_id,
        entity_id=entity_id,
        residue_index=np.arange(1, length + 1),
        insertion_code=np.full(length, "", dtype="<U4"),
        atom37_positions=positions,
        atom37_mask=present,
        confidence=np.linspace(0.5, 0.9, length).astype(np.float32),
    )


def dockq_report(mapping: str = "AB:AB", total: str = "0.9") -> bytes:
    """Text in the format that the DockQ program prints and that the complex scoring methods parse."""
    lines = [
        "****************************************************************",
        "*                            DockQ                             *",
        "****************************************************************",
        "Model  : /tmp/self.pdb",
        "Native : /tmp/native.pdb",
        f"Total DockQ over 1 native interfaces: {total} with {mapping} model:native mapping",
        "Native chains: A, B",
        "\tModel chains: A, B",
        "\tDockQ: 0.9",
        "\tirms: 1.2",
        "\tLrms: 2.3",
        "\tfnat: 0.8",
        "\tfnonnat: 0.1",
        "\tclashes: 0.0",
        "\tF1: 0.85",
        "\tDockQ_F1: 0.87",
    ]
    return "\n".join(lines).encode()


def stub_dockq(monkeypatch, module, report: bytes) -> list[list[str]]:
    """Replace the DockQ program in `module` with one that checks both PDB files exist and returns `report`."""
    commands: list[list[str]] = []

    def check_output(command):
        commands.append([str(part) for part in command])
        assert all(Path(part).exists() for part in command[1:])
        return report

    monkeypatch.setattr(module, "check_output", check_output)
    return commands


def cif_column(values: list[str]) -> CIFColumn:
    return CIFColumn(data=CIFData(array=np.asarray(values), dtype=np.str_))


def with_zinc(structure: MolecularComplex, position: np.ndarray) -> MolecularComplex:
    """The complex plus one zinc ion, as a chain of its own with one non-polymer token."""
    # position: (3,) coordinates of the ion.
    atoms = len(structure.atom_positions)
    new_chain = int(structure.chain_id.max()) + 1
    return replace(
        structure,
        sequence=[*structure.sequence, "ZN"],
        atom_positions=np.vstack([structure.atom_positions, np.asarray(position, dtype=np.float32)[None]]),  # (n + 1, 3)
        atom_elements=np.append(structure.atom_elements, "ZN"),
        token_to_atoms=np.vstack([structure.token_to_atoms, [[atoms, atoms + 1]]]).astype(np.int32),  # (t + 1, 2)
        chain_id=np.append(structure.chain_id, new_chain),
        plddt=np.append(structure.plddt, np.float32(1.0)),
        metadata=replace(structure.metadata, chain_lookup={**structure.metadata.chain_lookup, new_chain: "Z"}),
        atom_names=np.append(structure.atom_names, "ZN"),
        atom_hetero=np.append(structure.atom_hetero, True),
        entity_id=np.append(structure.entity_id, int(structure.entity_id.max()) + 1),
        sym_id=np.append(structure.sym_id, 0),
    )


def synthetic_mmcif(chains: list[ProteinChain], assemblies: bool = True, zinc_at: np.ndarray | None = None) -> str:
    """mmCIF text of the chains that the structure readers accept.

    The writer of `MolecularComplex` supplies the atoms, entities, and polymer sequences. This adds the residue scheme
    that maps sequence positions to residue numbers, and, with `assemblies`, two biological assemblies: assembly 1
    holds every chain once, and assembly 2 holds the first chain twice, the second copy moved 20 angstroms along x.
    """
    # zinc_at: (3,) coordinates of an optional zinc ion.
    molecular = MolecularComplex.from_protein_complex(ProteinComplex.from_chains(chains))
    if zinc_at is not None:
        molecular = with_zinc(molecular, zinc_at)
    cif = CIFFile.read(io.StringIO(molecular.to_mmcif()))
    block = cif.block
    scheme: dict[str, list[str]] = {name: [] for name in ("asym_id", "entity_id", "seq_id", "mon_id", "pdb_strand_id", "auth_seq_num", "pdb_ins_code", "hetero")}
    for chain in chains:
        for position, letter in enumerate(chain.sequence):
            row = (chain.chain_id, str(chain.entity_id), str(position + 1), residue_constants.restype_1to3[letter], chain.chain_id, str(int(chain.residue_index[position])), ".", "n")
            for name, value in zip(scheme, row, strict=True):
                scheme[name].append(value)
    block["pdbx_poly_seq_scheme"] = CIFCategory(name="pdbx_poly_seq_scheme", columns={name: cif_column(values) for name, values in scheme.items()})
    if assemblies:
        every_chain = ",".join(chain.chain_id for chain in chains)
        block["pdbx_struct_assembly_gen"] = CIFCategory(
            name="pdbx_struct_assembly_gen",
            columns={
                "assembly_id": cif_column(["1", "2"]),
                "oper_expression": cif_column(["1", "(1,2)"]),
                "asym_id_list": cif_column([every_chain, chains[0].chain_id]),
            },
        )
        operations = {
            "id": ["1", "2"],
            "type": ["identity operation", "translation"],
            "matrix[1][1]": ["1", "1"], "matrix[1][2]": ["0", "0"], "matrix[1][3]": ["0", "0"],
            "matrix[2][1]": ["0", "0"], "matrix[2][2]": ["1", "1"], "matrix[2][3]": ["0", "0"],
            "matrix[3][1]": ["0", "0"], "matrix[3][2]": ["0", "0"], "matrix[3][3]": ["1", "1"],
            "vector[1]": ["0", "20"], "vector[2]": ["0", "0"], "vector[3]": ["0", "0"],
        }
        block["pdbx_struct_oper_list"] = CIFCategory(name="pdbx_struct_oper_list", columns={name: cif_column(values) for name, values in operations.items()})
    output = io.StringIO()
    cif.write(output)
    return output.getvalue()
