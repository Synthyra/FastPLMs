"""Minimal biological constants required by the local Boltz2 runtime.

The upstream project contains additional training-time curation tables.  They
are intentionally excluded here because FastPLMs neither trains Boltz2 nor
uses upstream data-processing code.  Keeping only runtime inputs makes this
module auditable and prevents parity-oracle data from becoming a package
dependency.
"""

from __future__ import annotations


def _index(values: list[str]) -> dict[str, int]:
    return {value: index for index, value in enumerate(values)}


chain_types = ["PROTEIN", "DNA", "RNA", "NONPOLYMER"]
chain_type_ids = _index(chain_types)


canonical_tokens = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
]
tokens = [
    "<pad>",
    "-",
    *canonical_tokens,
    "A",
    "G",
    "C",
    "U",
    "N",
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",
]
token_ids = _index(tokens)
num_tokens = len(tokens)

prot_letter_to_token = dict(
    zip(
        "ARNDCEQGHILKMFPSTWYV",
        [
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLU",
            "GLN",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ],
        strict=True,
    )
)
prot_letter_to_token.update({letter: "UNK" for letter in "XJBZOU"} | {"-": "-"})


def _parse_atom_rows(rows: str) -> dict[str, list[str]]:
    table: dict[str, list[str]] = {}
    for row in rows.strip().splitlines():
        residue, *atom_names = row.split()
        table[residue] = atom_names
    return table


ref_atoms = {"PAD": [], "-": []}
ref_atoms.update(
    _parse_atom_rows(
        """
UNK N CA C O CB
ALA N CA C O CB
ARG N CA C O CB CG CD NE CZ NH1 NH2
ASN N CA C O CB CG OD1 ND2
ASP N CA C O CB CG OD1 OD2
CYS N CA C O CB SG
GLN N CA C O CB CG CD OE1 NE2
GLU N CA C O CB CG CD OE1 OE2
GLY N CA C O
HIS N CA C O CB CG ND1 CD2 CE1 NE2
ILE N CA C O CB CG1 CG2 CD1
LEU N CA C O CB CG CD1 CD2
LYS N CA C O CB CG CD CE NZ
MET N CA C O CB CG SD CE
PHE N CA C O CB CG CD1 CD2 CE1 CE2 CZ
PRO N CA C O CB CG CD
SER N CA C O CB OG
THR N CA C O CB OG1 CG2
TRP N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2
TYR N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH
VAL N CA C O CB CG1 CG2
"""
    )
)

protein_backbone_atom_names = ["N", "CA", "C", "O"]
nucleic_backbone_atom_names = [
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
]
protein_backbone_atom_index = _index(protein_backbone_atom_names)
nucleic_backbone_atom_index = _index(nucleic_backbone_atom_names)

_rna_backbone = nucleic_backbone_atom_names
_dna_backbone = [atom for atom in _rna_backbone if atom != "O2'"]
ref_atoms.update(
    {
        "A": [*_rna_backbone, "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
        "G": [*_rna_backbone, "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
        "C": [*_rna_backbone, "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
        "U": [*_rna_backbone, "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
        "N": list(_rna_backbone),
        "DA": [*_dna_backbone, "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
        "DG": [*_dna_backbone, "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
        "DC": [*_dna_backbone, "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
        "DT": [*_dna_backbone, "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"],
        "DN": list(_dna_backbone),
    }
)

_protein_tokens = ["UNK", *canonical_tokens[:-1]]
_nucleic_tokens = ["A", "G", "C", "U", "N", "DA", "DG", "DC", "DT", "DN"]
res_to_center_atom = {
    **dict.fromkeys(_protein_tokens, "CA"),
    **dict.fromkeys(_nucleic_tokens, "C1'"),
}
res_to_disto_atom = {
    **dict.fromkeys(_protein_tokens, "CB"),
    "GLY": "CA",
    "A": "C4",
    "G": "C4",
    "C": "C2",
    "U": "C2",
    "N": "C1'",
    "DA": "C4",
    "DG": "C4",
    "DC": "C2",
    "DT": "C2",
    "DN": "C1'",
}


num_elements = 128
bond_types = ["OTHER", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "COVALENT"]
contact_conditioning_info = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET>BINDER": 2,
    "BINDER>POCKET": 3,
    "CONTACT": 4,
}
chunk_size_threshold = 384

_method_groups = {
    0: ("MD",),
    1: ("X-RAY DIFFRACTION",),
    2: ("ELECTRON MICROSCOPY",),
    3: ("SOLUTION NMR",),
    4: (
        "SOLID-STATE NMR",
        "NEUTRON DIFFRACTION",
        "ELECTRON CRYSTALLOGRAPHY",
        "FIBER DIFFRACTION",
        "POWDER DIFFRACTION",
        "INFRARED SPECTROSCOPY",
        "FLUORESCENCE TRANSFER",
        "EPR",
        "THEORETICAL MODEL",
        "SOLUTION SCATTERING",
        "OTHER",
    ),
    5: ("AFDB",),
    6: ("BOLTZ-1",),
    7: ("FUTURE1",),
    8: ("FUTURE2",),
    9: ("FUTURE3",),
    10: ("FUTURE4",),
    11: ("FUTURE5",),
}
method_types_ids = {
    method.lower(): identifier
    for identifier, methods in _method_groups.items()
    for method in methods
}
num_method_types = len(_method_groups)

vdw_radii = [
    float(radius)
    for radius in [
        "1.2",
        "1.4",
        "2.2",
        "1.9",
        "1.8",
        "1.7",
        "1.6",
        "1.55",
        "1.5",
        "1.54",
        "2.4",
        "2.2",
        "2.1",
        "2.1",
        "1.95",
        "1.8",
        "1.8",
        "1.88",
        "2.8",
        "2.4",
        "2.3",
        "2.15",
        "2.05",
        "2.05",
        "2.05",
        "2.05",
        "2.0",
        "2.0",
        "2.0",
        "2.1",
        "2.1",
        "2.1",
        "2.05",
        "1.9",
        "1.9",
        "2.02",
        "2.9",
        "2.55",
        "2.4",
        "2.3",
        "2.15",
        "2.1",
        "2.05",
        "2.05",
        "2.0",
        "2.05",
        "2.1",
        "2.2",
        "2.2",
        "2.25",
        "2.2",
        "2.1",
        "2.1",
        "2.16",
        "3.0",
        "2.7",
        "2.5",
        "2.48",
        "2.47",
        "2.45",
        "2.43",
        "2.42",
        "2.4",
        "2.38",
        "2.37",
        "2.35",
        "2.33",
        "2.32",
        "2.3",
        "2.28",
        "2.27",
        "2.25",
        "2.2",
        "2.1",
        "2.05",
        "2.0",
        "2.0",
        "2.05",
        "2.1",
        "2.05",
        "2.2",
        "2.3",
        "2.3",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.4",
        "2.0",
        "2.3",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
        "2.0",
    ]
]
