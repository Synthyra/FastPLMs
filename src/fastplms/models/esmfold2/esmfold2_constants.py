"""Declarative molecular schema for ESMFold2 feature preparation.

The package manifest owns the upstream revision and license provenance. This
module expresses the corresponding checkpoint-facing integer schema as compact
ordered records, then derives lookup tables from those records. The generated
tables are validated at import without reading files, downloading assets, or
mutating process state.
"""

from __future__ import annotations

SCHEMA_PROVENANCE = {
    "manifest_family": "esmfold2",
    "contract": "biohub_esmfold2_input_v1",
}


def _words(value: str) -> list[str]:
    return value.split()


MOL_TYPE_PROTEIN = 0
MOL_TYPE_DNA = 1
MOL_TYPE_RNA = 2
MOL_TYPE_NONPOLYMER = 3

# The record order is part of the checkpoint input contract. Residue indices
# start at two because zero and one are reserved by the model feature schema.
_PROTEIN_SCHEMA = tuple(
    tuple(record.split(":"))
    for record in (
        "ALA:A:N CA C O CB",
        "ARG:R:N CA C O CB CG CD NE CZ NH1 NH2",
        "ASN:N:N CA C O CB CG OD1 ND2",
        "ASP:D:N CA C O CB CG OD1 OD2",
        "CYS:C:N CA C O CB SG",
        "GLN:Q:N CA C O CB CG CD OE1 NE2",
        "GLU:E:N CA C O CB CG CD OE1 OE2",
        "GLY:G:N CA C O",
        "HIS:H:N CA C O CB CG ND1 CD2 CE1 NE2",
        "ILE:I:N CA C O CB CG1 CG2 CD1",
        "LEU:L:N CA C O CB CG CD1 CD2",
        "LYS:K:N CA C O CB CG CD CE NZ",
        "MET:M:N CA C O CB CG SD CE",
        "PHE:F:N CA C O CB CG CD1 CD2 CE1 CE2 CZ",
        "PRO:P:N CA C O CB CG CD",
        "SER:S:N CA C O CB OG",
        "THR:T:N CA C O CB OG1 CG2",
        "TRP:W:N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
        "TYR:Y:N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH",
        "VAL:V:N CA C O CB CG1 CG2",
    )
)

PROTEIN_RESIDUE_TO_RES_TYPE = {
    residue: index for index, (residue, _letter, _atoms) in enumerate(_PROTEIN_SCHEMA, 2)
}
PROTEIN_RESIDUE_TO_RES_TYPE["MSE"] = PROTEIN_RESIDUE_TO_RES_TYPE["MET"]
PROTEIN_UNK_RES_TYPE = 22

RNA_RESIDUE_TO_RES_TYPE = dict(zip("AGCU", range(23, 27), strict=True))
RNA_UNK_RES_TYPE = 27
DNA_RESIDUE_TO_RES_TYPE = dict(zip(("DA", "DG", "DC", "DT"), range(28, 32), strict=True))
DNA_UNK_RES_TYPE = 32
GAP_RES_TYPE = DNA_UNK_RES_TYPE

PROTEIN_3TO1 = {residue: letter for residue, letter, _atoms in _PROTEIN_SCHEMA}
PROTEIN_3TO1["MSE"] = "M"
PROTEIN_1TO3 = {letter: residue for residue, letter, _atoms in _PROTEIN_SCHEMA}
PROTEIN_1TO3["X"] = "UNK"
DNA_1TO3 = dict(zip("ATCG", ("DA", "DT", "DC", "DG"), strict=True))
RNA_1TO3 = {letter: letter for letter in "AUCG"}

_ESM_RESIDUE_ORDER = "LAGVSERTIDPKQNFYM HWC".replace(" ", "")
ESM_PROTEIN_VOCAB = {residue: token_id for token_id, residue in enumerate(_ESM_RESIDUE_ORDER, 4)}
ESM_PROTEIN_VOCAB["X"] = 3
DNA_RNA_LIGAND_INPUT_ID = 24
MSA_PAD_TOKEN_ID = 0
MSA_GAP_TOKEN_ID = 1

RES_TYPE_TO_CCD = {
    **{
        index: residue for residue, index in PROTEIN_RESIDUE_TO_RES_TYPE.items() if residue != "MSE"
    },
    22: "UNK",
    **dict(zip(range(23, 28), ("A", "G", "C", "U", "N"), strict=True)),
    **dict(zip(range(28, 33), ("DA", "DG", "DC", "DT", "DN"), strict=True)),
}

_CHARGE_SCHEMA = _words(
    "LYS:NZ:1 ARG:NH2:1 HIS:ND1:1 PO4:O2:-1 PO4:O3:-1 PO4:O4:-1 "
    "SO4:O3:-1 SO4:O4:-1 MG:MG:2 ZN:ZN:2 CA:CA:2 FE2:FE:2 MN:MN:2 "
    "CO:CO:2 NCO:CO:3 CU:CU:2 NI:NI:2 K:K:1 NA:NA:1 CD:CD:2 CL:CL:-1 "
    "ACT:OXT:-1 NAD:O2N:-1 NAD:N1N:1 NAP:O2N:-1 NAP:N1N:1 IMD:N3:1 "
    "SAM:SD:1 FE:FE:3 A1BH3:N3:1"
)
CHARGED_ATOMS = {
    (component, atom): int(charge)
    for component, atom, charge in (record.split(":") for record in _CHARGE_SCHEMA)
}

_PERIODIC_SYMBOLS = _words(
    "H HE LI BE B C N O F NE NA MG AL SI P S CL AR K CA SC TI V CR MN FE CO NI CU ZN "
    "GA GE AS SE BR KR RB SR Y ZR NB MO TC RU RH PD AG CD IN SN SB TE I XE CS BA LA CE "
    "PR ND PM SM EU GD TB DY HO ER TM YB LU HF TA W RE OS IR PT AU HG TL PB BI PO AT RN "
    "FR RA AC TH PA U"
)
ELEMENT_TO_ATOMIC_NUM = {
    symbol: atomic_number
    for atomic_number, symbol in enumerate(_PERIODIC_SYMBOLS, 1)
    if symbol != "HE"
}
ELEMENT_NUMBER_TO_SYMBOL = {
    atomic_number: symbol for symbol, atomic_number in ELEMENT_TO_ATOMIC_NUM.items()
}

PROTEIN_HEAVY_ATOMS = {
    residue: atom_string.split() for residue, _letter, atom_string in _PROTEIN_SCHEMA
}
PROTEIN_HEAVY_ATOMS["MSE"] = PROTEIN_HEAVY_ATOMS["MET"].copy()
PROTEIN_HEAVY_ATOMS["UNK"] = _words("N CA C O")

DNA_BACKBONE_ATOMS = _words("P OP1 OP2 O5' C5' C4' O4' C3' O3' C2' C1'")
RNA_BACKBONE_ATOMS = _words("P OP1 OP2 O5' C5' C4' O4' C3' O3' C2' O2' C1'")
_NUCLEOBASE_ATOMS = {
    "A": _words("N9 C8 N7 C5 C6 N6 N1 C2 N3 C4"),
    "G": _words("N9 C8 N7 C5 C6 O6 N1 C2 N2 N3 C4"),
    "C": _words("N1 C2 O2 N3 C4 N4 C5 C6"),
    "U": _words("N1 C2 O2 N3 C4 O4 C5 C6"),
    "T": _words("N1 C2 O2 N3 C4 O4 C5 C7 C6"),
}
DNA_HEAVY_ATOMS = {
    "DA": DNA_BACKBONE_ATOMS + _NUCLEOBASE_ATOMS["A"],
    "DG": DNA_BACKBONE_ATOMS + _NUCLEOBASE_ATOMS["G"],
    "DC": DNA_BACKBONE_ATOMS + _NUCLEOBASE_ATOMS["C"],
    "DT": DNA_BACKBONE_ATOMS + _NUCLEOBASE_ATOMS["T"],
}
RNA_HEAVY_ATOMS = {residue: RNA_BACKBONE_ATOMS + _NUCLEOBASE_ATOMS[residue] for residue in "AGCU"}


def _validate_schema() -> None:
    if sorted(set(PROTEIN_RESIDUE_TO_RES_TYPE.values())) != list(range(2, 22)):
        raise RuntimeError("Protein residue indices must cover the checkpoint interval 2..21.")
    if len(_ESM_RESIDUE_ORDER) != 20 or len(set(_ESM_RESIDUE_ORDER)) != 20:
        raise RuntimeError("The ESM residue vocabulary must contain 20 canonical residues.")
    if RES_TYPE_TO_CCD[14] != "MET" or PROTEIN_RESIDUE_TO_RES_TYPE["MSE"] != 14:
        raise RuntimeError("Selenomethionine must share the methionine residue index.")
    if ELEMENT_TO_ATOMIC_NUM.get("U") != 92 or 2 in ELEMENT_NUMBER_TO_SYMBOL:
        raise RuntimeError("The element schema must preserve the training-time atomic-number map.")
    if set(DNA_HEAVY_ATOMS) != {"DA", "DG", "DC", "DT"}:
        raise RuntimeError("The DNA atom schema is incomplete.")


_validate_schema()

__all__ = [name for name in globals() if name.isupper()]
