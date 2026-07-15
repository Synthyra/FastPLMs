"""Translate typed sequence inputs into the tensors consumed by ESMFold2.

The conversion has four explicit stages: entity and chain assignment, residue
tokenization, structural feature construction, and atom-table padding.  Keeping
those stages separate makes the biological indexing rules testable without
loading model weights.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import torch

from .esmfold2_conformers import (
    get_ccd_leaving_atoms,
    get_idealized_atom_pos,
    get_ligand_ccd_atoms_with_charges,
    get_ligand_ccd_bonds,
    get_ligand_idealized_atom_pos,
)
from .esmfold2_constants import (
    CHARGED_ATOMS,
    DNA_1TO3,
    DNA_BACKBONE_ATOMS,
    DNA_HEAVY_ATOMS,
    DNA_RESIDUE_TO_RES_TYPE,
    DNA_RNA_LIGAND_INPUT_ID,
    DNA_UNK_RES_TYPE,
    ELEMENT_TO_ATOMIC_NUM,
    ESM_PROTEIN_VOCAB,
    MOL_TYPE_DNA,
    MOL_TYPE_NONPOLYMER,
    MOL_TYPE_PROTEIN,
    MOL_TYPE_RNA,
    MSA_GAP_TOKEN_ID,
    PROTEIN_1TO3,
    PROTEIN_3TO1,
    PROTEIN_HEAVY_ATOMS,
    PROTEIN_RESIDUE_TO_RES_TYPE,
    PROTEIN_UNK_RES_TYPE,
    RNA_1TO3,
    RNA_BACKBONE_ATOMS,
    RNA_HEAVY_ATOMS,
    RNA_RESIDUE_TO_RES_TYPE,
    RNA_UNK_RES_TYPE,
)
from .esmfold2_types import (
    MSA,
    DNAInput,
    LigandInput,
    Modification,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)

_ZERO_POS = np.zeros(3, dtype=np.float32)
_ENCODE_ATOM_NAME_CACHE: dict[str, list[int]] = {}
_ELEMENT_ATOMIC_NUM_CACHE: dict[str, int] = {}
_TWO_LETTER_ELEMENTS = frozenset({"FE", "ZN", "MG", "MN", "CO", "NI", "CU", "SE", "BR"})


@dataclass
class AtomInfo:
    """One row in the unpadded atom table."""

    name: str
    element: str
    charge: int
    ref_pos: np.ndarray  # R has shape (3,).
    pos: np.ndarray  # X has shape (3,).
    token_index: int = -1
    atom_index: int = -1
    space_uid: int = -1
    is_valid: bool = True


@dataclass
class TokenInfo:
    """Biological and atom-span annotations for one model token."""

    token_index: int
    residue_index: int
    residue_name: str
    mol_type: int
    res_type: int
    input_id: int
    asym_id: int
    sym_id: int
    entity_id: int
    atom_start: int
    atom_count: int


@dataclass
class ChainInfo:
    """One input chain after entity and symmetry assignment."""

    chain_id: str
    asym_id: int
    entity_id: int
    sym_id: int
    mol_type: int
    tokens: list[TokenInfo] = field(default_factory=list)
    ligand_bonds: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class _TokenizationState:
    """Mutable cursor shared by residue tokenizers."""

    token_index: int
    atom_index: int
    space_uid: int
    tokens: list[TokenInfo] = field(default_factory=list)
    atoms: list[AtomInfo] = field(default_factory=list)

    def _append_atom(
        self,
        name: str,
        element: str,
        charge: int,
        ref_pos: np.ndarray | None,
    ) -> None:
        self.atoms.append(
            AtomInfo(
                name=name,
                element=element,
                charge=charge,
                ref_pos=(ref_pos.copy() if ref_pos is not None else _ZERO_POS.copy()),
                pos=_ZERO_POS.copy(),
                token_index=self.token_index,
                atom_index=self.atom_index,
                space_uid=self.space_uid,
            )
        )
        self.atom_index += 1

    def _append_token(
        self,
        *,
        residue_index: int,
        residue_name: str,
        mol_type: int,
        res_type: int,
        input_id: int,
        asym_id: int,
        sym_id: int,
        entity_id: int,
        atom_start: int,
        atom_count: int,
    ) -> None:
        self.tokens.append(
            TokenInfo(
                token_index=self.token_index,
                residue_index=residue_index,
                residue_name=residue_name,
                mol_type=mol_type,
                res_type=res_type,
                input_id=input_id,
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                atom_start=atom_start,
                atom_count=atom_count,
            )
        )
        self.token_index += 1

    def add_residue_token(
        self,
        atom_specs: list[tuple[str, str, int, np.ndarray | None]],
        **token_fields: Any,
    ) -> None:
        start = self.atom_index
        for atom_spec in atom_specs:
            self._append_atom(*atom_spec)
        self._append_token(
            atom_start=start,
            atom_count=len(atom_specs),
            **token_fields,
        )
        self.space_uid += 1

    def add_atom_tokens(
        self,
        atom_specs: list[tuple[str, str, int, np.ndarray | None]],
        **token_fields: Any,
    ) -> None:
        for atom_spec in atom_specs:
            start = self.atom_index
            self._append_atom(*atom_spec)
            self._append_token(atom_start=start, atom_count=1, **token_fields)
        self.space_uid += 1


def encode_atom_name(name: str) -> list[int]:
    """Encode a four-character atom name with the model's ASCII offset."""
    cached = _ENCODE_ATOM_NAME_CACHE.get(name)
    if cached is None:
        cached = [0 if char == " " else ord(char) - 32 for char in name.ljust(4)[:4]]
        _ENCODE_ATOM_NAME_CACHE[name] = cached
    return cached


def get_element_atomic_num(element: str) -> int:
    """Map an element symbol to the model's atomic-number vocabulary."""
    cached = _ELEMENT_ATOMIC_NUM_CACHE.get(element)
    if cached is None:
        cached = ELEMENT_TO_ATOMIC_NUM.get(element.upper(), 0)
        _ELEMENT_ATOMIC_NUM_CACHE[element] = cached
    return cached


def _infer_element(atom_name: str) -> str:
    normalized = atom_name.strip()
    if not normalized:
        return "C"
    if normalized[0].isdigit():
        return normalized[1] if len(normalized) > 1 else "H"
    if len(normalized) == 2 and normalized in _TWO_LETTER_ELEMENTS:
        return normalized
    return normalized[0]


def _compute_res_type(name: str, mol_type: int) -> int:
    if mol_type == MOL_TYPE_PROTEIN:
        return PROTEIN_RESIDUE_TO_RES_TYPE.get(name, PROTEIN_UNK_RES_TYPE)
    if mol_type == MOL_TYPE_DNA:
        return DNA_RESIDUE_TO_RES_TYPE.get(
            name, RNA_RESIDUE_TO_RES_TYPE.get(name, DNA_UNK_RES_TYPE)
        )
    if mol_type == MOL_TYPE_RNA:
        return RNA_RESIDUE_TO_RES_TYPE.get(
            name, DNA_RESIDUE_TO_RES_TYPE.get(name, RNA_UNK_RES_TYPE)
        )
    return PROTEIN_UNK_RES_TYPE


def _compute_esm_input_id(name: str, mol_type: int) -> int:
    if mol_type != MOL_TYPE_PROTEIN:
        return DNA_RNA_LIGAND_INPUT_ID
    letter = PROTEIN_3TO1.get(name)
    return (
        DNA_RNA_LIGAND_INPUT_ID
        if letter is None
        else ESM_PROTEIN_VOCAB.get(letter, ESM_PROTEIN_VOCAB["X"])
    )


def _apply_modifications(residues: list[str], modifications: list[Modification] | None) -> set[int]:
    changed: set[int] = set()
    for modification in modifications or ():
        residues[modification.position] = modification.ccd
        changed.add(modification.position)
    return changed


def _ideal_atom_specs(
    residue_name: str,
    residue_type: int,
    atom_names: list[str],
    *,
    charges: bool = True,
) -> list[tuple[str, str, int, np.ndarray | None]]:
    return [
        (
            atom_name,
            _infer_element(atom_name),
            CHARGED_ATOMS.get((residue_name, atom_name), 0) if charges else 0,
            get_idealized_atom_pos(residue_type, atom_name),
        )
        for atom_name in atom_names
    ]


def _ccd_atom_specs(
    residue_name: str,
    atoms: list[tuple[str, str, int]],
    excluded: set[str],
    *,
    force_zero: bool = False,
) -> list[tuple[str, str, int, np.ndarray | None]]:
    return [
        (
            atom_name,
            element,
            charge,
            None if force_zero else get_ligand_idealized_atom_pos(residue_name, atom_name),
        )
        for atom_name, element, charge in atoms
        if atom_name not in excluded
    ]


def tokenize_protein(
    sequence: str,
    modifications: list[Modification] | None,
    entity_id: int,
    asym_id: int,
    sym_id: int,
    token_offset: int,
    atom_offset: int,
    space_uid_offset: int,
) -> tuple[list[TokenInfo], list[AtomInfo]]:
    """Tokenize protein residues, atom-tokenizing modified CCD components."""
    residues = [PROTEIN_1TO3.get(letter, "UNK") for letter in sequence]
    modified = _apply_modifications(residues, modifications)
    state = _TokenizationState(token_offset, atom_offset, space_uid_offset)

    for residue_index, residue_name in enumerate(residues):
        canonical_name = "MET" if residue_name == "MSE" else residue_name
        common_fields = {
            "residue_index": residue_index,
            "mol_type": MOL_TYPE_PROTEIN,
            "asym_id": asym_id,
            "sym_id": sym_id,
            "entity_id": entity_id,
        }
        if residue_index not in modified and canonical_name in PROTEIN_HEAVY_ATOMS:
            residue_type = _compute_res_type(canonical_name, MOL_TYPE_PROTEIN)
            state.add_residue_token(
                _ideal_atom_specs(
                    canonical_name,
                    residue_type,
                    PROTEIN_HEAVY_ATOMS[canonical_name],
                ),
                residue_name=canonical_name,
                res_type=residue_type,
                input_id=_compute_esm_input_id(canonical_name, MOL_TYPE_PROTEIN),
                **common_fields,
            )
            continue

        ccd_atoms = get_ligand_ccd_atoms_with_charges(residue_name)
        if ccd_atoms is None:
            ccd_atoms = [
                (_infer_element(name), _infer_element(name), 0) for name in ("N", "CA", "C", "O")
            ]
        excluded = (
            set() if residue_index == len(residues) - 1 else get_ccd_leaving_atoms(residue_name)
        )
        retained = [atom for atom in ccd_atoms if atom[0] not in excluded]
        state.add_atom_tokens(
            _ccd_atom_specs(
                residue_name,
                retained,
                set(),
                force_zero=len(retained) == 1,
            ),
            residue_name=residue_name,
            res_type=PROTEIN_UNK_RES_TYPE,
            input_id=DNA_RNA_LIGAND_INPUT_ID,
            **common_fields,
        )
    return state.tokens, state.atoms


def tokenize_nucleotide(
    sequence: str,
    modifications: list[Modification] | None,
    mol_type: int,
    entity_id: int,
    asym_id: int,
    sym_id: int,
    token_offset: int,
    atom_offset: int,
    space_uid_offset: int,
) -> tuple[list[TokenInfo], list[AtomInfo]]:
    """Tokenize DNA or RNA, retaining backbone atoms for unknown bases."""
    dna = mol_type == MOL_TYPE_DNA
    letter_map = DNA_1TO3 if dna else RNA_1TO3
    heavy_atoms = DNA_HEAVY_ATOMS if dna else RNA_HEAVY_ATOMS
    backbone_atoms = DNA_BACKBONE_ATOMS if dna else RNA_BACKBONE_ATOMS
    unknown_type = DNA_UNK_RES_TYPE if dna else RNA_UNK_RES_TYPE
    residues = [letter_map.get(letter, "UNK") for letter in sequence]
    modified = _apply_modifications(residues, modifications)
    state = _TokenizationState(token_offset, atom_offset, space_uid_offset)

    for residue_index, residue_name in enumerate(residues):
        common_fields = {
            "residue_index": residue_index,
            "residue_name": residue_name,
            "mol_type": mol_type,
            "asym_id": asym_id,
            "sym_id": sym_id,
            "entity_id": entity_id,
            "input_id": DNA_RNA_LIGAND_INPUT_ID,
        }
        if residue_index not in modified and residue_name in heavy_atoms:
            residue_type = _compute_res_type(residue_name, mol_type)
            state.add_residue_token(
                _ideal_atom_specs(residue_name, residue_type, heavy_atoms[residue_name]),
                res_type=residue_type,
                **common_fields,
            )
            continue
        if residue_index not in modified and residue_name == "UNK":
            state.add_residue_token(
                [(atom_name, _infer_element(atom_name), 0, None) for atom_name in backbone_atoms],
                res_type=unknown_type,
                **common_fields,
            )
            continue

        ccd_atoms = get_ligand_ccd_atoms_with_charges(residue_name)
        if ccd_atoms is None:
            ccd_atoms = [(_infer_element(name), _infer_element(name), 0) for name in backbone_atoms]
        excluded = (
            set() if residue_index == len(residues) - 1 else get_ccd_leaving_atoms(residue_name)
        )
        state.add_atom_tokens(
            _ccd_atom_specs(residue_name, ccd_atoms, excluded),
            res_type=PROTEIN_UNK_RES_TYPE,
            **common_fields,
        )
    return state.tokens, state.atoms


def tokenize_ligand_ccd(
    ccd_codes: list[str],
    entity_id: int,
    asym_id: int,
    sym_id: int,
    token_offset: int,
    atom_offset: int,
    space_uid_offset: int,
    has_covalent_bond: bool,
) -> tuple[list[TokenInfo], list[AtomInfo]]:
    """Tokenize CCD ligands with one model token per retained atom."""
    state = _TokenizationState(token_offset, atom_offset, space_uid_offset)
    for residue_index, code in enumerate(ccd_codes):
        ccd_atoms = get_ligand_ccd_atoms_with_charges(code)
        if ccd_atoms is None:
            raise ValueError(f"CCD component {code} not found")
        excluded = get_ccd_leaving_atoms(code) if has_covalent_bond else set()
        state.add_atom_tokens(
            _ccd_atom_specs(code, ccd_atoms, excluded),
            residue_index=residue_index,
            residue_name=code,
            mol_type=MOL_TYPE_NONPOLYMER,
            res_type=PROTEIN_UNK_RES_TYPE,
            input_id=DNA_RNA_LIGAND_INPUT_ID,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
        )
    return state.tokens, state.atoms


def tokenize_ligand_smiles(
    smiles: str,
    entity_id: int,
    asym_id: int,
    sym_id: int,
    token_offset: int,
    atom_offset: int,
    space_uid_offset: int,
    seed: int | None = None,
) -> tuple[list[TokenInfo], list[AtomInfo], list[tuple[str, str]]]:
    """Generate a conformer and tokenize each heavy atom of a SMILES ligand."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    molecule = Chem.AddHs(molecule)
    canonical_order = AllChem.CanonicalRankAtoms(molecule)  # type: ignore[attr-defined]
    for atom, canonical_index in zip(molecule.GetAtoms(), canonical_order, strict=True):
        name = atom.GetSymbol().upper() + str(canonical_index + 1)
        if len(name) > 4:
            raise ValueError(f"SMILES {smiles} has atom name longer than 4 chars: {name}")
        atom.SetProp("name", name)

    options = AllChem.ETKDGv3()  # type: ignore[attr-defined]
    options.clearConfs = False
    if seed is not None:
        options.randomSeed = seed
    conformer_id = AllChem.EmbedMolecule(molecule, options)  # type: ignore[attr-defined]
    if conformer_id == -1:
        options.useRandomCoords = True
        conformer_id = AllChem.EmbedMolecule(molecule, options)  # type: ignore[attr-defined]
    if conformer_id != -1:
        with suppress(RuntimeError, ValueError):
            AllChem.UFFOptimizeMolecule(  # type: ignore[attr-defined]
                molecule, confId=conformer_id, maxIters=1000
            )

    molecule = Chem.RemoveHs(molecule)
    if molecule.GetNumConformers() == 0:
        raise ValueError(f"Failed to generate conformer for SMILES: {smiles}")
    conformer = molecule.GetConformer(0)
    atom_specs: list[tuple[str, str, int, np.ndarray | None]] = []
    for atom in molecule.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        atom_specs.append(
            (
                atom.GetProp("name"),
                atom.GetSymbol(),
                atom.GetFormalCharge(),
                np.asarray([position.x, position.y, position.z], dtype=np.float32),
            )
        )
    state = _TokenizationState(token_offset, atom_offset, space_uid_offset)
    state.add_atom_tokens(
        atom_specs,
        residue_index=0,
        residue_name="LIG",
        mol_type=MOL_TYPE_NONPOLYMER,
        res_type=PROTEIN_UNK_RES_TYPE,
        input_id=DNA_RNA_LIGAND_INPUT_ID,
        asym_id=asym_id,
        sym_id=sym_id,
        entity_id=entity_id,
    )
    bonds = [
        (
            bond.GetBeginAtom().GetProp("name"),
            bond.GetEndAtom().GetProp("name"),
        )
        for bond in molecule.GetBonds()
    ]
    return state.tokens, state.atoms, bonds


def _get_sequence_key(item: Any) -> str:
    if isinstance(item, ProteinInput):
        return f"PROTEIN:{item.sequence}"
    if isinstance(item, DNAInput):
        return f"DNA:{item.sequence}"
    if isinstance(item, RNAInput):
        return f"RNA:{item.sequence}"
    if isinstance(item, LigandInput):
        return f"LIGAND_CCD:{','.join(item.ccd)}" if item.ccd else f"LIGAND_SMILES:{item.smiles}"
    raise ValueError(f"Unknown input type: {type(item)}")


def _tokenize_chain(
    item: Any,
    chain_id: str,
    *,
    entity_id: int,
    asym_id: int,
    sym_id: int,
    token_offset: int,
    atom_offset: int,
    space_uid_offset: int,
    covalent_chains: set[str],
    seed: int | None,
) -> tuple[list[TokenInfo], list[AtomInfo], list[tuple[str, str]]]:
    common = {
        "entity_id": entity_id,
        "asym_id": asym_id,
        "sym_id": sym_id,
        "token_offset": token_offset,
        "atom_offset": atom_offset,
        "space_uid_offset": space_uid_offset,
    }
    if isinstance(item, ProteinInput):
        if item.msa is None:
            warnings.warn(
                f"No MSA provided for {item.id}, using single sequence mode",
                stacklevel=2,
            )
        tokens, atoms = tokenize_protein(item.sequence, item.modifications, **common)
        return tokens, atoms, []
    if isinstance(item, (DNAInput, RNAInput)):
        mol_type = MOL_TYPE_DNA if isinstance(item, DNAInput) else MOL_TYPE_RNA
        tokens, atoms = tokenize_nucleotide(
            item.sequence, item.modifications, mol_type=mol_type, **common
        )
        return tokens, atoms, []
    if not isinstance(item, LigandInput):
        raise ValueError(f"Unknown input type: {type(item)}")
    if item.ccd is not None:
        if item.smiles is not None:
            warnings.warn("Both ccd and smiles provided, using ccd", stacklevel=2)
        tokens, atoms = tokenize_ligand_ccd(
            item.ccd,
            has_covalent_bond=chain_id in covalent_chains,
            **common,
        )
        return tokens, atoms, []
    if item.smiles is not None:
        return tokenize_ligand_smiles(item.smiles, seed=seed, **common)
    raise ValueError("LigandInput must have either ccd or smiles")


def build_chains_from_input(
    input: StructurePredictionInput, seed: int | None = None
) -> tuple[list[ChainInfo], list[TokenInfo], list[AtomInfo]]:
    """Assign entities and symmetry copies, then tokenize every input chain."""
    chains: list[ChainInfo] = []
    tokens: list[TokenInfo] = []
    atoms: list[AtomInfo] = []
    entity_for_sequence: dict[str, int] = {}
    next_symmetry: dict[int, int] = {}
    covalent_chains = {
        chain_id
        for bond in input.covalent_bonds or ()
        for chain_id in (bond.chain_id1, bond.chain_id2)
    }
    space_uid_offset = 0

    for item in input.sequences:
        key = _get_sequence_key(item)
        entity_id = entity_for_sequence.setdefault(key, len(entity_for_sequence))
        chain_ids = [item.id] if isinstance(item.id, str) else item.id
        for chain_id in chain_ids:
            sym_id = next_symmetry.get(entity_id, 0)
            next_symmetry[entity_id] = sym_id + 1
            asym_id = len(chains)
            new_tokens, new_atoms, ligand_bonds = _tokenize_chain(
                item,
                chain_id,
                entity_id=entity_id,
                asym_id=asym_id,
                sym_id=sym_id,
                token_offset=len(tokens),
                atom_offset=len(atoms),
                space_uid_offset=space_uid_offset,
                covalent_chains=covalent_chains,
                seed=seed,
            )
            chains.append(
                ChainInfo(
                    chain_id=chain_id,
                    asym_id=asym_id,
                    entity_id=entity_id,
                    sym_id=sym_id,
                    mol_type=(new_tokens[0].mol_type if new_tokens else MOL_TYPE_PROTEIN),
                    tokens=new_tokens,
                    ligand_bonds=ligand_bonds,
                )
            )
            tokens.extend(new_tokens)
            atoms.extend(new_atoms)
            space_uid_offset += len({atom.space_uid for atom in new_atoms})
    return chains, tokens, atoms


def _atom_indices_by_name(atoms: list[AtomInfo]) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = defaultdict(dict)
    for atom in atoms:
        if atom.is_valid:
            result[atom.token_index][atom.name] = atom.atom_index
    return result


def _ligand_frames(
    tokens: list[TokenInfo],
    atoms: list[AtomInfo],
    atom_indices: dict[int, dict[str, int]],
) -> dict[int, tuple[int, int, int]]:
    atom_for_token: dict[int, int] = {}
    tokens_by_residue: dict[tuple[int, int], list[int]] = defaultdict(list)
    for token in tokens:
        if token.mol_type != MOL_TYPE_NONPOLYMER:
            continue
        named_atoms = atom_indices.get(token.token_index)
        if named_atoms:
            atom_for_token[token.token_index] = next(iter(named_atoms.values()))
        tokens_by_residue[(token.asym_id, token.residue_index)].append(token.token_index)

    frames: dict[int, tuple[int, int, int]] = {}
    for residue_tokens in tokens_by_residue.values():
        residue_atoms = [
            atom_for_token[token] for token in residue_tokens if token in atom_for_token
        ]
        if len(residue_atoms) < 3:
            for token in residue_tokens:
                if token in atom_for_token:
                    atom_index = atom_for_token[token]
                    frames[token] = (atom_index, atom_index, atom_index)
            continue
        R = np.asarray([atoms[index].ref_pos for index in residue_atoms])
        distances = np.sqrt(((R[:, None] - R[None]) ** 2).sum(-1))
        nearest = np.argsort(distances, axis=1)
        local = np.column_stack((nearest[:, 1], nearest[:, 0], nearest[:, 2]))
        local_index = {atom_index: index for index, atom_index in enumerate(residue_atoms)}
        for token in residue_tokens:
            atom_index = atom_for_token.get(token)
            if atom_index is None:
                continue
            selected = local[local_index[atom_index]]
            frames[token] = tuple(residue_atoms[int(index)] for index in selected)
    return frames


def _frame_for_token(
    token: TokenInfo,
    named_atoms: dict[str, int],
    ligand_frames: dict[int, tuple[int, int, int]],
) -> tuple[int, int, int]:
    fallback = next(iter(named_atoms.values()), 0)
    if token.mol_type == MOL_TYPE_PROTEIN:
        return (
            (fallback, fallback, fallback)
            if token.res_type == PROTEIN_UNK_RES_TYPE
            else (
                named_atoms.get("N", 0),
                named_atoms.get("CA", 0),
                named_atoms.get("C", 0),
            )
        )
    if token.mol_type in (MOL_TYPE_DNA, MOL_TYPE_RNA):
        return (
            (fallback, fallback, fallback)
            if token.res_type == PROTEIN_UNK_RES_TYPE
            else (
                named_atoms.get("C1'", 0),
                named_atoms.get("C3'", 0),
                named_atoms.get("C4'", 0),
            )
        )
    if token.mol_type == MOL_TYPE_NONPOLYMER:
        return ligand_frames.get(token.token_index, (fallback, fallback, fallback))
    return fallback, fallback, fallback


def _resolved_frames(
    frames: np.ndarray, tokens: list[TokenInfo], atoms: list[AtomInfo]
) -> np.ndarray:
    if not tokens:
        return np.zeros(0, dtype=bool)
    X = (
        np.asarray([atom.pos for atom in atoms], dtype=np.float32)
        if atoms
        else np.zeros((0, 3), dtype=np.float32)
    )
    valid_atoms = (
        np.asarray([atom.is_valid for atom in atoms], dtype=bool)
        if atoms
        else np.zeros(0, dtype=bool)
    )
    resolved_atoms = valid_atoms & np.any(X != 0, axis=1)
    origin = X[frames[:, 1]]
    left = X[frames[:, 0]] - origin
    right = X[frames[:, 2]] - origin
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    valid_norms = (left_norm >= 1e-6) & (right_norm >= 1e-6)
    cosine = np.zeros(len(tokens), dtype=np.float32)
    if np.any(valid_norms):
        cosine[valid_norms] = np.sum(left[valid_norms] * right[valid_norms], axis=1) / (
            left_norm[valid_norms] * right_norm[valid_norms]
        )
    angle = np.degrees(np.arccos(np.abs(np.clip(cosine, -1, 1))))
    all_resolved = resolved_atoms[frames].all(axis=1)
    repeated = (frames[:, 0] == frames[:, 1]) & (frames[:, 1] == frames[:, 2])
    return all_resolved & ~repeated & valid_norms & (angle >= 25)


def compute_frame_indices(
    tokens: list[TokenInfo], atoms: list[AtomInfo]
) -> tuple[np.ndarray, np.ndarray]:
    """Return frame atom indices F with shape (l, 3) and validity M with shape (l,)."""
    named_atoms = _atom_indices_by_name(atoms)
    ligand_frames = _ligand_frames(tokens, atoms, named_atoms)
    frames = np.asarray(
        [
            _frame_for_token(token, named_atoms.get(token.token_index, {}), ligand_frames)
            for token in tokens
        ],
        dtype=np.int64,
    )
    return frames, _resolved_frames(frames, tokens, atoms)


def _atom_tokenized_residues(
    tokens: list[TokenInfo], atoms: list[AtomInfo]
) -> dict[tuple[int, int], list[tuple[str, int]]]:
    grouped: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    for atom in atoms:
        if not atom.is_valid or atom.token_index >= len(tokens):
            continue
        token = tokens[atom.token_index]
        if token.mol_type == MOL_TYPE_NONPOLYMER or token.res_type == PROTEIN_UNK_RES_TYPE:
            grouped[(token.asym_id, token.residue_index)].append((atom.name, atom.token_index))
    return grouped


def _backbone_token(
    residue_tokens: list[TokenInfo], atom_name: str, atoms: list[AtomInfo]
) -> int | None:
    if len(residue_tokens) == 1 and residue_tokens[0].res_type != PROTEIN_UNK_RES_TYPE:
        return residue_tokens[0].token_index
    for token in residue_tokens:
        for atom_index in range(token.atom_start, token.atom_start + token.atom_count):
            if atom_index < len(atoms) and atoms[atom_index].name == atom_name:
                return token.token_index
    return residue_tokens[0].token_index if residue_tokens else None


def compute_token_bonds(
    tokens: list[TokenInfo],
    atoms: list[AtomInfo],
    input: StructurePredictionInput,
    chains: list[ChainInfo],
) -> torch.Tensor:
    """Build the symmetric token-bond matrix M with shape (l, l, 1)."""
    edges: set[tuple[int, int]] = set()

    def connect(left: int | None, right: int | None) -> None:
        if left is not None and right is not None and left != right:
            edges.add((min(left, right), max(left, right)))

    explicit_bonds = {
        (chain.asym_id, 0): chain.ligand_bonds for chain in chains if chain.ligand_bonds
    }
    for residue_key, atom_list in _atom_tokenized_residues(tokens, atoms).items():
        if not atom_list:
            continue
        residue_name = tokens[atom_list[0][1]].residue_name
        token_for_name = {name: token_index for name, token_index in atom_list}
        bonds = explicit_bonds.get(residue_key)
        if bonds is None:
            bonds = get_ligand_ccd_bonds(residue_name)
        if bonds:
            for left_name, right_name in bonds:
                if left_name in token_for_name and right_name in token_for_name:
                    connect(token_for_name[left_name], token_for_name[right_name])
        else:
            for left, right in combinations([token_index for _, token_index in atom_list], 2):
                connect(left, right)

    if input.covalent_bonds:
        chain_for_id = {chain.chain_id: chain for chain in chains}
        residue_atoms: dict[tuple[int, int], list[AtomInfo]] = defaultdict(list)
        for atom in atoms:
            if atom.is_valid and atom.token_index < len(tokens):
                token = tokens[atom.token_index]
                residue_atoms[(token.asym_id, token.residue_index)].append(atom)
        for bond in input.covalent_bonds:
            left_chain = chain_for_id.get(bond.chain_id1)
            right_chain = chain_for_id.get(bond.chain_id2)
            if left_chain is None or right_chain is None:
                continue
            left_atoms = residue_atoms.get((left_chain.asym_id, bond.res_idx1), [])
            right_atoms = residue_atoms.get((right_chain.asym_id, bond.res_idx2), [])
            if bond.atom_idx1 < len(left_atoms) and bond.atom_idx2 < len(right_atoms):
                connect(
                    left_atoms[bond.atom_idx1].token_index,
                    right_atoms[bond.atom_idx2].token_index,
                )

    protein_residues: dict[tuple[int, int], list[TokenInfo]] = defaultdict(list)
    for token in tokens:
        if token.mol_type == MOL_TYPE_PROTEIN:
            protein_residues[(token.asym_id, token.residue_index)].append(token)
    for (asym_id, residue_index), residue_tokens in protein_residues.items():
        if not any(token.res_type == PROTEIN_UNK_RES_TYPE for token in residue_tokens):
            continue
        previous = protein_residues.get((asym_id, residue_index - 1))
        following = protein_residues.get((asym_id, residue_index + 1))
        if previous:
            connect(
                _backbone_token(previous, "C", atoms),
                _backbone_token(residue_tokens, "N", atoms),
            )
        if following:
            connect(
                _backbone_token(residue_tokens, "C", atoms),
                _backbone_token(following, "N", atoms),
            )

    matrix = torch.zeros(len(tokens), len(tokens), 1, dtype=torch.float32)
    for left, right in edges:
        matrix[left, right, 0] = 1.0
        matrix[right, left, 0] = 1.0
    return matrix


def compute_representative_atoms(tokens: list[TokenInfo], atoms: list[AtomInfo]) -> torch.Tensor:
    """Choose one distogram atom per token and return indices I with shape (l,)."""
    named_atoms = _atom_indices_by_name(atoms)
    representatives = torch.zeros(len(tokens), dtype=torch.int64)
    for token in tokens:
        names = named_atoms.get(token.token_index, {})
        fallback = next(iter(names.values()), 0)
        if token.mol_type == MOL_TYPE_PROTEIN:
            representative = names.get("CB", names.get("CA", fallback))
        elif token.mol_type in (MOL_TYPE_DNA, MOL_TYPE_RNA):
            if token.res_type in (27, 32):
                representative = names.get("C1'", fallback)
            elif token.res_type in (23, 24, 28, 29):
                representative = names.get("C4", names.get("C1'", fallback))
            else:
                representative = names.get("C2", names.get("C1'", fallback))
        else:
            representative = fallback
        representatives[token.token_index] = representative
    return representatives


def _msa_assignments(
    input: StructurePredictionInput, chains: list[ChainInfo]
) -> dict[int, MSA | None]:
    chain_msas: dict[int, MSA | None] = {}
    chain_index = 0
    for item in input.sequences:
        chain_ids = [item.id] if isinstance(item.id, str) else list(item.id)
        for _ in chain_ids:
            chain = chains[chain_index]
            if isinstance(item, ProteinInput):
                chain_msas[chain.asym_id] = (
                    MSA.from_sequences([item.sequence]) if item.msa is None else item.msa
                )
            else:
                chain_msas[chain.asym_id] = None
            chain_index += 1
    return chain_msas


def compute_msa_features(
    input: StructurePredictionInput,
    chains: list[ChainInfo],
    tokens: list[TokenInfo],
    max_seqs: int = 16384,
) -> dict[str, torch.Tensor]:
    """Pair per-chain MSAs and return row features with shape (m, l)."""
    from .esmfold2_paired_msa import (
        construct_paired_msa,
        protein_letter_to_res_type,
    )

    chain_msas = _msa_assignments(input, chains)
    query_types = {
        chain.asym_id: np.asarray(
            [token.res_type for token in tokens if token.asym_id == chain.asym_id],
            dtype=np.int64,
        )
        for chain in chains
    }
    msa_residues, deletion_counts, _ = construct_paired_msa(
        chain_msas,
        query_types,
        np.asarray([token.asym_id for token in tokens], dtype=np.int64),
        np.asarray([token.residue_index for token in tokens], dtype=np.int64),
        letter_to_res_type=protein_letter_to_res_type(),
        max_seqs=max_seqs,
    )
    for token in tokens:
        if chain_msas.get(token.asym_id) is None:
            msa_residues[:, token.token_index] = MSA_GAP_TOKEN_ID
            msa_residues[0, token.token_index] = token.res_type
    if msa_residues.shape[0] == 0:
        msa_residues = np.full((1, len(tokens)), MSA_GAP_TOKEN_ID, dtype=np.int64)
        deletion_counts = np.zeros((1, len(tokens)), dtype=np.float32)

    msa = torch.from_numpy(msa_residues)
    deletion_count = torch.from_numpy(deletion_counts)
    deletion_value = (np.pi / 2) * torch.arctan(deletion_count / 3)
    return {
        "msa": msa,
        "deletion_value": deletion_value,
        "has_deletion": deletion_count > 0,
        "deletion_mean": deletion_value.mean(dim=0),
        "msa_attention_mask": torch.ones_like(msa, dtype=torch.bool),
    }


def compute_distogram_conditioning(
    input: StructurePredictionInput,
    chains: list[ChainInfo],
    tokens: list[TokenInfo],
    disto_center: torch.Tensor,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    num_bins: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bin user distances into D and return D plus its Boolean mask M."""
    del disto_center
    n_tokens = len(tokens)
    bins = torch.zeros((n_tokens, n_tokens), dtype=torch.long)
    mask = torch.zeros((n_tokens, n_tokens), dtype=torch.bool)
    if not input.distogram_conditioning:
        return bins, mask
    asym_for_chain = {chain.chain_id: chain.asym_id for chain in chains}
    tokens_for_asym: dict[int, list[int]] = defaultdict(list)
    for token in tokens:
        tokens_for_asym[token.asym_id].append(token.token_index)
    boundaries = torch.linspace(min_dist, max_dist, num_bins + 1)

    for conditioning in input.distogram_conditioning:
        asym_id = asym_for_chain.get(conditioning.chain_id)
        if asym_id is None:
            continue
        indices = tokens_for_asym[asym_id]
        distances = torch.as_tensor(conditioning.distogram, dtype=torch.float32)
        expected_shape = (len(indices), len(indices))
        if distances.shape != expected_shape:
            raise ValueError(
                f"Distogram shape {distances.shape} doesn't match chain length {len(indices)}"
            )
        selected = torch.bucketize(distances, boundaries[:-1]).sub(1).clamp(0, num_bins - 1)
        token_indices_tensor = torch.as_tensor(indices, dtype=torch.long)
        bins[token_indices_tensor[:, None], token_indices_tensor[None, :]] = selected
        mask[token_indices_tensor[:, None], token_indices_tensor[None, :]] = True
    return bins, mask


def _padded_atoms(atoms: list[AtomInfo]) -> list[AtomInfo]:
    target = math.ceil(len(atoms) / 32) * 32 if atoms else 32
    padding = [
        AtomInfo(
            name="",
            element="",
            charge=0,
            ref_pos=_ZERO_POS.copy(),
            pos=_ZERO_POS.copy(),
            token_index=0,
            atom_index=index,
            space_uid=0,
            is_valid=False,
        )
        for index in range(len(atoms), target)
    ]
    return [*atoms, *padding]


def _token_tensors(tokens: list[TokenInfo]) -> dict[str, torch.Tensor]:
    fields = {
        "token_index": "token_index",
        "residue_index": "residue_index",
        "asym_id": "asym_id",
        "sym_id": "sym_id",
        "entity_id": "entity_id",
        "mol_type": "mol_type",
        "res_type": "res_type",
        "input_ids": "input_id",
    }
    return {
        output_name: torch.from_numpy(
            np.asarray([getattr(token, attribute) for token in tokens], dtype=np.int64)
        )
        for output_name, attribute in fields.items()
    }


def _atom_tensors(atoms: list[AtomInfo]) -> dict[str, torch.Tensor]:
    n_atoms = len(atoms)
    ref_pos = np.zeros((n_atoms, 3), dtype=np.float32)
    ref_element = np.zeros(n_atoms, dtype=np.int64)
    ref_charge = np.zeros(n_atoms, dtype=np.int8)
    ref_name = np.zeros((n_atoms, 4), dtype=np.int64)
    ref_space = np.zeros(n_atoms, dtype=np.int64)
    atom_mask = np.zeros(n_atoms, dtype=np.bool_)
    atom_to_token = np.zeros(n_atoms, dtype=np.int64)
    positions = np.zeros((n_atoms, 3), dtype=np.float64)
    valid = np.zeros(n_atoms, dtype=np.bool_)
    for index, atom in enumerate(atoms):
        if atom.ref_pos is not None:
            ref_pos[index] = atom.ref_pos
        ref_charge[index] = atom.charge
        ref_space[index] = atom.space_uid if atom.space_uid >= 0 else atom.token_index
        atom_mask[index] = atom.is_valid
        valid[index] = atom.is_valid
        positions[index] = atom.pos
        if atom.is_valid:
            ref_element[index] = get_element_atomic_num(atom.element)
            ref_name[index] = encode_atom_name(atom.name)
            atom_to_token[index] = atom.token_index

    resolved = valid & np.any(positions != 0, axis=1)
    X = torch.from_numpy(positions)
    resolved_mask = torch.from_numpy(resolved)
    valid_mask = torch.from_numpy(valid)
    if resolved_mask.any():
        X = X - X[resolved_mask].mean(dim=0, keepdim=True)
        X[~valid_mask] = 0.0
    return {
        "ref_pos": torch.from_numpy(ref_pos),
        "ref_element": torch.from_numpy(ref_element),
        "ref_charge": torch.from_numpy(ref_charge),
        "ref_atom_name_chars": torch.from_numpy(ref_name),
        "ref_space_uid": torch.from_numpy(ref_space),
        "gt_coords": X.float().unsqueeze(0),
        "atom_attention_mask": torch.from_numpy(atom_mask),
        "atom_to_token": torch.from_numpy(atom_to_token),
        "is_resolved": torch.tensor(resolved, dtype=torch.bool),
    }


def build_feature_tensors(
    chains: list[ChainInfo],
    tokens: list[TokenInfo],
    atoms: list[AtomInfo],
    input: StructurePredictionInput,
) -> dict[str, torch.Tensor]:
    """Assemble the complete unbatched ESMFold2 feature dictionary."""
    token_features = _token_tensors(tokens)
    atom_features = _atom_tensors(_padded_atoms(atoms))
    frames, _ = compute_frame_indices(tokens, atoms)
    msa_features = compute_msa_features(input, chains, tokens)
    distogram, distogram_mask = compute_distogram_conditioning(
        input,
        chains,
        tokens,
        torch.zeros(len(tokens), 3, dtype=torch.float32),
    )
    return {
        **token_features,
        "token_bonds": compute_token_bonds(tokens, atoms, input, chains),
        "token_attention_mask": torch.ones(len(tokens), dtype=torch.bool),
        "pocket_feature": torch.zeros(len(tokens), dtype=torch.long),
        **atom_features,
        "distogram_atom_idx": compute_representative_atoms(tokens, atoms),
        "frames_idx": torch.from_numpy(frames).to(torch.int64),
        "disto_cond": distogram,
        "disto_cond_mask": distogram_mask,
        **msa_features,
    }


def prepare_esmfold2_input(
    input: StructurePredictionInput, seed: int | None = None
) -> tuple[dict[str, torch.Tensor], list[ChainInfo]]:
    """Convert one typed request to model features and output-chain metadata."""
    chains, tokens, atoms = build_chains_from_input(input, seed)
    return build_feature_tensors(chains, tokens, atoms, input), chains
