"""Biotite-backed mmCIF parsing used by ESMFold2 structure records."""

from __future__ import annotations

import functools
import io
import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime

import biotite.structure as bs
import biotite.structure.io.pdbx as pdbx
import numpy as np
from biotite.structure.io.pdbx import CIFColumn, CIFData, CIFFile

from . import esmfold2_residue_constants as residue_constants

PathOrBuffer = str | os.PathLike | io.StringIO

PLDDT_B_FACTOR_SCALE = 100.0
_MMCIF_COLUMN_DECIMALS = {
    "Cartn_x": 3,
    "Cartn_y": 3,
    "Cartn_z": 3,
    "B_iso_or_equiv": 2,
}
_NONPOLYMER_ENTITY_TYPES = frozenset({"NON-POLYMER", "WATER", "BRANCHED"})


class NoProteinError(Exception):
    """Raised internally when an mmCIF block contains no model-one atoms."""


@dataclass
class Residue:
    residue_number: int | None = None
    insertion_code: str = ""
    hetflag: bool = False


@dataclass
class MmcifHeader:
    release_date: datetime | None = None
    resolution: float | None = None
    structure_method: str = "UNKNOWN"


def round_mmcif_columns(cif_file: CIFFile) -> None:
    """Round coordinate and confidence columns in place for stable exports."""

    if "atom_site" not in cif_file.block:
        return
    atom_site = cif_file.block["atom_site"]
    for name, decimals in _MMCIF_COLUMN_DECIMALS.items():
        if name not in atom_site:
            continue
        original = atom_site[name]
        values = original.as_array(np.float64)
        strings = np.asarray(
            [f"{value:.{decimals}f}" for value in values],
            dtype=np.str_,
        )
        atom_site[name] = CIFColumn(
            data=CIFData(array=strings, dtype=np.str_),
            mask=original.mask,
        )


def _clean_chain_list(value: str) -> list[str]:
    return [chain.strip() for chain in value.split(",") if chain.strip()]


def _empty_residue() -> Residue:
    return Residue(residue_number=None, insertion_code="", hetflag=False)


def _header_from_block(
    block,
    header: MmcifHeader | None = None,
) -> MmcifHeader:
    header = MmcifHeader() if header is None else header
    try:
        if "pdbx_database_status" in block:
            category = block["pdbx_database_status"]
            if "recvd_initial_deposition_date" in category:
                value = category["recvd_initial_deposition_date"].as_item()
                if value and value != "?":
                    with suppress(ValueError):
                        header.release_date = datetime.strptime(value, "%Y-%m-%d")
        if "refine" in block:
            category = block["refine"]
            if "ls_d_res_high" in category:
                value = category["ls_d_res_high"].as_item()
                if value and value != "?":
                    with suppress(ValueError):
                        header.resolution = float(value)
        if "exptl" in block:
            category = block["exptl"]
            if "method" in category:
                value = category["method"].as_item()
                if value and value != "?":
                    header.structure_method = value.upper()
    except Exception:
        pass
    return header


def _entities_from_block(
    block,
    entities: dict[int, list[str]] | None = None,
) -> dict[int, list[str]]:
    entities = {} if entities is None else entities
    if "entity" in block:
        category = block["entity"]
        ids = category["id"].as_array(str)
        types = category["type"].as_array(str)
        for entity_id, _ in zip(ids, types, strict=False):
            entities[int(entity_id)] = []
    if "entity_poly" in block:
        category = block["entity_poly"]
        ids = category["entity_id"].as_array(str)
        chain_lists = category["pdbx_strand_id"].as_array(str)
        for raw_id, raw_chains in zip(ids, chain_lists, strict=False):
            entity_id = int(raw_id)
            if entity_id in entities:
                entities[entity_id] = _clean_chain_list(raw_chains)
    if "struct_asym" in block:
        category = block["struct_asym"]
        asym_ids = category["id"].as_array(str)
        entity_ids = category["entity_id"].as_array(str)
        for asym_id, raw_id in zip(asym_ids, entity_ids, strict=False):
            entity_id = int(raw_id)
            if entity_id in entities and not entities[entity_id]:
                entities[entity_id].append(asym_id)
    return entities


def _polymer_sequences(block) -> dict[str, str]:
    sequences: dict[str, str] = {}
    if "entity_poly" not in block:
        return sequences
    category = block["entity_poly"]
    entity_ids = category["entity_id"].as_array(str)
    raw_sequences = category["pdbx_seq_one_letter_code_can"].as_array(str)
    chain_lists = category["pdbx_strand_id"].as_array(str)
    for _, raw_sequence, raw_chains in zip(
        entity_ids,
        raw_sequences,
        chain_lists,
        strict=False,
    ):
        sequence = "".join(raw_sequence.split())
        for chain_id in _clean_chain_list(raw_chains):
            sequences[chain_id] = sequence
    return sequences


def _scheme_columns(category):
    asym_ids = category["asym_id"].as_array(str)
    insertion_codes = (
        category["pdb_ins_code"].as_array(str)
        if "pdb_ins_code" in category
        else [""] * len(asym_ids)
    )
    hetflags = category["hetflag"].as_array(str) if "hetflag" in category else ["N"] * len(asym_ids)
    author_chains = (
        category["pdb_strand_id"].as_array(str) if "pdb_strand_id" in category else asym_ids
    )
    return (
        asym_ids,
        category["seq_id"].as_array(str),
        category["auth_seq_num"].as_array(str),
        insertion_codes,
        hetflags,
        author_chains,
    )


def _scheme_residue_map(category):
    (
        asym_ids,
        sequence_positions,
        author_numbers,
        insertion_codes,
        hetflags,
        author_chains,
    ) = _scheme_columns(category)
    asym_to_author = {
        asym_id: author_id for asym_id, author_id in zip(asym_ids, author_chains, strict=False)
    }
    per_chain: dict[str, dict[int, Residue]] = {}
    for asym_id, raw_position, raw_number, raw_code, raw_hetflag in zip(
        asym_ids,
        sequence_positions,
        author_numbers,
        insertion_codes,
        hetflags,
        strict=False,
    ):
        residues = per_chain.setdefault(asym_id, {})
        try:
            position = int(raw_position) - 1
            residue_number = int(raw_number) if raw_number != "?" else None
        except ValueError:
            continue
        if residue_number is None:
            insertion_code = ""
        else:
            insertion_code = "" if raw_code in (".", "?") else raw_code
        residues[position] = Residue(
            residue_number=residue_number,
            insertion_code=insertion_code,
            hetflag=raw_hetflag.upper() == "Y",
        )
    return per_chain, asym_to_author


def _renumber_duplicate_residues(
    per_chain: dict[str, dict[int, Residue]],
) -> None:
    for residues in per_chain.values():
        positions_by_number: dict[int, list[int]] = {}
        for position, residue in residues.items():
            if residue.residue_number is not None:
                positions_by_number.setdefault(residue.residue_number, []).append(position)
        for number, positions in positions_by_number.items():
            if len(positions) <= 1:
                continue
            positions.sort()
            for offset, position in enumerate(positions):
                previous = residues[position]
                residues[position] = Residue(
                    residue_number=number + offset,
                    insertion_code=previous.insertion_code,
                    hetflag=previous.hetflag,
                )


def _ordered_scheme_mapping(
    per_chain: dict[str, dict[int, Residue]],
    asym_to_author: dict[str, str],
    chain_sequences: dict[str, str],
) -> dict[str, dict[int, Residue]]:
    result: dict[str, dict[int, Residue]] = {}
    for asym_id, residues in per_chain.items():
        author_chain = asym_to_author.get(asym_id, asym_id)
        if author_chain in chain_sequences:
            result[author_chain] = {
                position: residues.get(position, _empty_residue())
                for position in range(len(chain_sequences[author_chain]))
            }
        elif residues:
            result[author_chain] = {
                index: residues[position] for index, position in enumerate(sorted(residues))
            }
    return result


def _complete_polymer_mappings(
    mappings: dict[str, dict[int, Residue]],
    chain_sequences: dict[str, str],
) -> None:
    for chain_id, sequence in chain_sequences.items():
        mapping = mappings.setdefault(chain_id, {})
        for position in range(len(sequence)):
            if position not in mapping:
                mapping[position] = _empty_residue()


def _add_structure_fallbacks(
    mappings: dict[str, dict[int, Residue]],
    structure: bs.AtomArray,
) -> None:
    if not (
        structure
        and hasattr(structure, "chain_id")
        and structure.chain_id is not None
        and hasattr(structure.chain_id, "__iter__")
    ):
        return
    for chain_id in set(structure.chain_id):
        if chain_id in mappings:
            continue
        chain = structure[structure.chain_id == chain_id]
        if not (
            hasattr(chain, "res_id")
            and chain.res_id is not None
            and hasattr(chain.res_id, "__iter__")
        ):
            continue
        residue_ids = sorted(set(chain.res_id))
        mappings[chain_id] = {
            index: Residue(
                residue_number=residue_id,
                insertion_code="",
                hetflag=False,
            )
            for index, residue_id in enumerate(residue_ids)
        }


def _nonpolymer_entity_ids(block) -> set[str]:
    result = set()
    if "entity" not in block:
        return result
    category = block["entity"]
    ids = category["id"].as_array(str)
    types = category["type"].as_array(str)
    for entity_id, entity_type in zip(ids, types, strict=False):
        if entity_type.upper() in _NONPOLYMER_ENTITY_TYPES:
            result.add(entity_id)
    return result


def _nonpolymer_component_map(block, entity_ids: set[str]) -> dict[str, str]:
    result = {}
    if "pdbx_entity_nonpoly" not in block:
        return result
    category = block["pdbx_entity_nonpoly"]
    ids = category["entity_id"].as_array(str)
    components = category["comp_id"].as_array(str)
    for entity_id, component in zip(ids, components, strict=False):
        if entity_id in entity_ids:
            result[entity_id] = component
    return result


class MmcifWrapper:
    """Parsed model-one structure, metadata, sequences, and residue mappings."""

    def __init__(self, id: str | None = None):
        self.id = id or ""
        self.raw: pdbx.CIFFile | None = None
        self.structure: bs.AtomArray
        self.header = MmcifHeader()
        self.entities: dict[int, list[str]] = {}
        self.chain_to_seqres: dict[str, str] = {}
        self.seqres_to_structure: dict[str, dict[int, Residue]] = {}

    @classmethod
    def read(cls, path: PathOrBuffer, id: str | None = None) -> MmcifWrapper:
        wrapper = cls(id=id)
        wrapper._load(path)
        return wrapper

    def _load(self, path: PathOrBuffer, fileid: str | None = None) -> None:
        self.raw = pdbx.CIFFile.read(path)
        self._parse_structure()
        self._parse_header()
        self._parse_entities()
        self._parse_sequences()

    def _parse_structure(self) -> None:
        try:
            structure = pdbx.get_structure(self.raw, model=1)
            if structure is None or not isinstance(structure, bs.AtomArray):
                raise NoProteinError("No structure found in mmCIF file")
            if len(structure) == 0:
                raise NoProteinError("Empty structure in mmCIF file")
            self.structure = structure
        except Exception as error:
            raise ValueError(f"Failed to parse structure: {error}") from error

    def _parse_header(self) -> None:
        if self.raw:
            self.header = _header_from_block(self.raw.block, self.header)

    def _parse_entities(self) -> None:
        if not self.raw:
            return
        try:
            self.entities = _entities_from_block(self.raw.block, self.entities)
        except Exception:
            if (
                self.structure
                and hasattr(self.structure, "chain_id")
                and self.structure.chain_id is not None
                and hasattr(self.structure.chain_id, "__iter__")
            ):
                self.entities = {1: list(set(self.structure.chain_id))}

    def _parse_sequences(self) -> None:
        if not self.raw:
            return
        block = self.raw.block
        self.chain_to_seqres.update(_polymer_sequences(block))
        if "pdbx_poly_seq_scheme" in block:
            per_chain, asym_to_author = _scheme_residue_map(block["pdbx_poly_seq_scheme"])
            _renumber_duplicate_residues(per_chain)
            self.seqres_to_structure.update(
                _ordered_scheme_mapping(
                    per_chain,
                    asym_to_author,
                    self.chain_to_seqres,
                )
            )
        _complete_polymer_mappings(
            self.seqres_to_structure,
            self.chain_to_seqres,
        )
        _add_structure_fallbacks(self.seqres_to_structure, self.structure)

    def _parse_nonpoly_from_mmcif(self) -> dict[tuple, bs.AtomArray]:
        assert self.raw is not None
        block = self.raw.block
        entity_ids = _nonpolymer_entity_ids(block)
        _nonpolymer_component_map(block, entity_ids)
        groups: dict[tuple[str, str], list[int]] = {}
        if "atom_site" in block:
            category = block["atom_site"]
            chain_ids = category["label_asym_id"].as_array(str)
            atom_entity_ids = category["label_entity_id"].as_array(str)
            component_ids = category["label_comp_id"].as_array(str)
            for index, (chain_id, entity_id, component_id) in enumerate(
                zip(chain_ids, atom_entity_ids, component_ids, strict=False)
            ):
                if entity_id in entity_ids:
                    groups.setdefault((component_id, chain_id), []).append(index)

        coordinates = {}
        for component_id, chain_id in groups:
            selection = (self.structure.chain_id == chain_id) & (
                self.structure.res_name == component_id
            )
            if not selection.any():
                continue
            atoms = self.structure[selection]
            if isinstance(atoms, (bs.AtomArray, bs.AtomArrayStack)) and len(atoms) > 0:
                coordinates[(component_id, chain_id)] = atoms
        return coordinates

    def _parse_nonpoly_fallback(self) -> dict[tuple, bs.AtomArray]:
        result = {}
        if not (self.structure and hasattr(self.structure, "chain_id")):
            return result
        standard_residues = set(residue_constants.resnames[:-1])
        standard_residues.update({"A", "C", "G", "T", "U"})
        if self.structure.chain_id is None:
            return result
        for chain_id in set(self.structure.chain_id):
            chain = self.structure[self.structure.chain_id == chain_id]
            if not (
                hasattr(chain, "res_name")
                and chain.res_name is not None
                and hasattr(chain.res_name, "__iter__")
            ):
                continue
            for residue_name in set(chain.res_name):
                if residue_name in standard_residues:
                    continue
                selection = (chain.chain_id == chain_id) & (chain.res_name == residue_name)
                if selection.any() and isinstance(
                    chain,
                    (bs.AtomArray, bs.AtomArrayStack),
                ):
                    result[(residue_name, chain_id)] = chain[selection]
        return result

    @functools.cached_property
    def non_polymer_coords(self) -> dict[tuple, bs.AtomArray]:
        """Map each non-polymer component and chain to its atoms."""

        if not self.structure or not self.raw:
            return {}
        try:
            return self._parse_nonpoly_from_mmcif()
        except Exception:
            return self._parse_nonpoly_fallback()
