"""Typed, JSON-safe inputs for ESMFold2 feature preparation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

from .esmfold2_msa import MSA

MSAInput: TypeAlias = MSA | None


@dataclass
class Modification:
    """A zero-indexed residue substitution using a CCD component."""

    position: int
    ccd: str
    smiles: str | None = None


@dataclass
class ProteinInput:
    id: str | list[str]
    sequence: str
    modifications: list[Modification] | None = None
    msa: MSAInput = None


@dataclass
class RNAInput:
    id: str | list[str]
    sequence: str
    modifications: list[Modification] | None = None


@dataclass
class DNAInput:
    id: str | list[str]
    sequence: str
    modifications: list[Modification] | None = None


@dataclass
class LigandInput:
    id: str | list[str]
    smiles: str | None = None
    ccd: list[str] | None = None


@dataclass
class DistogramConditioning:
    chain_id: str
    distogram: np.ndarray


@dataclass
class PocketConditioning:
    binder_chain_id: str
    contacts: list[tuple[str, int]]


@dataclass
class CovalentBond:
    chain_id1: str
    res_idx1: int
    atom_idx1: int
    chain_id2: str
    res_idx2: int
    atom_idx2: int


SequenceInput: TypeAlias = ProteinInput | RNAInput | DNAInput | LigandInput


@dataclass
class StructurePredictionInput:
    sequences: Sequence[SequenceInput]
    pocket: PocketConditioning | None = None
    distogram_conditioning: list[DistogramConditioning] | None = None
    covalent_bonds: list[CovalentBond] | None = None


_CHAIN_TYPE = {
    ProteinInput: "protein",
    RNAInput: "rna",
    DNAInput: "dna",
}


def _serialize_modifications(
    modifications: list[Modification] | None,
) -> list[dict[str, Any]] | None:
    if not modifications:
        return None
    return [{"position": item.position, "ccd": item.ccd} for item in modifications]


def _serialize_chain(chain: SequenceInput) -> dict[str, Any]:
    if isinstance(chain, LigandInput):
        return {
            "smiles": chain.smiles,
            "id": chain.id,
            "ccd": chain.ccd,
            "type": "ligand",
        }

    chain_type = _CHAIN_TYPE.get(type(chain))
    if chain_type is None:
        raise ValueError(f"Unsupported sequence input type: {type(chain)}")
    serialized: dict[str, Any] = {
        "sequence": chain.sequence,
        "id": chain.id,
        "type": chain_type,
    }
    if modifications := _serialize_modifications(chain.modifications):
        serialized["modifications"] = modifications
    if isinstance(chain, ProteinInput):
        if chain.msa is not None and not isinstance(chain.msa, MSA):
            raise AttributeError(f"MSA must be None or MSA. Got {chain.msa} instead.")
        serialized["msa"] = None if chain.msa is None else {"sequences": chain.msa.sequences}
    return serialized


def serialize_structure_prediction_input(
    structure_input: StructurePredictionInput,
) -> dict[str, Any]:
    """Convert an input object to a JSON-safe mapping."""

    serialized: dict[str, Any] = {
        "sequences": [_serialize_chain(chain) for chain in structure_input.sequences]
    }
    if structure_input.covalent_bonds is not None:
        serialized["covalent_bonds"] = [
            vars(bond).copy() for bond in structure_input.covalent_bonds
        ]
    if structure_input.pocket is not None:
        serialized["pocket"] = {
            "binder_chain_id": structure_input.pocket.binder_chain_id,
            "contacts": structure_input.pocket.contacts,
        }
    if structure_input.distogram_conditioning is not None:
        serialized["distogram_conditioning"] = [
            {"chain_id": item.chain_id, "distogram": item.distogram.tolist()}
            for item in structure_input.distogram_conditioning
        ]
    return serialized


def _deserialize_modifications(chain: dict[str, Any]) -> list[Modification] | None:
    raw = chain.get("modifications")
    if not raw:
        return None
    return [Modification(position=item["position"], ccd=item["ccd"]) for item in raw]


def _deserialize_msa(chain: dict[str, Any]) -> MSAInput:
    raw = chain.get("msa")
    if raw is None:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("sequences"), list):
        raise ValueError(f"Unexpected MSA value: {raw!r}")
    return MSA.from_sequences(raw["sequences"])


def _deserialize_chain(chain: dict[str, Any]) -> SequenceInput:
    chain_type = chain.get("type")
    common = {"id": chain["id"]}
    if chain_type == "protein":
        return ProteinInput(
            **common,
            sequence=chain["sequence"],
            modifications=_deserialize_modifications(chain),
            msa=_deserialize_msa(chain),
        )
    if chain_type == "rna":
        return RNAInput(
            **common,
            sequence=chain["sequence"],
            modifications=_deserialize_modifications(chain),
        )
    if chain_type == "dna":
        return DNAInput(
            **common,
            sequence=chain["sequence"],
            modifications=_deserialize_modifications(chain),
        )
    if chain_type == "ligand":
        return LigandInput(**common, smiles=chain.get("smiles"), ccd=chain.get("ccd"))
    raise ValueError(f"Unsupported sequence type: {chain_type!r}")


def deserialize_structure_prediction_input(data: dict[str, Any]) -> StructurePredictionInput:
    """Reconstruct the typed input represented by a serialized mapping."""

    pocket_data = data.get("pocket")
    pocket = None
    if pocket_data is not None:
        pocket = PocketConditioning(
            binder_chain_id=pocket_data["binder_chain_id"],
            contacts=[tuple(contact) for contact in pocket_data["contacts"]],
        )

    distogram_data = data.get("distogram_conditioning")
    distograms = None
    if distogram_data is not None:
        distograms = [
            DistogramConditioning(
                chain_id=item["chain_id"], distogram=np.asarray(item["distogram"])
            )
            for item in distogram_data
        ]

    bond_data = data.get("covalent_bonds")
    bonds = None
    if bond_data is not None:
        bonds = [CovalentBond(**item) for item in bond_data]

    return StructurePredictionInput(
        sequences=[_deserialize_chain(chain) for chain in data["sequences"]],
        pocket=pocket,
        distogram_conditioning=distograms,
        covalent_bonds=bonds,
    )


__all__ = [
    "CovalentBond",
    "DNAInput",
    "DistogramConditioning",
    "LigandInput",
    "MSAInput",
    "Modification",
    "PocketConditioning",
    "ProteinInput",
    "RNAInput",
    "SequenceInput",
    "StructurePredictionInput",
    "deserialize_structure_prediction_input",
    "serialize_structure_prediction_input",
]
