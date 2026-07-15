"""Lazy access to Chemical Component Dictionary conformers.

The feature pipeline depends on atom names, formal charges, bonds, leaving-atom
flags, and one preferred reference conformer. Asset resolution is explicit at
``load_ccd`` time; importing this module performs no download or file access.
"""

from __future__ import annotations

import os
import pickle
from hashlib import file_digest
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download

from fastplms.registry import RuntimeAsset, get_model_registry

from .esmfold2_constants import RES_TYPE_TO_CCD

_CCD_ENVIRONMENT_VARIABLE = "ESMCFOLD_CCD_PATH"
_CCD_ASSET_ID = "esmfold2_ccd"


def _asset_contract() -> RuntimeAsset:
    """Return the manifest-owned identity of the trusted CCD pickle."""

    try:
        asset = get_model_registry().runtime_assets[_CCD_ASSET_ID]
    except KeyError as error:
        raise RuntimeError(
            f"The package manifest does not declare runtime asset {_CCD_ASSET_ID!r}."
        ) from error
    if asset.trust_kind != "hash_pinned_pickle":
        raise RuntimeError(
            f"Runtime asset {_CCD_ASSET_ID!r} must use the hash_pinned_pickle trust policy."
        )
    return asset


def _verify_asset(asset_path: Path, contract: RuntimeAsset) -> None:
    """Verify the complete immutable asset identity before deserialization."""

    actual_size = asset_path.stat().st_size
    if actual_size != contract.size:
        raise ValueError(
            f"CCD asset size mismatch: expected {contract.size} bytes, received {actual_size}."
        )
    with asset_path.open("rb") as handle:
        actual_hash = file_digest(handle, "sha256").hexdigest()
    if actual_hash != contract.sha256:
        raise ValueError(
            "CCD asset SHA256 mismatch; refusing to cross the trusted-pickle boundary."
        )


class _ChemicalComponentStore:
    def __init__(self) -> None:
        self.molecules: dict[str, Any] | None = None
        self.conformers: dict[str, dict[str, np.ndarray]] = {}
        self.atoms: dict[str, list[tuple[str, str, int]]] = {}
        self.bonds: dict[str, list[tuple[str, str]]] = {}
        self.leaving_atoms: dict[str, set[str]] = {}
        self.standard_positions: dict[tuple[int, str], np.ndarray | None] = {}
        self.ligand_positions: dict[tuple[str, str], np.ndarray | None] = {}

    def load(self, cache_dir: Path | str | None = None) -> dict[str, Any]:
        if self.molecules is not None:
            return self.molecules
        asset = self._resolve_asset(cache_dir)
        try:
            # SECURITY: _resolve_asset verifies the manifest-pinned byte length and
            # SHA256 before this trusted Biohub pickle is deserialized.
            with asset.open("rb") as handle:
                loaded = pickle.load(handle)
        except Exception as error:
            raise ValueError(f"Could not read the CCD asset at {asset}: {error}") from error
        if loaded is not None and not isinstance(loaded, dict):
            raise TypeError("The CCD asset must contain a component dictionary.")
        self.molecules = loaded or {}
        return self.molecules

    @staticmethod
    def _resolve_asset(cache_dir: Path | str | None) -> Path:
        contract = _asset_contract()
        configured = os.environ.get(_CCD_ENVIRONMENT_VARIABLE)
        if configured:
            asset = Path(configured).expanduser()
        elif cache_dir is not None:
            asset = Path(cache_dir).expanduser() / contract.path
        else:
            try:
                asset = Path(
                    hf_hub_download(
                        repo_id=contract.repository,
                        filename=contract.path,
                        revision=contract.revision,
                    )
                )
            except Exception as error:
                raise FileNotFoundError(
                    "Could not resolve the ESMFold2 CCD asset. Set "
                    f"{_CCD_ENVIRONMENT_VARIABLE} or populate the Hugging Face cache."
                ) from error
        if not asset.is_file():
            raise FileNotFoundError(f"CCD asset does not exist: {asset}")
        _verify_asset(asset, contract)
        return asset

    def _component_with_conformer(self, component_id: str):
        molecule = self.load().get(component_id)
        if molecule is None or molecule.GetNumConformers() == 0:
            return None, None

        conformers = list(molecule.GetConformers())
        priority = {"Computed": 0, "Ideal": 1}
        selected_index = min(
            range(len(conformers)),
            key=lambda index: priority.get(conformers[index].GetPropsAsDict().get("name"), 2),
        )

        from rdkit import Chem

        heavy_molecule = Chem.RemoveHs(molecule, sanitize=False)
        if heavy_molecule.GetNumConformers() == 0:
            return None, None
        conformer_index = min(selected_index, heavy_molecule.GetNumConformers() - 1)
        return heavy_molecule, heavy_molecule.GetConformer(conformer_index)

    def conformer(self, component_id: str) -> dict[str, np.ndarray] | None:
        if component_id not in self.conformers:
            molecule, conformer = self._component_with_conformer(component_id)
            positions: dict[str, np.ndarray] = {}
            if molecule is not None and conformer is not None:
                for atom in molecule.GetAtoms():
                    atom_name = atom.GetPropsAsDict().get("name")
                    if not isinstance(atom_name, str) or not atom_name:
                        continue
                    point = conformer.GetAtomPosition(atom.GetIdx())
                    positions[atom_name] = np.asarray((point.x, point.y, point.z), dtype=np.float32)
            self.conformers[component_id] = positions
        result = self.conformers[component_id]
        return result or None

    def atom_records(self, component_id: str) -> list[tuple[str, str, int]] | None:
        if component_id not in self.atoms:
            molecule, _conformer = self._component_with_conformer(component_id)
            records: list[tuple[str, str, int]] = []
            if molecule is not None:
                for atom in molecule.GetAtoms():
                    atom_name = atom.GetPropsAsDict().get("name")
                    if isinstance(atom_name, str) and atom_name:
                        records.append((atom_name, atom.GetSymbol(), atom.GetFormalCharge()))
            self.atoms[component_id] = records
        result = self.atoms[component_id]
        return result or None

    def bond_records(self, component_id: str) -> list[tuple[str, str]] | None:
        if component_id not in self.bonds:
            molecule, _conformer = self._component_with_conformer(component_id)
            records: list[tuple[str, str]] = []
            if molecule is not None:
                names = {
                    atom.GetIdx(): atom.GetPropsAsDict().get("name") for atom in molecule.GetAtoms()
                }
                for bond in molecule.GetBonds():
                    first = names.get(bond.GetBeginAtomIdx())
                    second = names.get(bond.GetEndAtomIdx())
                    if isinstance(first, str) and first and isinstance(second, str) and second:
                        records.append((first, second))
            self.bonds[component_id] = records
        result = self.bonds[component_id]
        return result or None

    def component_leaving_atoms(self, component_id: str) -> set[str]:
        if component_id not in self.leaving_atoms:
            molecule = self.load().get(component_id)
            names: set[str] = set()
            if molecule is not None:
                for atom in molecule.GetAtoms():
                    if atom.HasProp("leaving_atom") and atom.GetProp("leaving_atom") == "1":
                        name = atom.GetProp("name") if atom.HasProp("name") else ""
                        if name:
                            names.add(name)
            self.leaving_atoms[component_id] = names
        return self.leaving_atoms[component_id]


_STORE = _ChemicalComponentStore()


def load_ccd(cache_dir: Path | str | None = None) -> dict[str, Any]:
    """Load and cache the CCD asset, resolving it only when called."""

    return _STORE.load(cache_dir)


def get_ccd_conformer(component_id: str) -> dict[str, np.ndarray] | None:
    """Return the preferred heavy-atom conformer by atom name."""

    return _STORE.conformer(component_id)


def get_idealized_atom_pos(res_type: int, atom_name: str) -> np.ndarray | None:
    """Return one standard-residue atom position from the preferred conformer."""

    key = (res_type, atom_name)
    if key not in _STORE.standard_positions:
        component_id = RES_TYPE_TO_CCD.get(res_type)
        conformer = _STORE.conformer(component_id) if component_id is not None else None
        _STORE.standard_positions[key] = None if conformer is None else conformer.get(atom_name)
    return _STORE.standard_positions[key]


def get_ligand_idealized_atom_pos(residue_name: str, atom_name: str) -> np.ndarray | None:
    """Return one ligand atom position from the preferred conformer."""

    key = (residue_name, atom_name)
    if key not in _STORE.ligand_positions:
        conformer = _STORE.conformer(residue_name)
        _STORE.ligand_positions[key] = None if conformer is None else conformer.get(atom_name)
    return _STORE.ligand_positions[key]


def get_ligand_ccd_atoms_with_charges(
    component_id: str,
) -> list[tuple[str, str, int]] | None:
    """Return heavy-atom name, element, and formal-charge records."""

    return _STORE.atom_records(component_id)


def get_ligand_ccd_bonds(component_id: str) -> list[tuple[str, str]] | None:
    """Return bonds as component atom-name pairs."""

    return _STORE.bond_records(component_id)


def get_ccd_leaving_atoms(component_id: str) -> set[str]:
    """Return atoms removed when a CCD component is polymerized."""

    return _STORE.component_leaving_atoms(component_id)


__all__ = [
    "get_ccd_conformer",
    "get_ccd_leaving_atoms",
    "get_idealized_atom_pos",
    "get_ligand_ccd_atoms_with_charges",
    "get_ligand_ccd_bonds",
    "get_ligand_idealized_atom_pos",
    "load_ccd",
]
