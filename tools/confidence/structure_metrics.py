"""TM-score and DockQ adapters for cached ESMFold2 structures."""

from __future__ import annotations

import json
import re

from pathlib import Path
from typing import Any
from collections.abc import Mapping


_ONE_TO_THREE = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def _array(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _squeeze_coordinates(value: Any) -> Any:
    array = _array(value)
    while len(array.shape) > 2 and array.shape[0] == 1:
        array = array[0]
    if len(array.shape) != 2 or array.shape[1] != 3:
        raise ValueError("coordinates must have shape (A, 3) after singleton removal")
    return array


def _atom_names(value: Any, atom_count: int) -> list[str]:
    array = _array(value)
    names: list[str] = []
    for row in array.reshape(atom_count, -1):
        chars = []
        for code in row[:4]:
            if isinstance(code, bytes | str):
                chars.append(code.decode() if isinstance(code, bytes) else code)
            else:
                chars.append(chr(int(code) + 32) if int(code) else " ")
        names.append("".join(chars).strip() or "CA")
    return names


def _chain_layout(record: dict[str, Any]) -> tuple[list[str], list[str], list[int]]:
    chains = record.get("chains")
    if not isinstance(chains, list) or not chains:
        raise ValueError("record requires non-empty chains")
    chain_ids: list[str] = []
    sequence: list[str] = []
    token_chain: list[int] = []
    for chain_index, chain in enumerate(chains):
        chain_id = str(chain.get("id", chr(65 + chain_index)))
        chain_sequence = str(chain.get("sequence", ""))
        if not chain_sequence:
            raise ValueError("each chain requires a sequence")
        chain_ids.append(chain_id)
        sequence.append(chain_sequence)
        token_chain.extend([chain_index] * len(chain_sequence))
    return chain_ids, sequence, token_chain


def _ca_coordinates(cache: Mapping[str, Any], record: dict[str, Any]) -> tuple[Any, Any, str, str]:
    import numpy as np

    predicted = _squeeze_coordinates(cache["x_pred"])
    truth = _squeeze_coordinates(cache["true_coords"])
    if predicted.shape != truth.shape:
        raise ValueError("predicted and true coordinates must have equal shape")
    resolved = _array(cache["resolved_mask"]).reshape(-1).astype(bool)
    if len(resolved) != len(predicted):
        raise ValueError("resolved_mask must match coordinate count")
    if not np.isfinite(predicted[resolved]).all() or not np.isfinite(truth[resolved]).all():
        raise ValueError("resolved structure coordinates must be finite")
    backbone = _array(cache["backbone_indices"])
    atom_count = predicted.shape[0]
    if backbone.ndim == 3:
        backbone = backbone[0]
    if backbone.ndim != 2 or backbone.shape[1] != 3:
        raise ValueError("backbone_indices must have shape (L, 3)")
    token_mask = (
        _array(cache.get("token_attention_mask", np.ones(backbone.shape[0], dtype=bool)))
        .reshape(-1)
        .astype(bool)
    )
    if len(token_mask) != backbone.shape[0]:
        raise ValueError("token_attention_mask must match backbone token count")
    chain_ids, sequences, token_chain = _chain_layout(record)
    if len(token_chain) > backbone.shape[0]:
        raise ValueError("chain sequences exceed cache token count")
    if token_mask[len(token_chain) :].any():
        raise ValueError("padded backbone tokens must be masked")
    ca_indices = backbone[:, 1].astype(int)
    safe_ca_indices = np.clip(ca_indices, 0, max(atom_count - 1, 0))
    real_token = np.arange(backbone.shape[0]) < len(token_chain)
    finite = np.isfinite(predicted).all(axis=1) & np.isfinite(truth).all(axis=1)
    valid = (
        token_mask
        & real_token
        & (ca_indices >= 0)
        & (ca_indices < atom_count)
        & resolved[safe_ca_indices]
        & finite[safe_ca_indices]
    )
    predicted_ca = predicted[ca_indices[valid]]
    truth_ca = truth[ca_indices[valid]]
    sequence = "".join(sequence for sequence in sequences)
    filtered_sequence = "".join(
        residue for residue, keep in zip(sequence, valid, strict=False) if keep
    )
    if len(filtered_sequence) != len(predicted_ca):
        raise ValueError("CA sequence and coordinates are inconsistent")
    return (
        predicted_ca,
        truth_ca,
        filtered_sequence,
        "".join(chain_ids[token_chain[position]] for position, keep in enumerate(valid) if keep),
    )


def _write_pdb(
    path: Path, coordinates: Any, cache: Mapping[str, Any], record: dict[str, Any]
) -> None:
    import numpy as np

    coords = _squeeze_coordinates(coordinates)
    resolved = _array(cache["resolved_mask"]).reshape(-1).astype(bool)
    atom_mask = (
        _array(cache.get("atom_attention_mask", np.ones(len(resolved), dtype=bool)))
        .reshape(-1)
        .astype(bool)
    )
    atom_to_token = _array(cache["atom_to_token"]).reshape(-1).astype(int)
    names = _atom_names(cache["ref_atom_name_chars"], len(coords))
    atom_count = len(coords)
    if any(len(values) != atom_count for values in (resolved, atom_mask, atom_to_token, names)):
        raise ValueError("cache atom-axis fields must match coordinate count")
    emitted = resolved & atom_mask
    if not np.isfinite(coords[emitted]).all():
        raise ValueError("emitted structure coordinates must be finite")
    chain_ids, sequences, token_chain = _chain_layout(record)
    token_offsets: dict[int, int] = {}
    offset = 0
    for chain_index, sequence in enumerate(sequences):
        token_offsets[chain_index + 1] = offset
        offset += len(sequence)
    lines: list[str] = []
    serial = 1
    for atom_index, coordinate in enumerate(coords):
        if not resolved[atom_index] or not atom_mask[atom_index]:
            continue
        token = int(atom_to_token[atom_index])
        if token < 0 or token >= len(token_chain):
            continue
        chain_index = token_chain[token]
        chain_number = chain_index + 1
        residue_index = token - token_offsets[chain_number] + 1
        sequence = sequences[chain_index]
        residue = sequence[residue_index - 1] if 0 < residue_index <= len(sequence) else "X"
        x, y, z = (float(value) for value in np.asarray(coordinate))
        pdb_chain_id = chr(65 + chain_index)
        lines.append(
            f"ATOM  {serial:5d} {names[atom_index]:>4s} {_ONE_TO_THREE.get(residue, 'UNK'):>3s} "
            f"{pdb_chain_id:1s}{residue_index:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {names[atom_index][0]:>2s}"
        )
        serial += 1
    path.write_text("\n".join(lines) + "\nEND\n", encoding="ascii")


def _dockq_score(model_path: Path, native_path: Path) -> float:
    from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

    result = run_on_all_native_interfaces(load_PDB(str(model_path)), load_PDB(str(native_path)))
    if not isinstance(result, tuple) or len(result) < 2:
        raise ValueError("DockQ API returned an unexpected result")
    return float(result[1])


def compute_structure_metrics(
    cache: Mapping[str, Any], record: dict[str, Any], workdir: Path
) -> dict[str, float | None]:
    """Compute native TM-score and, for dimers, DockQ from a structure cache."""
    import numpy as np
    from tmtools import tm_align

    predicted_ca, truth_ca, sequence, _ = _ca_coordinates(cache, record)
    if len(sequence) < 2:
        raise ValueError("at least two resolved C-alpha atoms are required")
    alignment = tm_align(np.asarray(predicted_ca), np.asarray(truth_ca), sequence, sequence)
    tm_score = float(max(alignment.tm_norm_chain1, alignment.tm_norm_chain2))
    if len(record["chains"]) < 2:
        return {"tm_score": tm_score, "dockq": None}
    workdir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record.get("id", "target")))
    model_path = workdir / f"{safe_id}.model.pdb"
    native_path = workdir / f"{safe_id}.native.pdb"
    _write_pdb(model_path, cache["x_pred"], cache, record)
    _write_pdb(native_path, cache["true_coords"], cache, record)
    return {"tm_score": tm_score, "dockq": _dockq_score(model_path, native_path)}


def audit_smoke_structures(root: Path) -> dict[str, Any]:
    """Audit one cached monomer and dimer for each model on CPU.

    This reads at most the newest two benchmark cache files per model and never
    invokes a folding model. Every missing or invalid target is retained in the
    JSON report so a smoke failure cannot be mistaken for a successful audit.
    """
    from safetensors import safe_open

    records_path = root / "smoke" / "data" / "records.json"
    report: dict[str, Any] = {"status": "complete", "models": {}}
    if not records_path.exists():
        report.update(status="failed", error="missing smoke/data/records.json")
        (root / "smoke" / "structure-audit.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report
    records = json.loads(records_path.read_text(encoding="utf-8"))
    by_id = {str(record.get("id")): record for record in records}
    for model_id in ("esmfold2_300", "esmfold2_600"):
        model_report: dict[str, Any] = {"status": "complete", "targets": {}}
        benchmark_dirs = sorted(
            (root / "smoke" / model_id / "benchmark").glob("*/cache"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        cache_paths = (
            sorted(
                benchmark_dirs[0].glob("*.safetensors"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[:2]
            if benchmark_dirs
            else []
        )
        metadata_by_kind: dict[str, tuple[Path, dict[str, str]]] = {}
        for cache_path in cache_paths:
            with safe_open(str(cache_path), framework="pt", device="cpu") as handle:
                metadata = dict(handle.metadata() or {})
            record = by_id.get(metadata.get("record_id", ""))
            if record is None:
                continue
            kind = (
                "monomer"
                if len(record.get("chains", [])) == 1
                else "dimer"
                if len(record.get("chains", [])) == 2
                else "unsupported"
            )
            if kind in ("monomer", "dimer") and kind not in metadata_by_kind:
                metadata_by_kind[kind] = (cache_path, metadata)
        for kind in ("monomer", "dimer"):
            item: dict[str, Any] = {"kind": kind}
            selected = metadata_by_kind.get(kind)
            if selected is None:
                item.update(status="missing", error="no matching benchmark cache")
                model_report["status"] = "failed"
                model_report["targets"][kind] = item
                continue
            cache_path, metadata = selected
            record = by_id[metadata["record_id"]]
            try:
                from .cache import load_cache
                from fastplms.registry import get_model_spec

                cache, _ = load_cache(
                    cache_path,
                    model_id=model_id,
                    model_revision=get_model_spec(model_id).fast.revision,
                )
                metrics = compute_structure_metrics(
                    cache, record, root / "smoke" / model_id / "structure-audit" / kind
                )
                item.update(
                    status="passed", record_id=record["id"], cache=str(cache_path), metrics=metrics
                )
            except Exception as error:
                item.update(
                    status="invalid",
                    record_id=record.get("id"),
                    cache=str(cache_path),
                    error=f"{type(error).__name__}: {error}",
                )
                model_report["status"] = "failed"
            model_report["targets"][kind] = item
        report["models"][model_id] = model_report
        if model_report["status"] != "complete":
            report["status"] = "failed"
    destination = root / "smoke" / "structure-audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
