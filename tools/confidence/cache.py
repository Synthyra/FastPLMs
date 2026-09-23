"""Build immutable, model-specific caches for confidence-head training.

The cache keeps the frozen trunk states and the inputs needed by the native
confidence head.  Relative-position and token-bond embeddings are deliberately
recomputed from the frozen model when a cache is consumed.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
from torch import Tensor

from fastplms.models.esmfold2.esmfold2_input_builder import ProteinInput, StructurePredictionInput
from fastplms.models.esmfold2.modeling_esmfold2_experimental import ESMFold2ExperimentalModel
from fastplms.registry import get_model_spec
from .data import validate_structure_npz


_CACHE_SCHEMA = "fastplms.confidence.cache.v2"
_MODEL_IDS = frozenset({"esmfold2_300", "esmfold2_600"})
_FEATURE_KEYS = (
    "token_index",
    "residue_index",
    "asym_id",
    "sym_id",
    "entity_id",
    "mol_type",
    "token_bonds",
    "token_attention_mask",
    "ref_atom_name_chars",
    "atom_attention_mask",
    "atom_to_token",
    "distogram_atom_idx",
)
_AMBIGUOUS_ATOM_PAIRS = {
    "ASP": (("OD1", "OD2"),),
    "GLU": (("OE1", "OE2"),),
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
    # Equivalent-atom groups from the pinned AtlasFold residue table.
    "ARG": (("NH1", "NH2"),),
    "LEU": (("CD1", "CD2"),),
    "VAL": (("CG1", "CG2"),),
}


def _record_hash(record: Mapping[str, object]) -> str:
    encoded = json.dumps(dict(record), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _backbone_indices(features: Mapping[str, Tensor]) -> Tensor:
    # Native feature axes: atoms, tokens, and four encoded atom-name characters.
    atom_to_token = features["atom_to_token"].reshape(-1).long()  # (atoms,)
    atom_mask = features["atom_attention_mask"].reshape(-1).bool()  # (atoms,)
    names = [_decode_atom_name(row) for row in features["ref_atom_name_chars"].reshape(-1, 4)]
    token_count = int(features["token_attention_mask"].reshape(-1).shape[0])
    indices = torch.full((token_count, 3), -1, dtype=torch.long)  # (tokens, 3), N/CA/C atom indices
    for atom_index, token in enumerate(atom_to_token.tolist()):
        if not atom_mask[atom_index] or token >= token_count:
            continue
        slot = {"N": 0, "CA": 1, "C": 2}.get(names[atom_index])
        if slot is not None:
            indices[token, slot] = atom_index  # scalar slot in (tokens, 3)
    return indices  # (tokens, 3)


def _decode_atom_name(chars: Tensor) -> str:
    # chars: (4,) encoded atom-name characters.
    values = chars.detach().cpu().tolist()
    return "".join(" " if value == 0 else chr(int(value) + 32) for value in values).rstrip()


def _kabsch_aligned(predicted: Tensor, target: Tensor) -> Tensor:
    """Return target coordinates aligned onto predicted; both inputs are (points, 3)."""
    if predicted.shape != target.shape or predicted.shape[0] < 3:
        return target  # unchanged input shape on unsupported alignment inputs
    predicted_centered = predicted - predicted.mean(0, keepdim=True)  # (points, 3)
    target_centered = target - target.mean(0, keepdim=True)  # (points, 3)
    covariance = target_centered.transpose(0, 1) @ predicted_centered  # (3, 3)
    left, _, right = torch.linalg.svd(covariance)  # (3, 3), (3,), (3, 3)
    correction = torch.eye(3, device=predicted.device, dtype=predicted.dtype)  # (3, 3)
    correction[-1, -1] = torch.linalg.det(left @ right).sign()  # scalar orientation correction
    # Coordinates are row vectors: target @ rotation aligns to predicted.
    rotation = left @ correction @ right  # (3, 3)
    aligned = target_centered @ rotation + predicted.mean(0, keepdim=True)  # (points, 3)
    return aligned  # (points, 3)


def _chain_assignment(
    predicted: Tensor,
    true: Tensor,
    chain_atoms: list[Tensor],
    chain_sequences: list[str],
    atom_names: list[str],
    atom_mask: Tensor,
    atom_to_token: Tensor,
) -> list[int]:
    """Match equivalent two-chain records to native chain order by CA RMSD."""
    # predicted/true: (atoms, 3); atom_mask/atom_to_token: (atoms,).
    # chain_atoms contains one (chain atoms,) index vector per chain.
    if len(chain_atoms) != 2 or chain_sequences[0] != chain_sequences[1]:
        return [0, 1] if len(chain_atoms) == 2 else list(range(len(chain_atoms)))
    candidates: list[float] = []
    for assignment in ([0, 1], [1, 0]):
        pred_points: list[Tensor] = []
        true_points: list[Tensor] = []
        for native_chain, record_chain in enumerate(assignment):
            native = chain_atoms[native_chain]  # (atoms in native chain,)
            record = chain_atoms[record_chain]  # (atoms in record chain,)
            native_ca = [index for index in native.tolist() if atom_names[index] == "CA"]
            record_ca = [index for index in record.tolist() if atom_names[index] == "CA"]
            for native_index, record_index in zip(native_ca, record_ca, strict=False):
                if (
                    atom_mask[native_index]
                    and torch.isfinite(predicted[native_index]).all()
                    and torch.isfinite(true[record_index]).all()
                ):
                    pred_points.append(predicted[native_index])  # (3,)
                    true_points.append(true[record_index])  # (3,)
        if len(pred_points) < 3:
            candidates.append(float("inf"))
        else:
            aligned = _kabsch_aligned(torch.stack(pred_points), torch.stack(true_points))  # (resolved CA atoms, 3)
            candidates.append(
                float(torch.sqrt((aligned - torch.stack(pred_points)).square().mean()).item())
            )
    return [0, 1] if candidates[0] <= candidates[1] else [1, 0]


def _resolve_ambiguous_atoms(
    predicted: Tensor,
    true: Tensor,
    resolved: Tensor,
    atom_to_token: Tensor,
    atom_names: Sequence[str],
    token_residue_names: Mapping[int, str],
) -> Tensor:
    """Choose crystallographic equivalent-atom labels by predicted distance."""
    # predicted/true: (atoms, 3); resolved/atom_to_token: (atoms,).
    valid = resolved & torch.isfinite(predicted).all(-1) & torch.isfinite(true).all(-1)  # (atoms,)
    if int(valid.sum()) < 3:
        return true  # (atoms, 3)
    aligned_valid = _kabsch_aligned(predicted[valid], true[valid])  # (valid atoms, 3)
    aligned = torch.full_like(true, float("nan"))  # (atoms, 3)
    aligned[valid] = aligned_valid  # update (valid atoms, 3)
    output = true.clone()  # (atoms, 3)
    for token_value, residue_name in token_residue_names.items():
        pairs = _AMBIGUOUS_ATOM_PAIRS.get(residue_name)
        if pairs is None:
            continue
        indices = {
            atom_names[index]: index
            for index, candidate_token in enumerate(atom_to_token.tolist())
            if candidate_token == token_value
        }
        if any(left not in indices or right not in indices for left, right in pairs):
            continue
        pairs = tuple((indices[left], indices[right]) for left, right in pairs)
        if any(not (valid[left] and valid[right]) for left, right in pairs):
            continue
        direct = sum(
            (predicted[left] - aligned[left]).square().sum()
            + (predicted[right] - aligned[right]).square().sum()
            for left, right in pairs
        )  # ()
        swapped = sum(
            (predicted[left] - aligned[right]).square().sum()
            + (predicted[right] - aligned[left]).square().sum()
            for left, right in pairs
        )  # ()
        if swapped < direct:
            for left, right in pairs:
                output[left], output[right] = true[right].clone(), true[left].clone()  # each (3,)
    return output  # (atoms, 3)


def _structure_input(record: Mapping[str, object]) -> StructurePredictionInput:
    chains = record.get("chains")
    if not isinstance(chains, Sequence) or not chains:
        raise ValueError("record chains must be a non-empty sequence")
    inputs: list[ProteinInput] = []
    for chain in chains:
        if not isinstance(chain, Mapping):
            raise ValueError("record chain must be an object")
        chain_id, sequence = chain.get("id"), chain.get("sequence")
        if not isinstance(chain_id, str) or not chain_id:
            raise ValueError("record chain id must be a non-empty string")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("record chain sequence must be a non-empty string")
        inputs.append(ProteinInput(id=chain_id, sequence=sequence))
    return StructurePredictionInput(sequences=inputs)


def _aligned_true_coordinates(
    features: Mapping[str, Tensor],
    chain_infos: Sequence[Any],
    record: Mapping[str, object],
    structure_path: Path,
    predicted_coords: Tensor,
) -> tuple[Tensor, Tensor]:
    """Map atom14 coordinates onto the native padded atom order."""
    # predicted_coords: (atoms, 3); structure arrays use residue and atom14 axes.
    with np.load(structure_path, allow_pickle=False) as arrays:
        coordinates = np.asarray(arrays["coordinates"], dtype=np.float32)  # (residues, 14, 3)
        atom_names = np.asarray(arrays["atom_names"])  # (residues, 14)
        chain_index = np.asarray(arrays["chain_index"])  # (residues,)
        residue_index = np.asarray(arrays["residue_index"])  # (residues,)

    atom_to_token = features["atom_to_token"].reshape(-1).long()  # (atoms,)
    atom_mask = features["atom_attention_mask"].reshape(-1).bool()  # (atoms,)
    chars = features["ref_atom_name_chars"].reshape(-1, 4)  # (atoms, 4)
    decoded_names = [_decode_atom_name(chars[index]) for index in range(chars.shape[0])]
    token_locations: dict[int, tuple[int, int]] = {}
    for chain_number, chain_info in enumerate(chain_infos):
        for token in chain_info.tokens:
            # Native token residues are zero-based; normalized structure records
            # retain one-based residue positions within each complete chain.
            token_locations[int(token.token_index)] = (chain_number, int(token.residue_index) + 1)
    true = torch.full((atom_to_token.numel(), 3), float("nan"), dtype=torch.float32)  # (atoms, 3)
    resolved = torch.zeros(atom_to_token.numel(), dtype=torch.bool)  # (atoms,)
    for atom_index in range(atom_to_token.numel()):
        if not atom_mask[atom_index]:
            continue
        location = token_locations.get(int(atom_to_token[atom_index]))
        if location is None:
            continue
        chain_number, local_residue = location
        # (matching residues,)
        matches = np.flatnonzero((chain_index == chain_number) & (residue_index == local_residue))
        if matches.size != 1:
            continue
        names = atom_names[matches[0]]  # (14,)
        name = _decode_atom_name(chars[atom_index])
        normalized_names = np.asarray(
            [item.decode() if isinstance(item, bytes) else str(item).strip() for item in names]
        )  # (14,)
        name_matches = np.flatnonzero(normalized_names == name)  # (matching atom slots,)
        if name_matches.size != 1:
            continue
        xyz = coordinates[matches[0], name_matches[0]]  # (3,)
        if np.isfinite(xyz).all():
            true[atom_index] = torch.from_numpy(xyz)  # (3,)
            resolved[atom_index] = True  # scalar atom mask entry
    chain_atoms = [
        torch.tensor(
            [
                index
                for index in range(atom_to_token.numel())
                if atom_mask[index]
                and token_locations.get(int(atom_to_token[index]), (-1,))[0] == chain_number
            ],
            dtype=torch.long,
        )
        for chain_number in range(len(chain_infos))
    ]  # per chain (chain atoms,)
    sequences = [str(chain.get("sequence", "")) for chain in record.get("chains", [])]
    assignment = _chain_assignment(
        predicted_coords,
        true,
        chain_atoms,
        sequences,
        decoded_names,
        atom_mask,
        atom_to_token,
    )
    if assignment == [1, 0]:
        reordered = true.clone()  # (atoms, 3)
        reordered[chain_atoms[0]] = true[chain_atoms[1]]  # (chain atoms, 3)
        reordered[chain_atoms[1]] = true[chain_atoms[0]]  # (chain atoms, 3)
        true = reordered  # (atoms, 3)
        resolved = torch.isfinite(true).all(-1) & atom_mask  # (atoms,)
    token_residue_names = {
        int(token.token_index): str(token.residue_name)
        for chain in chain_infos
        for token in chain.tokens
    }
    true = _resolve_ambiguous_atoms(
        predicted_coords,
        true,
        resolved,
        atom_to_token,
        decoded_names,
        token_residue_names,
    )  # (atoms, 3)
    return true, resolved  # (atoms, 3), (atoms,)


def load_folding_model(
    model_id: str, device: str | torch.device = "cuda"
) -> ESMFold2ExperimentalModel:
    """Load a manifest-pinned native ESMFold2 model without remote code."""
    if model_id not in _MODEL_IDS:
        raise ValueError(f"confidence caching supports only {sorted(_MODEL_IDS)}")
    spec = get_model_spec(model_id)
    from fastplms.models.esmfold2.configuration_esmfold2 import ESMFold2Config

    config = ESMFold2Config.from_pretrained(
        spec.confidence_training_base.repo_id,
        revision=spec.confidence_training_base.revision,
        attn_implementation="sdpa",
        esmc_precision="bf16",
    )
    model = ESMFold2ExperimentalModel.from_pretrained(
        spec.confidence_training_base.repo_id,
        revision=spec.confidence_training_base.revision,
        config=config,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        esmc_precision="bf16",
        load_esmc=True,
    )
    model._fastplms_model_id = model_id
    model._fastplms_revision = spec.confidence_training_base.revision
    model._fastplms_repo = spec.confidence_training_base.repo_id
    model._fastplms_pins = ";".join(item.encoded for item in spec.confidence_training_base.files)
    model.set_chunk_size(32)
    return model.to(device).eval().requires_grad_(False)


def _output_tensor(output: Any, name: str) -> Tensor:
    value = output[name] if isinstance(output, Mapping) else getattr(output, name)  # field-specific shape
    if value is None:
        raise RuntimeError(f"model output omitted {name}")
    return value  # shape unchanged; the caller selects and validates the field


def _single_sample_coordinates(prediction: Tensor) -> Tensor:
    # Native outputs use (batch * sample, atom, xyz) or (batch, sample, atom, xyz).
    if prediction.ndim not in (3, 4) or prediction.shape[-1] != 3:
        raise ValueError("Unexpected native coordinate shape")
    coordinates = prediction.reshape(-1, prediction.shape[-2], 3)  # [sample, atom, xyz]
    if coordinates.shape[0] != 1:
        raise ValueError("Each confidence cache requires exactly one target and sample")
    return coordinates  # (1, atoms, 3)


@torch.no_grad()
def cache_target(
    model: ESMFold2ExperimentalModel,
    record: Mapping[str, object],
    data_root: Path,
    output: Path,
    seed: int,
) -> dict[str, object]:
    """Run one frozen fold and write its confidence-head cache."""
    structure_value = record.get("structure_path")
    if not isinstance(structure_value, str) or not structure_value:
        raise ValueError("record structure_path must be a non-empty string")
    structure_path = (data_root / structure_value).resolve()
    if data_root.resolve() not in structure_path.parents:
        raise ValueError("structure_path escapes data_root")
    with structure_path.open("rb") as handle:
        structure_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if structure_digest != record.get("structure_sha256"):
        raise ValueError("Normalized structure differs from the selected dataset hash")
    validate_structure_npz(structure_path)
    prepared, chain_infos = model.prepare_structure_input(_structure_input(record), seed=seed)
    device = next(model.parameters()).device
    prepared = {name: value.to(device) for name, value in prepared.items()}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        result = model(
            **prepared,
            num_loops=3,
            num_sampling_steps=15,
            num_diffusion_samples=1,
            seed=seed,
            calculate_confidence=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden_states = getattr(result, "hidden_states", None)
    if hidden_states is None or len(hidden_states) < 2:
        raise RuntimeError("model output did not include token and pair hidden states")
    prediction = _single_sample_coordinates(
        _output_tensor(result, "sample_atom_coords").detach().float().cpu()
    )  # [1, atom, xyz]
    # Alignment and atom-name bookkeeping are CPU-only. Keep them on the same
    # device even when the frozen fold ran on CUDA.
    prepared_cpu = {name: value.detach().cpu() for name, value in prepared.items()}
    true_coords, resolved_mask = _aligned_true_coordinates(
        prepared_cpu, chain_infos, record, structure_path, prediction[0]
    )  # (atoms, 3), (atoms,)
    tensor_cache: dict[str, Tensor] = {
        "s_inputs": hidden_states[0].detach().float().cpu(),
        "z": hidden_states[1].detach().float().cpu(),
        "x_pred": prediction,
        "true_coords": true_coords,
        "resolved_mask": resolved_mask,
    }
    for name in _FEATURE_KEYS:
        if name not in prepared_cpu:
            raise RuntimeError(f"prepared features omitted required cache field {name!r}")
        tensor_cache[name] = prepared_cpu[name]
    tensor_cache["backbone_indices"] = _backbone_indices(prepared_cpu)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": _CACHE_SCHEMA,
        "model_id": str(getattr(model, "_fastplms_model_id", "")),
        "model_revision": str(getattr(model, "_fastplms_revision", "")),
        "model_repo": str(getattr(model, "_fastplms_repo", "")),
        "model_pins": str(getattr(model, "_fastplms_pins", "")),
        "record_id": str(record.get("id", "")),
        "split": str(record.get("split", "")),
        "record_sha256": _record_hash(record),
        "seed": str(seed),
        "num_loops": "3",
        "num_sampling_steps": "15",
    }
    temporary = output.with_name(f".{output.name}.tmp")
    save_file(tensor_cache, str(temporary), metadata=metadata)
    os.replace(temporary, output)
    return {**metadata, "output": str(output), "resolved_atoms": int(resolved_mask.sum())}


def load_cache(
    path: Path,
    *,
    model_id: str | None = None,
    model_revision: str | None = None,
    model_pins: str | None = None,
    seed: int | None = None,
) -> tuple[dict[str, Tensor], dict[str, str]]:
    """Load and validate one immutable confidence cache."""
    tensors = load_file(str(path), device="cpu")
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("schema") != _CACHE_SCHEMA:
        raise ValueError("unsupported confidence cache schema")
    expected = {
        "model_id": model_id,
        "model_revision": model_revision,
        "model_pins": model_pins,
        "seed": None if seed is None else str(seed),
    }
    for name, value in expected.items():
        if value is not None and metadata.get(name) != value:
            raise ValueError(f"confidence cache metadata mismatch for {name}")
    required = {
        "s_inputs",
        "z",
        "x_pred",
        "true_coords",
        "resolved_mask",
        "backbone_indices",
        *_FEATURE_KEYS,
    }
    missing = sorted(required.difference(tensors))
    if missing:
        raise ValueError(f"confidence cache is missing tensors: {', '.join(missing)}")
    if "relative_position_encoding" in tensors or "token_bonds_encoding" in tensors:
        raise ValueError("cache must not store derived positional or bond embeddings")
    expected_ranks = {
        "s_inputs": 3,
        "z": 4,
        "x_pred": 3,
        "true_coords": 2,
        "resolved_mask": 1,
        "backbone_indices": 2,
    }
    for name, rank in expected_ranks.items():
        if tensors[name].ndim != rank:
            raise ValueError(f"confidence cache tensor {name!r} must have rank {rank}")
    token_mask = tensors["token_attention_mask"].reshape(-1)  # (tokens,)
    atom_mask = tensors["atom_attention_mask"].reshape(-1)  # (atoms,)
    if (
        tensors["s_inputs"].shape[0] != 1
        or tensors["s_inputs"].shape[1] != token_mask.numel()
        or tensors["z"].shape[:3] != (1, token_mask.numel(), token_mask.numel())
    ):
        raise ValueError("confidence cache token state shapes are inconsistent")
    atom_count = atom_mask.numel()
    if tensors["x_pred"].shape != (1, atom_count, 3) or tensors["true_coords"].shape != (
        atom_count,
        3,
    ):
        raise ValueError("confidence cache atom coordinate shapes are inconsistent")
    if tensors["resolved_mask"].shape != (atom_count,):
        raise ValueError("confidence cache resolved_mask shape is inconsistent")
    if tensors["backbone_indices"].shape != (token_mask.numel(), 3):
        raise ValueError("confidence cache backbone_indices shape is inconsistent")
    atom_to_token = tensors["atom_to_token"].reshape(-1)  # (atoms,)
    if atom_to_token.numel() != atom_count or atom_to_token.dtype not in (torch.int32, torch.int64):
        raise ValueError("confidence cache atom_to_token shape or dtype is inconsistent")
    if atom_to_token.numel() and (
        atom_to_token.min() < 0 or atom_to_token.max() >= token_mask.numel()
    ):
        raise ValueError("confidence cache atom_to_token contains an invalid token index")
    return tensors, metadata


def confidence_inputs(model: Any, cache: Mapping[str, Tensor]) -> dict[str, Tensor | int]:
    """Reconstruct native confidence-head inputs from a frozen model cache."""
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = cache["s_inputs"].device
    tensors = {name: value.to(device) for name, value in cache.items()}
    autocast_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
        relative = model.rel_pos(
            residue_index=tensors["residue_index"],
            asym_id=tensors["asym_id"],
            sym_id=tensors["sym_id"],
            entity_id=tensors["entity_id"],
            token_index=tensors["token_index"],
        )  # (1, tokens, tokens, pair channels)
        bonds = model.token_bonds(tensors["token_bonds"].float())  # (1, tokens, tokens, pair channels)
    relative = relative.float()  # (1, tokens, tokens, pair channels)
    bonds = bonds.float()  # (1, tokens, tokens, pair channels)
    return {
        "s_inputs": tensors["s_inputs"],
        "z": tensors["z"],
        "x_pred": tensors["x_pred"],
        "distogram_atom_idx": tensors["distogram_atom_idx"],
        "token_attention_mask": tensors["token_attention_mask"],
        "atom_to_token": tensors["atom_to_token"],
        "atom_attention_mask": tensors["atom_attention_mask"],
        "asym_id": tensors["asym_id"],
        "mol_type": tensors["mol_type"],
        "num_diffusion_samples": 1,
        "relative_position_encoding": relative,
        "token_bonds_encoding": bonds,
    }


__all__ = ["cache_target", "confidence_inputs", "load_cache", "load_folding_model"]
