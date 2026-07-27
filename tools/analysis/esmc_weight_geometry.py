"""Weights-only layer geometry analysis for ESMC-6B and ESMFold2.

The analysis never instantiates a model or executes a forward pass. Large
checkpoint tensors are opened lazily with ``safetensors.safe_open`` and released
after each matrix family is processed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

ESMC_MODEL_ID = "esmc_6b"
ESMFOLD2_MODEL_IDS = (
    "esmfold2",
    "esmfold2_fast",
    "esmfold2_experimental_cutoff2025",
    "esmfold2_experimental_fast_cutoff2025",
)
N_BLOCKS = 80
N_STATES = 81
D_MODEL = 2560
N_HEADS = 40
D_HEAD = 64
FFN_WIDTH = 6912
SPECTRUM_RANKS = (16, 32, 64, 128, 256, 512, 1024, 2048)
ALIGNMENT_RANKS = (16, 32, 64, 128, 256)
ENERGY_TARGETS = (0.90, 0.95, 0.99)
ID_K_VALUES = (10, 20, 50)
ANOMALY_BLOCKS = frozenset({48, 49, 50, 51, 52, 53, 76, 77, 78, 79})
PRIMARY_BLOCKS = frozenset({50, 51})
SCHEMA_VERSION = 1
STAGES = (
    "inventory",
    "spectra",
    "dimension",
    "heads",
    "alignment",
    "compression",
    "report",
)

MatrixFamily = Literal["q", "k", "v", "o", "gate", "value", "down"]
CenterMode = Literal["operator", "rows", "columns"]

MATRIX_FAMILIES: tuple[MatrixFamily, ...] = (
    "q",
    "k",
    "v",
    "o",
    "gate",
    "value",
    "down",
)

_BLOCK_SUFFIXES = {
    "attn_qkv": "attn.layernorm_qkv.weight",
    "attn_input_norm_weight": "attn.layernorm_qkv.layer_norm_weight",
    "attn_input_norm_bias": "attn.layernorm_qkv.layer_norm_bias",
    "attn_output": "attn.out_proj.weight",
    "q_norm_weight": "attn.q_ln.weight",
    "k_norm_weight": "attn.k_ln.weight",
    "ffn_fc1": "ffn.fc1_weight",
    "ffn_down": "ffn.fc2_weight",
    "ffn_norm_weight": "ffn.layer_norm_weight",
    "ffn_norm_bias": "ffn.layer_norm_bias",
}

_EXPECTED_SHAPES = {
    "attn_qkv": (3 * D_MODEL, D_MODEL),
    "attn_input_norm_weight": (D_MODEL,),
    "attn_input_norm_bias": (D_MODEL,),
    "attn_output": (D_MODEL, D_MODEL),
    "q_norm_weight": (D_MODEL,),
    "k_norm_weight": (D_MODEL,),
    "ffn_fc1": (2 * FFN_WIDTH, D_MODEL),
    "ffn_down": (D_MODEL, FFN_WIDTH),
    "ffn_norm_weight": (D_MODEL,),
    "ffn_norm_bias": (D_MODEL,),
}

_FOLD_SUFFIXES = {
    "mixing_logits": "language_model.base_z_combine",
    "projection_norm_weight": "language_model.base_z_linear.0.weight",
    "projection_norm_bias": "language_model.base_z_linear.0.bias",
    "projection": "language_model.base_z_linear.1.weight",
}

_PALETTE = {
    "blue": "#3568A6",
    "blue_light": "#9BB7D6",
    "gold": "#C69C3C",
    "orange": "#C96B36",
    "ink": "#20242A",
    "muted": "#68717D",
    "grid": "#D9DEE5",
    "paper": "#FCFCFD",
}

_FAMILY_STYLES = {
    "q": (_PALETTE["blue"], "-"),
    "k": (_PALETTE["blue"], "--"),
    "v": (_PALETTE["blue"], ":"),
    "o": (_PALETTE["orange"], "-"),
    "gate": (_PALETTE["gold"], "-"),
    "value": (_PALETTE["gold"], "--"),
    "down": (_PALETTE["ink"], "-."),
}


class AnalysisError(RuntimeError):
    """Raised when an analysis contract or numerical invariant fails."""


@dataclass(frozen=True, slots=True)
class TensorRecord:
    """One tensor in a local sharded or unsharded safetensors checkpoint."""

    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    block: int | None
    role: str | None


@dataclass(frozen=True, slots=True)
class FoldWeights:
    """Weights required from one ESMFold2 checkpoint."""

    label: str
    path: Path
    mixing_logits: Any
    mixing_weights: Any
    projection: Any
    norm_weight: Any
    norm_bias: Any


@dataclass(frozen=True, slots=True)
class SpectrumResult:
    """One complete matrix spectrum and its numerical diagnostics."""

    singular_values: Any
    negative_mass: float
    tolerance: float
    stored_rank: int
    fp32_rank: int
    condition_estimate: float | None


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise AnalysisError(
            "This analysis requires PyTorch. Install the FastPLMs core dependencies."
        ) from error
    return torch


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise AnalysisError("This analysis requires NumPy.") from error
    return np


def _require_safetensors() -> Any:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise AnalysisError(
            "This analysis requires safetensors. Install the FastPLMs core dependencies."
        ) from error
    return safe_open


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        fields = ["empty"]
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fields})
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def state_for_block_output(block: int) -> int:
    """Return the hidden-state index produced by an ESMC block."""

    if block < 0 or block >= N_BLOCKS:
        raise ValueError(f"Block index must be in [0, {N_BLOCKS - 1}], got {block}")
    return block + 1


def block_producing_state(state: int) -> int | None:
    """Return the block producing a hidden state, or ``None`` for state zero."""

    if state < 0 or state >= N_STATES:
        raise ValueError(f"State index must be in [0, {N_STATES - 1}], got {state}")
    return None if state == 0 else state - 1


def _block_key(block: int, role: str) -> str:
    try:
        suffix = _BLOCK_SUFFIXES[role]
    except KeyError as error:
        raise KeyError(f"Unknown ESMC block role: {role!r}") from error
    return f"esmc.transformer.blocks.{block}.{suffix}"


def _tensor_nbytes(dtype: str, shape: Sequence[int]) -> int:
    bytes_per_value = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "I16": 2,
        "U16": 2,
        "F16": 2,
        "BF16": 2,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }.get(dtype)
    if bytes_per_value is None:
        raise AnalysisError(f"Unsupported safetensors dtype: {dtype!r}")
    return math.prod(shape) * bytes_per_value


def _parse_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        encoded_length = handle.read(8)
        if len(encoded_length) != 8:
            raise AnalysisError(f"Invalid safetensors header in {path}")
        header_length = int.from_bytes(encoded_length, "little")
        if header_length <= 0 or header_length > 128 * 1024 * 1024:
            raise AnalysisError(f"Unsafe safetensors header length in {path}: {header_length}")
        payload = handle.read(header_length)
    try:
        header = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"Invalid safetensors JSON header in {path}") from error
    if not isinstance(header, dict):
        raise AnalysisError(f"Safetensors header in {path} is not an object")
    return cast(dict[str, Any], header)


def _role_from_name(name: str) -> tuple[int | None, str | None]:
    prefix = "esmc.transformer.blocks."
    if not name.startswith(prefix):
        return None, None
    remainder = name[len(prefix) :]
    encoded_block, separator, suffix = remainder.partition(".")
    if not separator or not encoded_block.isdigit():
        return None, None
    block = int(encoded_block)
    for role, expected_suffix in _BLOCK_SUFFIXES.items():
        if suffix == expected_suffix:
            return block, role
    return block, None


class SafetensorCheckpoint:
    """Lazy reader and header inventory for a local safetensors checkpoint."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        if self.root.is_file():
            files = [self.root]
            self.root = self.root.parent
        elif self.root.is_dir():
            files = sorted(self.root.glob("*.safetensors"))
        else:
            raise AnalysisError(f"Checkpoint does not exist: {self.root}")
        if not files:
            raise AnalysisError(f"No safetensors files found in {self.root}")
        records: dict[str, TensorRecord] = {}
        for file in files:
            header = _parse_safetensors_header(file)
            for name, descriptor in header.items():
                if name == "__metadata__":
                    continue
                if not isinstance(descriptor, dict):
                    raise AnalysisError(f"Invalid tensor descriptor for {name!r}")
                dtype = str(descriptor.get("dtype"))
                shape = tuple(int(item) for item in descriptor.get("shape", ()))
                if name in records:
                    raise AnalysisError(f"Duplicate tensor {name!r} across checkpoint shards")
                block, role = _role_from_name(name)
                records[name] = TensorRecord(
                    name=name,
                    file=file.name,
                    dtype=dtype,
                    shape=shape,
                    nbytes=_tensor_nbytes(dtype, shape),
                    block=block,
                    role=role,
                )
        self.records = records

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.records))

    def record(self, name: str) -> TensorRecord:
        try:
            return self.records[name]
        except KeyError as error:
            raise AnalysisError(f"Checkpoint is missing tensor {name!r}") from error

    def tensor(self, name: str) -> Any:
        record = self.record(name)
        safe_open = _require_safetensors()
        with safe_open(self.root / record.file, framework="pt", device="cpu") as reader:
            return reader.get_tensor(name)

    def find_unique_suffix(self, suffix: str) -> str:
        matches = [name for name in self.records if name.endswith(suffix)]
        if len(matches) != 1:
            raise AnalysisError(
                f"Expected one tensor ending in {suffix!r}, found {len(matches)}: {matches[:5]}"
            )
        return matches[0]


def validate_esmc_inventory(checkpoint: SafetensorCheckpoint) -> list[TensorRecord]:
    """Validate all required ESMC block tensors and return their records."""

    selected: list[TensorRecord] = []
    for block in range(N_BLOCKS):
        for role, expected_shape in _EXPECTED_SHAPES.items():
            record = checkpoint.record(_block_key(block, role))
            if record.shape != expected_shape:
                raise AnalysisError(
                    f"{record.name} has shape {record.shape}, expected {expected_shape}"
                )
            selected.append(record)
    final_norm_matches = [
        record
        for name, record in checkpoint.records.items()
        if name.endswith("transformer.norm.weight")
    ]
    if len(final_norm_matches) != 1 or final_norm_matches[0].shape != (D_MODEL,):
        raise AnalysisError("Checkpoint must contain one 2560-element final transformer norm")
    selected.extend(final_norm_matches)
    return selected


def _split_matrix(family: MatrixFamily, qkv: Any, fc1: Any, output: Any, down: Any) -> Any:
    if family == "q":
        return qkv[:D_MODEL]
    if family == "k":
        return qkv[D_MODEL : 2 * D_MODEL]
    if family == "v":
        return qkv[2 * D_MODEL :]
    if family == "o":
        return output
    if family == "gate":
        return fc1[:FFN_WIDTH]
    if family == "value":
        return fc1[FFN_WIDTH:]
    if family == "down":
        return down
    raise AssertionError(f"Unhandled matrix family {family!r}")


def load_block_matrix(
    checkpoint: SafetensorCheckpoint,
    block: int,
    family: MatrixFamily,
) -> Any:
    """Load one logical block matrix, splitting fused QKV and SwiGLU weights."""

    if family in {"q", "k", "v"}:
        qkv = checkpoint.tensor(_block_key(block, "attn_qkv"))
        return _split_matrix(family, qkv, None, None, None)
    if family in {"gate", "value"}:
        fc1 = checkpoint.tensor(_block_key(block, "ffn_fc1"))
        return _split_matrix(family, None, fc1, None, None)
    if family == "o":
        return checkpoint.tensor(_block_key(block, "attn_output"))
    return checkpoint.tensor(_block_key(block, "ffn_down"))


def _device(value: str) -> Any:
    torch = _require_torch()
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as error:
        raise AnalysisError(f"Invalid device {value!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise AnalysisError(f"Only CPU and CUDA devices are supported, got {device.type!r}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise AnalysisError(f"CUDA device {device} was requested but CUDA is unavailable")
    return device


def gram_spectrum(
    matrix: Any,
    *,
    center: CenterMode,
    device: Any,
    accumulation_dtype: str = "float64",
) -> SpectrumResult:
    """Compute a complete spectrum from the smaller Gram matrix."""

    torch = _require_torch()
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError(f"Expected a non-empty matrix, got shape {tuple(matrix.shape)}")
    dtype = torch.float64 if accumulation_dtype == "float64" else torch.float32
    value = matrix.detach().to(device=device, dtype=dtype)
    if center == "rows":
        value = (value - value.mean(dim=0, keepdim=True)) / math.sqrt(
            max(1, value.shape[0] - 1)
        )
    elif center == "columns":
        value = (value - value.mean(dim=1, keepdim=True)) / math.sqrt(
            max(1, value.shape[1] - 1)
        )
    elif center != "operator":
        raise ValueError(f"Unknown centering mode: {center!r}")
    gram = value.mT @ value if value.shape[0] >= value.shape[1] else value @ value.mT
    eigenvalues = torch.linalg.eigvalsh(gram)
    largest = float(eigenvalues[-1].item()) if eigenvalues.numel() else 0.0
    epsilon = torch.finfo(dtype).eps
    tolerance = (
        10.0
        * epsilon
        * eigenvalues.numel()
        * max(largest, torch.finfo(dtype).tiny)
    )
    minimum = float(eigenvalues[0].item()) if eigenvalues.numel() else 0.0
    if minimum < -tolerance:
        raise AnalysisError(
            f"Gram matrix is materially non-PSD: minimum={minimum:.6g}, "
            f"tolerance={tolerance:.6g}"
        )
    negative = torch.clamp(-eigenvalues, min=0)
    positive = torch.clamp(eigenvalues, min=0)
    positive_mass = float(positive.sum().item())
    negative_mass = (
        float(negative.sum().item()) / positive_mass if positive_mass > 0 else 0.0
    )
    singular_values = torch.sqrt(positive).flip(0).to("cpu")
    stored_rank = int((positive > tolerance).sum().item())
    if singular_values.numel() and float(singular_values[0]) > 0:
        fp32_floor = (
            torch.finfo(torch.float32).eps
            * max(matrix.shape)
            * float(singular_values[0])
        )
        fp32_rank = int((singular_values > fp32_floor).sum().item())
    else:
        fp32_rank = 0
    condition: float | None = None
    if fp32_rank:
        resolved = singular_values[:fp32_rank]
        if float(resolved[-1]) > 0:
            condition = float(resolved[0] / resolved[-1])
    del value, gram, eigenvalues, positive, negative
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    return SpectrumResult(
        singular_values=singular_values,
        negative_mass=negative_mass,
        tolerance=float(tolerance),
        stored_rank=stored_rank,
        fp32_rank=fp32_rank,
        condition_estimate=condition,
    )


def spectrum_metrics(
    singular_values: Any,
    shape: Sequence[int],
    *,
    center: CenterMode = "operator",
) -> dict[str, float | int | None]:
    """Calculate complete rank, concentration, and low-rank storage metrics."""

    torch = _require_torch()
    values = singular_values.detach().to(dtype=torch.float64, device="cpu")
    energy = values.square()
    total = float(energy.sum().item())
    rows, columns = int(shape[0]), int(shape[1])
    mean_parameters = columns if center == "rows" else rows if center == "columns" else 0
    break_even = (rows * columns - mean_parameters) / (rows + columns)
    if total <= 0 or not values.numel():
        result: dict[str, float | int | None] = {
            "frobenius_norm": 0.0,
            "spectral_norm": 0.0,
            "stable_rank": 0.0,
            "participation_ratio": 0.0,
            "effective_rank": 0.0,
            "nuclear_frobenius_ratio": 0.0,
            "leading_energy_fraction": 0.0,
            "low_rank_break_even_rank": float(break_even),
            "center_mean_parameters": mean_parameters,
        }
        for target in ENERGY_TARGETS:
            result[f"rank_{int(target * 100)}"] = 0
        for rank in SPECTRUM_RANKS:
            stored_fraction = (
                rank * (rows + columns) + mean_parameters
            ) / (rows * columns)
            result[f"rank_{rank}_relative_error"] = 0.0
            result[f"rank_{rank}_storage_fraction"] = stored_fraction
            result[f"rank_{rank}_compression_factor"] = (
                1.0 / stored_fraction if stored_fraction > 0 else None
            )
            result[f"rank_{rank}_is_compression"] = int(stored_fraction < 1.0)
        return result
    probabilities = energy / total
    nonzero = probabilities > 0
    cumulative = torch.cumsum(probabilities, dim=0)
    result = {
        "frobenius_norm": math.sqrt(total),
        "spectral_norm": float(values[0]),
        "stable_rank": total / float(energy[0]),
        "participation_ratio": total**2 / float(energy.square().sum()),
        "effective_rank": float(
            torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
        ),
        "nuclear_frobenius_ratio": float(values.sum()) / math.sqrt(total),
        "leading_energy_fraction": float(probabilities[0]),
    }
    for target in ENERGY_TARGETS:
        result[f"rank_{int(target * 100)}"] = int(
            torch.searchsorted(cumulative, target).item() + 1
        )
    result["low_rank_break_even_rank"] = float(break_even)
    result["center_mean_parameters"] = mean_parameters
    for rank in SPECTRUM_RANKS:
        bounded = min(rank, len(values))
        retained = float(energy[:bounded].sum()) / total
        stored_fraction = (
            rank * (rows + columns) + mean_parameters
        ) / (rows * columns)
        result[f"rank_{rank}_relative_error"] = math.sqrt(max(0.0, 1.0 - retained))
        result[f"rank_{rank}_storage_fraction"] = stored_fraction
        result[f"rank_{rank}_compression_factor"] = (
            1.0 / stored_fraction if stored_fraction > 0 else None
        )
        result[f"rank_{rank}_is_compression"] = int(stored_fraction < 1.0)
    return result


def direct_svd_validation(matrix: Any, reference: Any, device: Any) -> dict[str, float]:
    """Validate leading spectrum and energy ranks against a direct FP32 SVD."""

    torch = _require_torch()
    direct = torch.linalg.svdvals(matrix.detach().to(device=device, dtype=torch.float32)).cpu()
    compared = min(64, direct.numel(), reference.numel())
    denominator = reference[:compared].abs().clamp_min(torch.finfo(torch.float32).tiny)
    relative = (direct[:compared] - reference[:compared]).abs() / denominator
    direct_metrics = spectrum_metrics(direct, matrix.shape)
    reference_metrics = spectrum_metrics(reference, matrix.shape)
    return {
        "leading_64_max_relative_error": float(relative.max()) if compared else 0.0,
        "total_energy_relative_error": abs(
            float(direct.square().sum()) - float(reference.square().sum())
        )
        / max(float(reference.square().sum()), torch.finfo(torch.float64).tiny),
        "rank_90_absolute_difference": abs(
            int(direct_metrics["rank_90"]) - int(reference_metrics["rank_90"])
        ),
        "rank_95_absolute_difference": abs(
            int(direct_metrics["rank_95"]) - int(reference_metrics["rank_95"])
        ),
        "rank_99_absolute_difference": abs(
            int(direct_metrics["rank_99"]) - int(reference_metrics["rank_99"])
        ),
    }


def _deterministic_points(points: Any, maximum: int) -> Any:
    if maximum <= 0 or points.shape[0] <= maximum:
        return points
    torch = _require_torch()
    indices = torch.linspace(0, points.shape[0] - 1, maximum, dtype=torch.float64)
    return points[indices.round().to(torch.long)]


def nearest_neighbor_distances(
    points: Any,
    *,
    k: int,
    normalized: bool,
    device: Any,
    chunk_size: int = 128,
    maximum_points: int = 0,
) -> Any:
    """Return exact k-nearest distances with explicit self-neighbor removal."""

    torch = _require_torch()
    if points.ndim != 2:
        raise ValueError(f"Expected a point matrix, got shape {tuple(points.shape)}")
    points = _deterministic_points(points.detach(), maximum_points)
    if points.shape[0] <= k:
        raise ValueError(f"Need more than {k} points, found {points.shape[0]}")
    value = points.to(device=device, dtype=torch.float32)
    if normalized:
        norms = torch.linalg.vector_norm(value, dim=1, keepdim=True)
        value = value / norms.clamp_min(torch.finfo(value.dtype).tiny)
    result = torch.empty((value.shape[0], k), dtype=torch.float32, device="cpu")
    for start in range(0, value.shape[0], chunk_size):
        stop = min(value.shape[0], start + chunk_size)
        distances = torch.cdist(value[start:stop], value)
        query = torch.arange(stop - start, device=device)
        self_indices = torch.arange(start, stop, device=device)
        distances[query, self_indices] = torch.inf
        result[start:stop] = torch.topk(
            distances,
            k=k,
            dim=1,
            largest=False,
            sorted=True,
        ).values.cpu()
        del distances
    del value
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    return result


def intrinsic_dimension_metrics(
    neighbors: Any,
    *,
    trim_fraction: float = 0.01,
) -> dict[str, float | int | None]:
    """Calculate guarded TwoNN and pooled/local Levina-Bickel estimates."""

    torch = _require_torch()
    distances = neighbors.detach().to(dtype=torch.float64, device="cpu")
    if distances.ndim != 2 or distances.shape[1] < 2:
        raise ValueError("At least two nearest-neighbor distances are required")
    scale = float(torch.median(distances[:, -1]))
    duplicate_tolerance = max(torch.finfo(torch.float64).eps * max(scale, 1.0) * 100, 1e-15)
    duplicate = distances[:, 0] <= duplicate_tolerance
    valid_two = (~duplicate) & torch.isfinite(distances[:, 1]) & (distances[:, 1] > 0)
    ratios = distances[valid_two, 1] / distances[valid_two, 0]
    ratios = ratios[torch.isfinite(ratios) & (ratios >= 1)]
    trimmed_count = 0
    if ratios.numel() and trim_fraction > 0:
        keep = max(1, math.floor((1.0 - trim_fraction) * ratios.numel()))
        ratios = torch.sort(ratios).values[:keep]
        trimmed_count = int(valid_two.sum()) - keep
    denominator = float(torch.log(ratios).sum()) if ratios.numel() else 0.0
    twonn = float(ratios.numel()) / denominator if denominator > 0 else None
    result: dict[str, float | int | None] = {
        "n_points": int(distances.shape[0]),
        "duplicate_fraction": float(duplicate.float().mean()),
        "twonn_valid_points": int(ratios.numel()),
        "twonn_trimmed_points": trimmed_count,
        "twonn_dimension": twonn,
        "nearest_distance_mean": float(distances[:, 0].mean()),
        "nearest_distance_sd": float(distances[:, 0].std(correction=1)),
        "nearest_distance_cv": (
            float(distances[:, 0].std(correction=1) / distances[:, 0].mean())
            if float(distances[:, 0].mean()) > 0
            else None
        ),
    }
    for k in ID_K_VALUES:
        if distances.shape[1] < k:
            result[f"mle_{k}_dimension"] = None
            result[f"mle_{k}_local_median"] = None
            result[f"mle_{k}_local_iqr"] = None
            result[f"mle_{k}_local_mad"] = None
            continue
        selected = distances[:, :k]
        radius = selected[:, k - 1]
        valid = (
            (~duplicate)
            & torch.isfinite(radius)
            & (radius > 0)
            & torch.all(selected[:, : k - 1] > duplicate_tolerance, dim=1)
        )
        logs = torch.log(radius[valid, None] / selected[valid, : k - 1])
        row_denominators = logs.sum(dim=1)
        valid_denominators = row_denominators > torch.finfo(torch.float64).eps
        row_denominators = row_denominators[valid_denominators]
        pooled_denominator = float(row_denominators.sum())
        count = row_denominators.numel()
        pooled = count * (k - 2) / pooled_denominator if pooled_denominator > 0 else None
        local = (k - 2) / row_denominators if count else torch.empty(0)
        if local.numel():
            q1, median, q3 = torch.quantile(
                local,
                torch.tensor([0.25, 0.50, 0.75], dtype=torch.float64),
            )
            mad = torch.median((local - median).abs())
            result[f"mle_{k}_local_median"] = float(median)
            result[f"mle_{k}_local_iqr"] = float(q3 - q1)
            result[f"mle_{k}_local_mad"] = float(mad)
        else:
            result[f"mle_{k}_local_median"] = None
            result[f"mle_{k}_local_iqr"] = None
            result[f"mle_{k}_local_mad"] = None
        result[f"mle_{k}_dimension"] = pooled
    return result


def orthonormal_basis(
    matrix: Any,
    *,
    side: Literal["row", "column"],
    rank: int,
    device: Any | None = None,
) -> Any:
    """Return a leading singular subspace basis in residual coordinates."""

    torch = _require_torch()
    if side not in {"row", "column"}:
        raise ValueError(f"Unknown subspace side {side!r}")
    value = matrix.detach().to(device=device or matrix.device, dtype=torch.float64)
    maximum_rank = min(value.shape)
    if rank >= maximum_rank:
        vectors = value.mT if side == "row" else value
        basis, _ = torch.linalg.qr(vectors, mode="reduced")
        return basis[:, :maximum_rank]
    if side == "row":
        gram = value.mT @ value
    elif side == "column":
        gram = value @ value.mT
    eigenvalues, vectors = torch.linalg.eigh(gram)
    rank = min(rank, vectors.shape[1])
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    return vectors[:, order]


def subspace_metrics(left: Any, right: Any) -> dict[str, float]:
    """Return normalized chordal affinity and principal-angle summaries."""

    torch = _require_torch()
    left_value = left.detach().to(dtype=torch.float64)
    right_value = right.detach().to(dtype=torch.float64)
    cosines = torch.linalg.svdvals(left_value.mT @ right_value).clamp(0, 1)
    angles = torch.rad2deg(torch.acos(cosines))
    denominator = max(1, min(left_value.shape[1], right_value.shape[1]))
    return {
        "normalized_overlap": float(cosines.square().sum() / denominator),
        "minimum_angle_degrees": float(angles.min()) if angles.numel() else 90.0,
        "median_angle_degrees": float(angles.median()) if angles.numel() else 90.0,
        "maximum_angle_degrees": float(angles.max()) if angles.numel() else 90.0,
    }


def subspace_overlap(left: Any, right: Any) -> float:
    """Return normalized chordal affinity without computing principal angles."""

    torch = _require_torch()
    left_value = left.detach().to(dtype=torch.float64)
    right_value = right.detach().to(dtype=torch.float64)
    denominator = max(1, min(left_value.shape[1], right_value.shape[1]))
    return float((left_value.mT @ right_value).square().sum() / denominator)


def vector_metrics(vector: Any) -> dict[str, float | int | None]:
    """Return scale and tail statistics for a normalization vector."""

    torch = _require_torch()
    value = vector.detach().to(dtype=torch.float64, device="cpu").flatten()
    mean = float(value.mean())
    standard_deviation = float(value.std(correction=1))
    median = float(value.median())
    mad = float((value - median).abs().median())
    q01, q05, q95, q99 = torch.quantile(
        value,
        torch.tensor([0.01, 0.05, 0.95, 0.99], dtype=torch.float64),
    )
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "coefficient_of_variation": standard_deviation / abs(mean) if mean else None,
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "median": median,
        "mad": mad,
        "q01": float(q01),
        "q05": float(q05),
        "q95": float(q95),
        "q99": float(q99),
        "amplified_fraction": float((value > mean + 3 * standard_deviation).float().mean()),
        "suppressed_fraction": float((value < mean - 3 * standard_deviation).float().mean()),
    }


def reconstruction_metrics(original: Any, reconstructed: Any) -> dict[str, float]:
    torch = _require_torch()
    source = original.detach().to(dtype=torch.float64)
    target = reconstructed.detach().to(dtype=torch.float64)
    source_norm = torch.linalg.vector_norm(source)
    target_norm = torch.linalg.vector_norm(target)
    error = torch.linalg.vector_norm(source - target)
    denominator = max(float(source_norm), torch.finfo(torch.float64).tiny)
    cosine_denominator = max(
        float(source_norm * target_norm),
        torch.finfo(torch.float64).tiny,
    )
    return {
        "relative_frobenius_error": float(error) / denominator,
        "flattened_cosine": float((source * target).sum()) / cosine_denominator,
        "mse": float((source - target).square().mean()),
    }


def symmetric_row_quantize(matrix: Any, bits: Literal[4, 8]) -> tuple[Any, dict[str, float]]:
    """Simulate deterministic packed INT4 or INT8 symmetric row quantization."""

    torch = _require_torch()
    if bits not in {4, 8}:
        raise ValueError("Only INT4 and INT8 are supported")
    value = matrix.detach().to(dtype=torch.float32)
    maximum = 127 if bits == 8 else 7
    scales = value.abs().amax(dim=1, keepdim=True) / maximum
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = torch.clamp(torch.round(value / scales), -maximum, maximum)
    reconstructed = quantized * scales
    metrics = reconstruction_metrics(value, reconstructed)
    matrix_bits = value.numel() * bits
    scale_bits = value.shape[0] * 32
    metrics["storage_fraction"] = (matrix_bits + scale_bits) / (value.numel() * 32)
    metrics["bits"] = float(bits)
    return reconstructed, metrics


def magnitude_prune(matrix: Any, sparsity: float) -> tuple[Any, dict[str, float]]:
    """Apply deterministic global magnitude pruning with flattened-index tie breaks."""

    torch = _require_torch()
    if not 0 <= sparsity < 1:
        raise ValueError("Sparsity must be in [0, 1)")
    value = matrix.detach().to(dtype=torch.float32)
    flat = value.flatten()
    remove = math.floor(flat.numel() * sparsity)
    mask = torch.ones(flat.numel(), dtype=torch.bool, device=flat.device)
    if remove:
        order = torch.argsort(flat.abs(), stable=True)
        mask[order[:remove]] = False
    reconstructed = torch.where(mask.reshape(value.shape), value, 0)
    metrics = reconstruction_metrics(value, reconstructed)
    metrics["requested_sparsity"] = sparsity
    metrics["actual_sparsity"] = float((reconstructed == 0).float().mean())
    metrics["density"] = 1.0 - metrics["actual_sparsity"]
    metrics["ideal_value_storage_fraction"] = metrics["density"]
    return reconstructed, metrics


def structured_two_of_four(matrix: Any) -> tuple[Any, dict[str, float]]:
    """Retain exactly two entries in each contiguous input-dimension group of four."""

    torch = _require_torch()
    value = matrix.detach().to(dtype=torch.float32)
    if value.shape[1] % 4:
        raise ValueError("2:4 sparsity requires the input dimension to be divisible by four")
    groups = value.reshape(value.shape[0], -1, 4)
    order = torch.argsort(groups.abs(), dim=2, descending=True, stable=True)
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(2, order[:, :, :2], True)
    reconstructed = torch.where(mask, groups, 0).reshape(value.shape)
    metrics = reconstruction_metrics(value, reconstructed)
    metrics["requested_sparsity"] = 0.5
    metrics["actual_sparsity"] = float((reconstructed == 0).float().mean())
    metrics["density"] = 1.0 - metrics["actual_sparsity"]
    metrics["ideal_value_storage_fraction"] = 0.5
    return reconstructed, metrics


def power_spectral_norm(matrix: Any, *, iterations: int = 30, seed: int = 0) -> float:
    """Estimate spectral norm with deterministic, convergence-bounded power iteration."""

    torch = _require_torch()
    value = matrix.detach().to(dtype=torch.float32)
    generator = torch.Generator(device=value.device).manual_seed(seed)
    vector = torch.randn(value.shape[1], generator=generator, device=value.device)
    vector = vector / torch.linalg.vector_norm(vector).clamp_min(torch.finfo(value.dtype).tiny)
    previous = 0.0
    estimate = 0.0
    for _ in range(iterations):
        left = value @ vector
        norm_left = torch.linalg.vector_norm(left)
        if float(norm_left) == 0:
            return 0.0
        right = value.mT @ (left / norm_left)
        norm_right = torch.linalg.vector_norm(right)
        if float(norm_right) == 0:
            return 0.0
        vector = right / norm_right
        estimate = float(norm_left)
        if previous and abs(estimate - previous) <= 1e-6 * max(estimate, 1.0):
            break
        previous = estimate
    return estimate


def spectral_distortion(
    original: Any,
    reconstructed: Any,
    *,
    original_spectral_norm: float | None = None,
) -> float:
    denominator = (
        power_spectral_norm(original)
        if original_spectral_norm is None
        else original_spectral_norm
    )
    if denominator == 0:
        return 0.0
    return power_spectral_norm(original - reconstructed) / denominator


def parse_fold_argument(value: str) -> tuple[str, Path]:
    label, separator, encoded_path = value.partition("=")
    if not separator or not label.strip() or not encoded_path.strip():
        raise argparse.ArgumentTypeError(
            "--esmfold2-checkpoint must use LABEL=/path/to/checkpoint"
        )
    return label.strip(), Path(encoded_path).expanduser().resolve()


def load_fold_weights(label: str, path: Path) -> FoldWeights:
    torch = _require_torch()
    checkpoint = SafetensorCheckpoint(path)
    names = {
        role: checkpoint.find_unique_suffix(suffix)
        for role, suffix in _FOLD_SUFFIXES.items()
    }
    logits = checkpoint.tensor(names["mixing_logits"]).to(dtype=torch.float64)
    projection = checkpoint.tensor(names["projection"]).to(dtype=torch.float64)
    norm_weight = checkpoint.tensor(names["projection_norm_weight"]).to(dtype=torch.float64)
    norm_bias = checkpoint.tensor(names["projection_norm_bias"]).to(dtype=torch.float64)
    if logits.shape != (N_STATES,):
        raise AnalysisError(f"{label}: mixing logits have shape {tuple(logits.shape)}")
    if projection.shape != (256, D_MODEL):
        raise AnalysisError(f"{label}: folding projection has shape {tuple(projection.shape)}")
    if norm_weight.shape != (D_MODEL,) or norm_bias.shape != (D_MODEL,):
        raise AnalysisError(f"{label}: folding LayerNorm has incompatible shape")
    return FoldWeights(
        label=label,
        path=path,
        mixing_logits=logits,
        mixing_weights=torch.softmax(logits, dim=0),
        projection=projection,
        norm_weight=norm_weight,
        norm_bias=norm_bias,
    )


def projection_bases(fold: FoldWeights, device: Any | None = None) -> dict[str, Any]:
    """Return raw and channel-scaled weights-only projection row spaces."""

    return {
        "raw": orthonormal_basis(
            fold.projection,
            side="row",
            rank=256,
            device=device,
        ),
        "layernorm_scaled_approximation": orthonormal_basis(
            fold.projection * fold.norm_weight.unsqueeze(0),
            side="row",
            rank=256,
            device=device,
        ),
    }


class AnalysisRun:
    """Fingerprint-checked, resumable output directory."""

    def __init__(self, output_dir: Path, configuration: Mapping[str, Any], resume: bool):
        self.output_dir = output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.configuration = dict(configuration)
        self.fingerprint = _fingerprint(self.configuration)
        self.resume = resume
        self.progress_dir = self.output_dir / ".progress"
        self.progress_dir.mkdir(exist_ok=True)

    def result_path(self, stage: str, key: str) -> Path:
        return self.progress_dir / stage / f"{key}.json"

    def is_complete(self, stage: str, key: str) -> bool:
        path = self.result_path(stage, key)
        if not self.resume or not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("analysis_fingerprint") != self.fingerprint:
            return False
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            return False
        for relative, expected in artifacts.items():
            artifact = self.output_dir / relative
            if not artifact.is_file() or _sha256_file(artifact) != expected:
                return False
        return True

    def write_result(
        self,
        stage: str,
        key: str,
        payload: Mapping[str, Any],
        artifacts: Sequence[Path] = (),
    ) -> None:
        hashes = {
            str(path.relative_to(self.output_dir)): _sha256_file(path)
            for path in artifacts
        }
        _atomic_json(
            self.result_path(stage, key),
            {
                "schema_version": SCHEMA_VERSION,
                "analysis_fingerprint": self.fingerprint,
                "stage": stage,
                "key": key,
                "artifacts": hashes,
                "payload": payload,
            },
        )

    def stage_rows(self, stage: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        stage_dir = self.progress_dir / stage
        if not stage_dir.is_dir():
            return rows
        for path in sorted(stage_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("analysis_fingerprint") != self.fingerprint:
                continue
            value = payload.get("payload")
            if isinstance(value, dict):
                rows.append(cast(dict[str, Any], value))
        return rows


def _registry_sources() -> tuple[Any, tuple[Any, ...]]:
    try:
        from fastplms.registry import get_model_registry
    except ImportError as error:
        raise AnalysisError(
            "FastPLMs must be installed, or run with PYTHONPATH=src from the repository root."
        ) from error
    registry = get_model_registry()
    esmc = registry[ESMC_MODEL_ID].official
    folds = tuple(registry[model_id].official for model_id in ESMFOLD2_MODEL_IDS)
    return esmc, folds


def download_esmc_checkpoint(cache_dir: Path) -> Path:
    """Download the pinned official ESMC weights into an explicit local directory."""

    esmc, _ = _registry_sources()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise AnalysisError("Pinned download requires huggingface-hub.") from error
    weight_files = tuple(
        item.path for item in esmc.files if item.path.endswith(".safetensors")
    )
    destination = (
        cache_dir.expanduser().resolve()
        / f"{esmc.repo_id.replace('/', '--')}--{esmc.revision}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=esmc.repo_id,
        revision=esmc.revision,
        allow_patterns=[*weight_files, "model.safetensors.index.json", "config.json"],
        local_dir=destination,
    )
    verify_esmc_files(destination)
    return destination


def _hub_range(
    url: str,
    start: int,
    stop: int,
) -> tuple[bytes, dict[str, str]]:
    """Read an inclusive byte range from a pinned Hub URL and require HTTP 206."""

    headers = {"Range": f"bytes={start}-{stop}", "User-Agent": "fastplms-weight-geometry/1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status = getattr(response, "status", None)
            if status != 206:
                raise AnalysisError(
                    f"Hub server did not honor byte range {start}-{stop}; status={status}"
                )
            payload = response.read(stop - start + 1)
            metadata = {
                name: value
                for name, value in response.headers.items()
                if name.lower() in {"etag", "x-linked-etag", "x-repo-commit", "content-range"}
            }
    except urllib.error.URLError as error:
        raise AnalysisError(f"Failed pinned Hub range request: {error.reason}") from error
    if len(payload) != stop - start + 1:
        raise AnalysisError(
            f"Short Hub range response: expected {stop - start + 1} bytes, got {len(payload)}"
        )
    return payload, metadata


def _torch_dtype(encoded: str) -> Any:
    torch = _require_torch()
    mapping = {
        "F64": torch.float64,
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I64": torch.int64,
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    try:
        return mapping[encoded]
    except KeyError as error:
        raise AnalysisError(f"Unsupported remote safetensors dtype: {encoded}") from error


def download_safetensor_subset(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    suffixes: Sequence[str],
    output: Path,
) -> dict[str, Any]:
    """Download selected tensors from one pinned Hub safetensors file by byte range."""

    torch = _require_torch()
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise AnalysisError("Subset download requires safetensors.") from error
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    encoded_length, first_metadata = _hub_range(url, 0, 7)
    header_length = int.from_bytes(encoded_length, "little")
    if header_length <= 0 or header_length > 128 * 1024 * 1024:
        raise AnalysisError(f"Unsafe remote safetensors header length: {header_length}")
    encoded_header, header_metadata = _hub_range(url, 8, 8 + header_length - 1)
    try:
        header = json.loads(encoded_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisError("Remote safetensors header is invalid") from error
    names: dict[str, str] = {}
    for suffix in suffixes:
        matches = [name for name in header if name != "__metadata__" and name.endswith(suffix)]
        if len(matches) != 1:
            raise AnalysisError(
                f"Expected one remote tensor ending in {suffix!r}, found {len(matches)}"
            )
        names[suffix] = matches[0]
    data_start = 8 + header_length
    tensors: dict[str, Any] = {}
    tensor_metadata: dict[str, Any] = {}
    for suffix, name in names.items():
        descriptor = header[name]
        offsets = descriptor["data_offsets"]
        start = data_start + int(offsets[0])
        stop = data_start + int(offsets[1]) - 1
        payload, response_metadata = _hub_range(url, start, stop)
        buffer = bytearray(payload)
        tensor = torch.frombuffer(buffer, dtype=_torch_dtype(str(descriptor["dtype"])))
        tensor = tensor.reshape(tuple(int(item) for item in descriptor["shape"])).clone()
        tensors[name] = tensor
        tensor_metadata[suffix] = {
            "name": name,
            "dtype": descriptor["dtype"],
            "shape": descriptor["shape"],
            "source_offsets": offsets,
            "response": response_metadata,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "filename": filename,
        "subset_sha256": _sha256_file(output),
        "header_response": {**first_metadata, **header_metadata},
        "tensors": tensor_metadata,
    }


def download_fold_subsets(cache_dir: Path) -> list[tuple[str, Path]]:
    """Download only the ESMFold2 mixing, projection, and LayerNorm tensors."""

    _, sources = _registry_sources()
    root = cache_dir.expanduser().resolve() / "esmfold2-subsets"
    resolved: list[tuple[str, Path]] = []
    for model_id, source in zip(ESMFOLD2_MODEL_IDS, sources, strict=True):
        directory = root / f"{model_id}--{source.revision}"
        output = directory / "model.safetensors"
        metadata_path = directory / "source.json"
        if not output.is_file() or not metadata_path.is_file():
            metadata = download_safetensor_subset(
                repo_id=source.repo_id,
                revision=source.revision,
                filename="model.safetensors",
                suffixes=tuple(_FOLD_SUFFIXES.values()),
                output=output,
            )
            _atomic_json(metadata_path, metadata)
        resolved.append((model_id, directory))
    return resolved


def verify_esmc_files(root: Path) -> dict[str, str]:
    """Verify all pinned official ESMC safetensors SHA-256 identities."""

    esmc, _ = _registry_sources()
    verified: dict[str, str] = {}
    for expected in esmc.files:
        if not expected.path.endswith(".safetensors"):
            continue
        path = root / expected.path
        if not path.is_file():
            raise AnalysisError(f"Missing pinned ESMC shard: {path}")
        actual = _sha256_file(path)
        if expected.algorithm != "sha256" or actual != expected.digest:
            raise AnalysisError(
                f"Checkpoint digest mismatch for {expected.path}: "
                f"expected {expected.encoded}, got sha256:{actual}"
            )
        verified[expected.path] = actual
    return verified


def _runtime_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for distribution in ("torch", "numpy", "safetensors", "matplotlib", "scipy"):
        try:
            from importlib.metadata import version

            result[distribution] = version(distribution)
        except Exception:
            result[distribution] = None
    return result


def run_inventory(
    run: AnalysisRun,
    checkpoint: SafetensorCheckpoint,
    esmc_root: Path,
    folds: Sequence[FoldWeights],
    verified_hashes: Mapping[str, str],
) -> None:
    records = validate_esmc_inventory(checkpoint)
    rows = [asdict(record) for record in records]
    output = run.output_dir / "inventory.csv"
    _write_csv(output, rows)
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        for state, (logit, weight) in enumerate(
            zip(fold.mixing_logits.tolist(), fold.mixing_weights.tolist(), strict=True)
        ):
            fold_rows.append(
                {
                    "checkpoint": fold.label,
                    "state_index": state,
                    "producing_block": block_producing_state(state),
                    "mixing_logit": logit,
                    "mixing_weight": weight,
                    "mixing_weight_pct": weight * 100,
                }
            )
    fold_output = run.output_dir / "esmfold2_mixing_weights.csv"
    _write_csv(fold_output, fold_rows)
    provenance = run.output_dir / "provenance.json"
    esmc_source, fold_sources = _registry_sources()
    _atomic_json(
        provenance,
        {
            "schema_version": SCHEMA_VERSION,
            "analysis_fingerprint": run.fingerprint,
            "weights_only": True,
            "model_execution": False,
            "hidden_states_collected": False,
            "esmc": {
                "path": esmc_root,
                "repo": esmc_source.repo_id,
                "revision": esmc_source.revision,
                "verified_sha256": verified_hashes,
                "tensor_count": len(checkpoint.records),
                "selected_parameter_bytes": sum(record.nbytes for record in records),
            },
            "esmfold2": [
                {
                    "label": fold.label,
                    "path": fold.path,
                }
                for fold in folds
            ],
            "pinned_esmfold2_sources": [
                {
                    "repo": source.repo_id,
                    "revision": source.revision,
                }
                for source in fold_sources
            ],
            "state_block_mapping": {
                "state_51_produced_by_block": block_producing_state(51),
                "state_51_consumed_by_block": 51,
            },
            "runtime": _runtime_versions(),
            "configuration": run.configuration,
        },
    )
    run.write_result(
        "inventory",
        "complete",
        {
            "selected_tensor_count": len(records),
            "fold_checkpoint_count": len(folds),
        },
        (output, fold_output, provenance),
    )


def _save_spectrum(path: Path, spectra: Mapping[str, SpectrumResult]) -> None:
    np = _require_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    values: dict[str, Any] = {}
    for mode, result in spectra.items():
        values[f"{mode}_singular_values"] = result.singular_values.numpy()
        values[f"{mode}_negative_mass"] = np.asarray(result.negative_mass)
        values[f"{mode}_tolerance"] = np.asarray(result.tolerance)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def _spectrum_row(
    block: int,
    family: MatrixFamily,
    matrix: Any,
    spectra: Mapping[str, SpectrumResult],
    validation: Mapping[str, float] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "block": block,
        "produced_state": state_for_block_output(block),
        "family": family,
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "parameter_count": int(matrix.numel()),
        "prespecified_candidate": int(block in PRIMARY_BLOCKS),
        "exploratory_region": int(block in ANOMALY_BLOCKS),
    }
    for mode, result in spectra.items():
        row.update(
            {
                f"{mode}_negative_mass": result.negative_mass,
                f"{mode}_gram_tolerance": result.tolerance,
                f"{mode}_stored_rank": result.stored_rank,
                f"{mode}_fp32_rank": result.fp32_rank,
                f"{mode}_condition_estimate": result.condition_estimate,
            }
        )
        row.update(
            {
                f"{mode}_{key}": value
                for key, value in spectrum_metrics(
                    result.singular_values,
                    matrix.shape,
                    center=cast(CenterMode, mode),
                ).items()
            }
        )
    if validation:
        row.update({f"direct_svd_{key}": value for key, value in validation.items()})
    return row


def run_spectra(
    run: AnalysisRun,
    checkpoint: SafetensorCheckpoint,
    *,
    device: Any,
    accumulation_dtype: str,
) -> None:
    validation_blocks = {0, 50, 51, 79}
    for block in range(N_BLOCKS):
        for family in MATRIX_FAMILIES:
            key = f"block_{block:02d}_{family}"
            if run.is_complete("spectra", key):
                continue
            matrix = load_block_matrix(checkpoint, block, family)
            spectra = {
                mode: gram_spectrum(
                    matrix,
                    center=cast(CenterMode, mode),
                    device=device,
                    accumulation_dtype=accumulation_dtype,
                )
                for mode in ("operator", "rows", "columns")
            }
            validation = (
                direct_svd_validation(matrix, spectra["operator"].singular_values, device)
                if block in validation_blocks
                else None
            )
            spectrum_path = run.output_dir / "spectra" / f"{key}.npz"
            _save_spectrum(spectrum_path, spectra)
            row = _spectrum_row(block, family, matrix, spectra, validation)
            run.write_result("spectra", key, row, (spectrum_path,))
            del matrix, spectra
    rows = sorted(
        run.stage_rows("spectra"),
        key=lambda row: (int(row["block"]), str(row["family"])),
    )
    _write_csv(run.output_dir / "tensor_metrics.csv", rows)


def run_normalization_metrics(
    run: AnalysisRun,
    checkpoint: SafetensorCheckpoint,
) -> None:
    roles = (
        "attn_input_norm_weight",
        "attn_input_norm_bias",
        "q_norm_weight",
        "k_norm_weight",
        "ffn_norm_weight",
        "ffn_norm_bias",
    )
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] = {}
    torch = _require_torch()
    for block in range(N_BLOCKS):
        for role in roles:
            vector = checkpoint.tensor(_block_key(block, role))
            row: dict[str, Any] = {"block": block, "role": role, **vector_metrics(vector)}
            if role in previous:
                left = vector.to(dtype=torch.float64)
                right = previous[role].to(dtype=torch.float64)
                denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
                row["previous_layer_cosine"] = (
                    float((left * right).sum() / denominator)
                    if float(denominator) > 0
                    else None
                )
            else:
                row["previous_layer_cosine"] = None
            previous[role] = vector
            rows.append(row)
    final_name = checkpoint.find_unique_suffix("transformer.norm.weight")
    rows.append(
        {
            "block": 80,
            "role": "final_norm_weight",
            **vector_metrics(checkpoint.tensor(final_name)),
        }
    )
    _write_csv(run.output_dir / "normalization_metrics.csv", rows)


def run_ffn_pair_metrics(run: AnalysisRun, checkpoint: SafetensorCheckpoint) -> None:
    torch = _require_torch()
    rows: list[dict[str, Any]] = []
    for block in range(N_BLOCKS):
        fused = checkpoint.tensor(_block_key(block, "ffn_fc1")).to(dtype=torch.float64)
        down = checkpoint.tensor(_block_key(block, "ffn_down")).to(dtype=torch.float64)
        gate, value = fused.chunk(2, dim=0)
        gate_norm = torch.linalg.vector_norm(gate, dim=1)
        value_norm = torch.linalg.vector_norm(value, dim=1)
        down_norm = torch.linalg.vector_norm(down.mT, dim=1)
        tiny = torch.finfo(torch.float64).tiny
        gate_value_cos = (gate * value).sum(dim=1) / (gate_norm * value_norm).clamp_min(tiny)
        gate_down_cos = (gate * down.mT).sum(dim=1) / (gate_norm * down_norm).clamp_min(tiny)
        value_down_cos = (value * down.mT).sum(dim=1) / (value_norm * down_norm).clamp_min(tiny)
        rows.append(
            {
                "block": block,
                "gate_value_cosine_mean": float(gate_value_cos.mean()),
                "gate_value_cosine_sd": float(gate_value_cos.std(correction=1)),
                "gate_down_cosine_mean": float(gate_down_cos.mean()),
                "gate_down_cosine_sd": float(gate_down_cos.std(correction=1)),
                "value_down_cosine_mean": float(value_down_cos.mean()),
                "value_down_cosine_sd": float(value_down_cos.std(correction=1)),
                "gate_value_norm_ratio_median": float(torch.median(gate_norm / value_norm)),
                "gate_down_norm_ratio_median": float(torch.median(gate_norm / down_norm)),
                "value_down_norm_ratio_median": float(torch.median(value_norm / down_norm)),
            }
        )
    _write_csv(run.output_dir / "ffn_pair_metrics.csv", rows)


def _trajectory_from_stack(stack: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Calculate exact Frobenius geometry and centered kernel PCA for layer matrices."""

    torch = _require_torch()
    value = stack.to(dtype=torch.float64)
    gram = value @ value.mT
    diagonal = torch.diagonal(gram)
    squared_distances = (
        diagonal[:, None] + diagonal[None, :] - 2 * gram
    ).clamp_min(0)
    cosine = gram / torch.sqrt(
        (diagonal[:, None] * diagonal[None, :]).clamp_min(torch.finfo(torch.float64).tiny)
    )
    center = torch.eye(value.shape[0], dtype=torch.float64, device=value.device)
    center -= torch.full_like(center, 1.0 / value.shape[0])
    centered_gram = center @ gram @ center
    eigenvalues = torch.linalg.eigvalsh(centered_gram)
    largest = max(float(eigenvalues[-1]), 1.0)
    tolerance = 10 * torch.finfo(torch.float64).eps * value.shape[0] * largest
    if float(eigenvalues[0]) < -tolerance:
        raise AnalysisError("Centered trajectory Gram matrix is materially non-PSD")
    eigenvalues = eigenvalues.clamp_min(0).flip(0)
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / total if total else eigenvalues
    nonzero = probabilities > 0
    summary = {
        "trajectory_rank": int((eigenvalues > tolerance).sum()),
        "trajectory_participation_ratio": (
            total**2 / float(eigenvalues.square().sum()) if total else 0.0
        ),
        "trajectory_effective_dimension": (
            float(torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum()))
            if total
            else 0.0
        ),
        "trajectory_leading_fraction": float(probabilities[0]) if total else 0.0,
        "trajectory_gram_tolerance": float(tolerance),
    }
    return gram.cpu(), squared_distances.sqrt().cpu(), cosine.cpu(), summary


def run_trajectory(
    run: AnalysisRun,
    checkpoint: SafetensorCheckpoint,
    *,
    device: Any,
    maximum_gib: float,
) -> None:
    torch = _require_torch()
    np = _require_numpy()
    for family in MATRIX_FAMILIES:
        key = f"trajectory_{family}"
        if run.is_complete("dimension", key):
            continue
        first = load_block_matrix(checkpoint, 0, family)
        numel = first.numel()
        required_gib = N_BLOCKS * numel * 4 / 1024**3
        if required_gib > maximum_gib:
            raise AnalysisError(
                f"Exact {family} trajectory requires {required_gib:.2f} GiB for its "
                f"80 x {numel} FP32 stack, above --trajectory-max-gib={maximum_gib:.2f}"
            )
        stack = torch.empty((N_BLOCKS, numel), dtype=torch.float32, device=device)
        stack[0].copy_(first.flatten().to(device))
        del first
        for block in range(1, N_BLOCKS):
            matrix = load_block_matrix(checkpoint, block, family)
            stack[block].copy_(matrix.flatten().to(device))
            del matrix
        gram, distances, cosine, summary = _trajectory_from_stack(stack)
        masked_distances = distances.clone()
        masked_distances.fill_diagonal_(torch.inf)
        neighbors = torch.topk(
            masked_distances,
            k=10,
            largest=False,
            dim=1,
        ).values
        id_summary = intrinsic_dimension_metrics(neighbors, trim_fraction=0)
        summary.update(
            {
                f"diagnostic_{name}": value
                for name, value in id_summary.items()
                if name in {
                    "twonn_dimension",
                    "mle_10_dimension",
                    "nearest_distance_cv",
                    "duplicate_fraction",
                }
            }
        )
        summary["family"] = family
        output = run.output_dir / "trajectory" / f"{family}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            gram=gram.numpy(),
            distances=distances.numpy(),
            cosine=cosine.numpy(),
        )
        run.write_result("dimension", key, summary, (output,))
        del stack
        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    trajectory_rows = [
        row for row in run.stage_rows("dimension") if "trajectory_rank" in row
    ]
    _write_csv(run.output_dir / "trajectory_metrics.csv", trajectory_rows)


def run_intrinsic_dimension(
    run: AnalysisRun,
    checkpoint: SafetensorCheckpoint,
    *,
    device: Any,
    maximum_points: int,
    chunk_size: int,
) -> None:
    for block in range(N_BLOCKS):
        for family in MATRIX_FAMILIES:
            for orientation in ("rows", "columns"):
                for normalized in (False, True):
                    geometry = "unit_normalized" if normalized else "raw_euclidean"
                    key = f"id_{block:02d}_{family}_{orientation}_{geometry}"
                    if run.is_complete("dimension", key):
                        continue
                    matrix = load_block_matrix(checkpoint, block, family)
                    points = matrix if orientation == "rows" else matrix.mT
                    effective_points = (
                        min(points.shape[0], maximum_points)
                        if maximum_points > 0
                        else points.shape[0]
                    )
                    maximum_k = min(max(ID_K_VALUES), effective_points - 1)
                    neighbors = nearest_neighbor_distances(
                        points,
                        k=maximum_k,
                        normalized=normalized,
                        device=device,
                        chunk_size=chunk_size,
                        maximum_points=maximum_points,
                    )
                    row = {
                        "block": block,
                        "produced_state": state_for_block_output(block),
                        "family": family,
                        "orientation": orientation,
                        "geometry": geometry,
                        **intrinsic_dimension_metrics(neighbors),
                    }
                    run.write_result("dimension", key, row)
                    del matrix, points, neighbors
    rows = [
        row
        for row in run.stage_rows("dimension")
        if "orientation" in row
    ]
    rows.sort(
        key=lambda row: (
            int(row["block"]),
            str(row["family"]),
            str(row["orientation"]),
            str(row["geometry"]),
        )
    )
    _write_csv(run.output_dir / "intrinsic_dimension_metrics.csv", rows)


def _head_basis(matrix: Any, family: MatrixFamily, head: int, device: Any) -> Any:
    start = head * D_HEAD
    stop = start + D_HEAD
    if family in {"q", "k", "v"}:
        return orthonormal_basis(
            matrix[start:stop],
            side="row",
            rank=D_HEAD,
            device=device,
        )
    if family == "o":
        return orthonormal_basis(
            matrix[:, start:stop],
            side="column",
            rank=D_HEAD,
            device=device,
        )
    raise ValueError(f"Head basis is undefined for {family!r}")


def _head_spectrum_metrics(matrix: Any) -> dict[str, float | int | None]:
    torch = _require_torch()
    singular = torch.linalg.svdvals(matrix.to(dtype=torch.float64))
    return spectrum_metrics(singular, matrix.shape)


def _hungarian_assignment(cost: Any) -> tuple[Any, Any]:
    np = _require_numpy()
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise AnalysisError(
            "Attention-head matching requires scipy from the FastPLMs reporting extra."
        ) from error
    return linear_sum_assignment(np.asarray(cost, dtype=float))


def run_heads(
    run: AnalysisRun,
    checkpoint: SafetensorCheckpoint,
    folds: Sequence[FoldWeights],
    *,
    device: Any,
) -> None:
    torch = _require_torch()
    fold_bases = {
        fold.label: projection_bases(fold, device)
        for fold in folds
    }
    head_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    previous_bases: list[dict[str, Any]] | None = None
    for block in range(N_BLOCKS):
        matrices = {
            family: load_block_matrix(
                checkpoint,
                block,
                cast(MatrixFamily, family),
            ).to(device)
            for family in ("q", "k", "v", "o")
        }
        current_bases: list[dict[str, Any]] = []
        output_bases: list[Any] = []
        for head in range(N_HEADS):
            bases = {
                family: _head_basis(
                    matrix,
                    cast(MatrixFamily, family),
                    head,
                    device,
                )
                for family, matrix in matrices.items()
            }
            current_bases.append(bases)
            output_bases.append(bases["o"])
            qk = subspace_metrics(bases["q"], bases["k"])
            vo = subspace_metrics(bases["v"], bases["o"])
            q_slice = matrices["q"][head * D_HEAD : (head + 1) * D_HEAD]
            k_slice = matrices["k"][head * D_HEAD : (head + 1) * D_HEAD]
            v_slice = matrices["v"][head * D_HEAD : (head + 1) * D_HEAD]
            o_slice = matrices["o"][:, head * D_HEAD : (head + 1) * D_HEAD]
            spectral = {
                family: _head_spectrum_metrics(value)
                for family, value in {
                    "q": q_slice,
                    "k": k_slice,
                    "v": v_slice,
                    "o": o_slice,
                }.items()
            }
            row: dict[str, Any] = {
                "block": block,
                "produced_state": state_for_block_output(block),
                "head": head,
                "qk_overlap": qk["normalized_overlap"],
                "qk_median_angle_degrees": qk["median_angle_degrees"],
                "vo_overlap": vo["normalized_overlap"],
                "vo_median_angle_degrees": vo["median_angle_degrees"],
            }
            for family, metrics in spectral.items():
                for name in (
                    "frobenius_norm",
                    "spectral_norm",
                    "stable_rank",
                    "effective_rank",
                ):
                    row[f"{family}_{name}"] = metrics[name]
            for label, variants in fold_bases.items():
                for variant, basis in variants.items():
                    row[f"{label}_{variant}_output_overlap"] = subspace_metrics(
                        bases["o"], basis
                    )["normalized_overlap"]
            head_rows.append(row)
        concatenated_output = torch.cat(output_bases, dim=1)
        output_cross = (concatenated_output.mT @ concatenated_output).reshape(
            N_HEADS,
            D_HEAD,
            N_HEADS,
            D_HEAD,
        )
        affinity = output_cross.square().sum(dim=(1, 3)) / D_HEAD
        affinity.fill_diagonal_(0)
        redundancy = float(affinity.sum() / (N_HEADS * (N_HEADS - 1)))
        for row in head_rows[-N_HEADS:]:
            row["within_layer_output_redundancy_mean"] = redundancy
        if previous_bases is not None:
            similarity = torch.zeros(
                (N_HEADS, N_HEADS),
                dtype=torch.float64,
                device=device,
            )
            for family in ("q", "k", "v", "o"):
                previous_cat = torch.cat(
                    [basis[family] for basis in previous_bases],
                    dim=1,
                )
                current_cat = torch.cat(
                    [basis[family] for basis in current_bases],
                    dim=1,
                )
                cross = (previous_cat.mT @ current_cat).reshape(
                    N_HEADS,
                    D_HEAD,
                    N_HEADS,
                    D_HEAD,
                )
                similarity += cross.square().sum(dim=(1, 3)) / D_HEAD
            similarity /= 4
            left, right = _hungarian_assignment(1 - similarity.cpu().numpy())
            transition_rows.append(
                {
                    "from_block": block - 1,
                    "to_block": block,
                    "fixed_index_similarity_mean": float(torch.diagonal(similarity).mean()),
                    "hungarian_similarity_mean": float(
                        similarity[left.tolist(), right.tolist()].mean()
                    ),
                    "head_permutation": json.dumps(
                        {int(i): int(j) for i, j in zip(left, right, strict=True)},
                        sort_keys=True,
                    ),
                }
            )
        previous_bases = current_bases
        del matrices, affinity
    _write_csv(run.output_dir / "attention_head_metrics.csv", head_rows)
    _write_csv(run.output_dir / "attention_head_transitions.csv", transition_rows)


def run_alignment(
    run: AnalysisRun,
    checkpoint: SafetensorCheckpoint,
    folds: Sequence[FoldWeights],
    *,
    device: Any,
) -> None:
    bases = {
        fold.label: projection_bases(fold, device)
        for fold in folds
    }
    rows: list[dict[str, Any]] = []
    side_by_family: dict[MatrixFamily, Literal["row", "column"]] = {
        "q": "row",
        "k": "row",
        "v": "row",
        "gate": "row",
        "value": "row",
        "o": "column",
        "down": "column",
    }
    for block in range(N_BLOCKS):
        for family in MATRIX_FAMILIES:
            matrix = load_block_matrix(checkpoint, block, family).to(device)
            full_layer_basis = orthonormal_basis(
                matrix,
                side=side_by_family[family],
                rank=max(ALIGNMENT_RANKS),
                device=device,
            )
            for rank in ALIGNMENT_RANKS:
                layer_basis = full_layer_basis[:, :rank]
                for fold_label, variants in bases.items():
                    for variant, fold_basis in variants.items():
                        metrics = subspace_metrics(layer_basis, fold_basis)
                        rows.append(
                            {
                                "block": block,
                                "produced_state": state_for_block_output(block),
                                "family": family,
                                "side": side_by_family[family],
                                "rank": rank,
                                "checkpoint": fold_label,
                                "projection_variant": variant,
                                **metrics,
                            }
                        )
            del matrix, full_layer_basis
    _write_csv(run.output_dir / "projection_alignment.csv", rows)


def run_compression(
    run: AnalysisRun,
    checkpoint: SafetensorCheckpoint,
    *,
    device: Any,
) -> None:
    rows: list[dict[str, Any]] = []
    for block in range(N_BLOCKS):
        for family in MATRIX_FAMILIES:
            matrix = load_block_matrix(checkpoint, block, family).to(device)
            original_spectral_norm = power_spectral_norm(matrix)
            reconstructions: list[tuple[str, Any, dict[str, float]]] = []
            for bits in (8, 4):
                reconstructed, metrics = symmetric_row_quantize(matrix, cast(Any, bits))
                reconstructions.append((f"int{bits}_per_row", reconstructed, metrics))
            for sparsity in (0.10, 0.25, 0.50, 0.75):
                reconstructed, metrics = magnitude_prune(matrix, sparsity)
                reconstructions.append(
                    (f"unstructured_{int(sparsity * 100)}pct", reconstructed, metrics)
                )
            reconstructed, metrics = structured_two_of_four(matrix)
            reconstructions.append(("structured_2_of_4", reconstructed, metrics))
            for method, reconstructed, metrics in reconstructions:
                rows.append(
                    {
                        "block": block,
                        "produced_state": state_for_block_output(block),
                        "family": family,
                        "method": method,
                        **metrics,
                        "spectral_distortion": spectral_distortion(
                            matrix,
                            reconstructed,
                            original_spectral_norm=original_spectral_norm,
                        ),
                    }
                )
                del reconstructed
            del matrix
    _write_csv(run.output_dir / "compression_metrics.csv", rows)


def _rankdata(values: Any) -> Any:
    np = _require_numpy()
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def _pearson(left: Any, right: Any) -> float | None:
    np = _require_numpy()
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _detrend(values: Any) -> tuple[Any, Any]:
    np = _require_numpy()
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    degree = min(3, len(y) - 1)
    coefficients = np.polyfit(x, y, degree)
    trend = np.polyval(coefficients, x)
    return y - trend, trend


def _bh_adjust(p_values: Sequence[float]) -> list[float]:
    np = _require_numpy()
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 1.0
    total = len(values)
    for reversed_index in range(total - 1, -1, -1):
        original_index = order[reversed_index]
        candidate = values[original_index] * total / (reversed_index + 1)
        running = min(running, candidate)
        adjusted[original_index] = min(1.0, running)
    return adjusted.tolist()


def _phase_randomized_pvalues(
    residuals: Any,
    *,
    draws: int,
    seed: int,
) -> list[float]:
    """Return two-sided p-values from phase-randomized autocorrelation surrogates."""

    np = _require_numpy()
    values = np.asarray(residuals, dtype=float)
    spectrum = np.fft.rfft(values)
    amplitudes = np.abs(spectrum)
    original_phase = np.angle(spectrum)
    rng = np.random.default_rng(seed)
    exceedances = np.zeros(len(values), dtype=np.int64)
    observed = np.abs(values)
    remaining = draws
    while remaining:
        batch = min(512, remaining)
        phases = rng.uniform(-np.pi, np.pi, size=(batch, len(spectrum)))
        phases[:, 0] = original_phase[0]
        if len(values) % 2 == 0:
            phases[:, -1] = original_phase[-1]
        randomized = amplitudes[None, :] * np.exp(1j * phases)
        surrogates = np.fft.irfft(randomized, n=len(values), axis=1)
        exceedances += np.sum(np.abs(surrogates) >= observed[None, :], axis=0)
        remaining -= batch
    return ((exceedances + 1) / (draws + 1)).tolist()


def _numeric_metric_names(rows: Sequence[Mapping[str, Any]], excluded: set[str]) -> list[str]:
    names: list[str] = []
    for name in sorted({key for row in rows for key in row} - excluded):
        valid = True
        found = False
        for row in rows:
            value = row.get(name)
            if value in (None, ""):
                valid = False
                break
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                valid = False
                break
            found = found or math.isfinite(parsed)
        if valid and found:
            names.append(name)
    return names


def anomaly_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Calculate depth residuals, local robust scores, and exact-shift q-values."""

    np = _require_numpy()
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
    output: list[dict[str, Any]] = []
    for family, family_rows in sorted(by_family.items()):
        ordered = sorted(family_rows, key=lambda row: int(row["block"]))
        if len(ordered) != N_BLOCKS:
            raise AnalysisError(f"Expected 80 {family} rows, found {len(ordered)}")
        metric_names = _numeric_metric_names(
            ordered,
            {
                "block",
                "produced_state",
                "rows",
                "columns",
                "parameter_count",
                "prespecified_candidate",
                "exploratory_region",
            },
        )
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in ordered])
            if not np.all(np.isfinite(values)) or np.std(values) == 0:
                continue
            residuals, trend = _detrend(values)
            absolute = np.abs(residuals)
            shift_p_values = []
            for index in range(N_BLOCKS):
                shifted = np.asarray(
                    [np.roll(residuals, shift)[index] for shift in range(1, N_BLOCKS)]
                )
                shift_p_values.append(
                    (1 + int(np.sum(np.abs(shifted) >= absolute[index]))) / N_BLOCKS
                )
            seed = int.from_bytes(
                hashlib.sha256(f"{family}\0{metric}".encode()).digest()[:8],
                "little",
            )
            phase_p_values = _phase_randomized_pvalues(
                residuals,
                draws=10_000,
                seed=seed,
            )
            q_values = _bh_adjust(phase_p_values)
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            for index, value in enumerate(values):
                neighbors = [
                    values[candidate]
                    for candidate in range(max(0, index - 5), min(N_BLOCKS, index + 6))
                    if candidate != index
                ]
                local_median = float(np.median(neighbors))
                local_mad = float(np.median(np.abs(np.asarray(neighbors) - local_median)))
                local_z = (
                    0.67449 * (float(value) - local_median) / local_mad
                    if local_mad > 0
                    else None
                )
                global_z = (
                    0.67449 * (float(value) - median) / mad
                    if mad > 0
                    else None
                )
                percentile = (
                    100.0
                    * (
                        np.sum(values < value)
                        + 0.5 * np.sum(values == value)
                    )
                    / N_BLOCKS
                )
                output.append(
                    {
                        "family": family,
                        "metric": metric,
                        "block": index,
                        "produced_state": index + 1,
                        "value": float(value),
                        "depth_trend": float(trend[index]),
                        "depth_residual": float(residuals[index]),
                        "local_robust_z": local_z,
                        "global_robust_z": global_z,
                        "global_percentile": float(percentile),
                        "exact_circular_shift_p": shift_p_values[index],
                        "phase_randomized_p": phase_p_values[index],
                        "phase_randomized_draws": 10_000,
                        "bh_q": q_values[index],
                        "prespecified_candidate": int(index in PRIMARY_BLOCKS),
                    }
                )
    return output


def correlation_table(
    tensor_rows: Sequence[Mapping[str, Any]],
    mixing_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Relate block metrics to fold mixing weights under exact state indexing."""

    np = _require_numpy()
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in tensor_rows:
        by_family[str(row["family"])].append(row)
    profiles: dict[str, dict[int, float]] = defaultdict(dict)
    for row in mixing_rows:
        profiles[str(row["checkpoint"])][int(row["state_index"])] = float(
            row["mixing_weight"]
        )
    output: list[dict[str, Any]] = []
    for family, family_rows in sorted(by_family.items()):
        ordered = sorted(family_rows, key=lambda row: int(row["block"]))
        metric_names = _numeric_metric_names(
            ordered,
            {
                "block",
                "produced_state",
                "rows",
                "columns",
                "parameter_count",
                "prespecified_candidate",
                "exploratory_region",
            },
        )
        for metric in metric_names:
            metric_values = np.asarray([float(row[metric]) for row in ordered])
            if not np.all(np.isfinite(metric_values)) or np.std(metric_values) == 0:
                continue
            for checkpoint, profile in sorted(profiles.items()):
                for lag in range(-2, 3):
                    pairs = [
                        (metric_values[block], profile[block + 1 + lag])
                        for block in range(N_BLOCKS)
                        if block + 1 + lag in profile
                    ]
                    if len(pairs) < 10:
                        continue
                    left = np.asarray([pair[0] for pair in pairs])
                    right = np.asarray([pair[1] for pair in pairs])
                    left_residual, _ = _detrend(left)
                    right_residual, _ = _detrend(right)
                    residual_correlation = _pearson(left_residual, right_residual)
                    shifted = [
                        _pearson(left_residual, np.roll(right_residual, shift))
                        for shift in range(1, len(right_residual))
                    ]
                    valid_shifted = [
                        value for value in shifted if value is not None and math.isfinite(value)
                    ]
                    exact_p = (
                        (
                            1
                            + sum(
                                abs(value) >= abs(cast(float, residual_correlation))
                                for value in valid_shifted
                            )
                        )
                        / (1 + len(valid_shifted))
                        if residual_correlation is not None
                        else None
                    )
                    output.append(
                        {
                            "family": family,
                            "metric": metric,
                            "checkpoint": checkpoint,
                            "lag": lag,
                            "n": len(pairs),
                            "pearson": _pearson(left, right),
                            "spearman": _pearson(_rankdata(left), _rankdata(right)),
                            "depth_detrended_pearson": residual_correlation,
                            "exact_circular_shift_p": exact_p,
                            "unique_nonzero_shifts": len(right_residual) - 1,
                            "primary_indexing": int(lag == 0),
                        }
                    )
    return output


def _aggregate_layer_metrics(tensor_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    np = _require_numpy()
    metrics = (
        "operator_stable_rank",
        "operator_effective_rank",
        "operator_participation_ratio",
        "operator_rank_95",
        "operator_leading_energy_fraction",
    )
    output: list[dict[str, Any]] = []
    for block in range(N_BLOCKS):
        selected = [row for row in tensor_rows if int(row["block"]) == block]
        row: dict[str, Any] = {
            "block": block,
            "produced_state": block + 1,
            "prespecified_candidate": int(block in PRIMARY_BLOCKS),
        }
        for metric in metrics:
            values = [
                float(item[metric])
                for item in selected
                if item.get(metric) not in (None, "")
            ]
            row[f"mean_{metric}"] = float(np.mean(values)) if values else None
            row[f"median_{metric}"] = float(np.median(values)) if values else None
        output.append(row)
    return output


def run_statistics(run: AnalysisRun) -> None:
    tensor_path = run.output_dir / "tensor_metrics.csv"
    mixing_path = run.output_dir / "esmfold2_mixing_weights.csv"
    if not tensor_path.is_file():
        raise AnalysisError("Report requires tensor_metrics.csv from the spectra stage")
    tensor_rows = _read_csv(tensor_path)
    layer_rows = _aggregate_layer_metrics(tensor_rows)
    _write_csv(run.output_dir / "layer_metrics.csv", layer_rows)
    anomalies = anomaly_table(tensor_rows)
    _write_csv(run.output_dir / "layer_anomalies.csv", anomalies)
    if mixing_path.is_file():
        correlations = correlation_table(tensor_rows, _read_csv(mixing_path))
        _write_csv(run.output_dir / "mixing_correlations.csv", correlations)


def _require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise AnalysisError(
            "Report plots require matplotlib from the FastPLMs reporting extra."
        ) from error
    return plt


def _save_figure(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor=_PALETTE["paper"])


def _plot_spectral_profiles(tensor_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _require_matplotlib()
    metrics = (
        ("operator_stable_rank", "Stable rank"),
        ("operator_effective_rank", "Effective rank"),
        ("operator_rank_95", "Rank retaining 95% energy"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        for family in MATRIX_FAMILIES:
            selected = sorted(
                (row for row in tensor_rows if row["family"] == family),
                key=lambda row: int(row["block"]),
            )
            color, linestyle = _FAMILY_STYLES[family]
            axis.plot(
                [int(row["block"]) for row in selected],
                [float(row[metric]) for row in selected],
                label=family,
                color=color,
                linestyle=linestyle,
                linewidth=1.5,
            )
        for block, color in ((50, _PALETTE["gold"]), (51, _PALETTE["orange"])):
            axis.axvline(block, color=color, linewidth=1.2, linestyle="--")
        axis.set_title(title, loc="left", color=_PALETTE["ink"])
        axis.set_ylabel("Dimension")
        axis.grid(axis="y", color=_PALETTE["grid"], linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    axes[-1].set_xlabel("ESMC transformer block (state = block + 1)")
    figure.suptitle(
        "ESMC-6B weight-matrix spectral geometry",
        x=0.08,
        y=1.01,
        ha="left",
        color=_PALETTE["ink"],
        fontsize=14,
    )
    figure.text(
        0.08,
        0.98,
        "All 80 blocks; dashed references mark blocks 50 and 51",
        color=_PALETTE["muted"],
    )
    figure.tight_layout()
    _save_figure(figure, path)
    plt.close(figure)


def _plot_mixing_profiles(mixing_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plt = _require_matplotlib()
    figure, axis = plt.subplots(figsize=(11, 4.8))
    checkpoints = sorted({str(row["checkpoint"]) for row in mixing_rows})
    styles = ("-", "--", "-.", ":")
    for checkpoint, style in zip(checkpoints, styles, strict=False):
        selected = sorted(
            (row for row in mixing_rows if row["checkpoint"] == checkpoint),
            key=lambda row: int(row["state_index"]),
        )
        axis.plot(
            [int(row["state_index"]) for row in selected],
            [100 * float(row["mixing_weight"]) for row in selected],
            label=checkpoint,
            linestyle=style,
            color=_PALETTE["blue"],
            linewidth=1.7,
        )
    axis.axvline(51, color=_PALETTE["orange"], linewidth=1.4, linestyle="--")
    axis.set_title("ESMFold2 ESMC state-mixing weights", loc="left")
    axis.set_xlabel("ESMC hidden-state index")
    axis.set_ylabel("Softmax weight (%)")
    axis.grid(axis="y", color=_PALETTE["grid"], linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2)
    figure.text(
        0.125,
        0.91,
        "State 51 is produced by block 50 and consumed by block 51",
        color=_PALETTE["muted"],
    )
    figure.tight_layout()
    _save_figure(figure, path)
    plt.close(figure)


def _plot_alignment(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    np = _require_numpy()
    plt = _require_matplotlib()
    selected = [
        row
        for row in rows
        if int(row["rank"]) == 256 and row["projection_variant"] == "raw"
    ]
    figure, axis = plt.subplots(figsize=(11, 5.4))
    for family in MATRIX_FAMILIES:
        family_rows = [row for row in selected if row["family"] == family]
        values = []
        for block in range(N_BLOCKS):
            block_values = [
                float(row["normalized_overlap"])
                for row in family_rows
                if int(row["block"]) == block
            ]
            values.append(float(np.mean(block_values)) if block_values else np.nan)
        color, linestyle = _FAMILY_STYLES[family]
        axis.plot(
            range(N_BLOCKS),
            values,
            label=family,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
        )
    axis.axvline(50, color=_PALETTE["gold"], linewidth=1.2, linestyle="--")
    axis.axvline(51, color=_PALETTE["orange"], linewidth=1.2, linestyle="--")
    axis.set_title("ESMC weight subspace overlap with ESMFold2 projections", loc="left")
    axis.set_xlabel("ESMC transformer block")
    axis.set_ylabel("Normalized overlap")
    axis.grid(axis="y", color=_PALETTE["grid"], linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=7)
    figure.text(
        0.125,
        0.91,
        "Rank 256; mean across checkpoints; raw projection row spaces",
        color=_PALETTE["muted"],
    )
    figure.tight_layout()
    _save_figure(figure, path)
    plt.close(figure)


def _candidate_summary(anomalies: Sequence[Mapping[str, Any]]) -> list[str]:
    candidates = [
        row
        for row in anomalies
        if int(row["block"]) in PRIMARY_BLOCKS and float(row["bh_q"]) <= 0.05
    ]
    candidates.sort(key=lambda row: float(row["bh_q"]))
    if not candidates:
        return [
            "Neither block 50 nor block 51 passes the prespecified 5% FDR threshold "
            "for the analyzed spectral metrics."
        ]
    return [
        (
            f"Block {row['block']} is an outlier for {row['family']} "
            f"{row['metric']} (q={float(row['bh_q']):.3g}, "
            f"local robust z={float(row['local_robust_z']):.2f})."
        )
        for row in candidates[:10]
        if row.get("local_robust_z") not in (None, "")
    ]


def write_report(run: AnalysisRun) -> None:
    run_statistics(run)
    tensor_rows = _read_csv(run.output_dir / "tensor_metrics.csv")
    mixing_rows = _read_csv(run.output_dir / "esmfold2_mixing_weights.csv")
    anomaly_rows = _read_csv(run.output_dir / "layer_anomalies.csv")
    figures = run.output_dir / "figures"
    spectral_figure = figures / "layer_spectral_profiles.png"
    mixing_figure = figures / "esmfold2_mixing_weights.png"
    _plot_spectral_profiles(tensor_rows, spectral_figure)
    _plot_mixing_profiles(mixing_rows, mixing_figure)
    alignment_figure: Path | None = None
    alignment_path = run.output_dir / "projection_alignment.csv"
    if alignment_path.is_file():
        alignment_figure = figures / "projection_alignment.png"
        _plot_alignment(_read_csv(alignment_path), alignment_figure)
    findings = _candidate_summary(anomaly_rows)
    findings_markdown = "\n".join(f"- {finding}" for finding in findings)
    alignment_section = (
        "\n## Projection alignment\n\n"
        "The overlap curves compare ESMC read/write singular subspaces with each "
        "ESMFold2 projection row space. The LayerNorm-scaled variant is a weights-only "
        "channel-scaling approximation, not the input-dependent LayerNorm Jacobian.\n\n"
        f"![Projection alignment](figures/{alignment_figure.name})\n"
        if alignment_figure is not None
        else ""
    )
    report = f"""# ESMC-6B weights-only layer geometry

## Technical summary

This report analyzes checkpoint parameters only. No sequences were supplied, no model
was instantiated, and no hidden states or outputs were collected. ESMFold2 state 51 maps
to the output of ESMC block 50; block 51 consumes that state.

{findings_markdown}

These are weight-space findings. They do not establish representational content,
folding utility, perplexity effects, or biological information retention.

## Spectral geometry across all 80 blocks

The profiles show the uncentered operator spectra for Q, K, V, attention output,
SwiGLU gate/value, and FFN down matrices. Dimension metrics are computed from the full
singular spectrum rather than a top-rank approximation.

![Layer-wise spectral profiles](figures/{spectral_figure.name})

## ESMFold2 state mixing

All four checkpoint profiles are shown on the 81-state axis. The coefficient at state 51
must be compared primarily with block 50 weight geometry, with block 51 retained as the
prespecified consumer-side candidate.

![ESMFold2 mixing profiles](figures/{mixing_figure.name})
{alignment_section}
## Scope and metric definitions

- Stable rank is Frobenius energy divided by squared spectral norm.
- Effective rank is the exponential entropy of normalized squared singular values.
- Rank-95 is the smallest rank retaining 95% of squared Frobenius energy.
- Row and column PCA are centered separately from the uncentered operator spectrum.
- Intrinsic-dimension estimates, when present, describe clouds of weight vectors, not
  hidden representations.

## Methods and robustness

The smaller Gram matrix is accumulated in the configured precision. Tolerance-scale
negative eigenvalues are recorded and clamped; materially negative eigenvalues fail the
analysis. Blocks 50 and 51 are prespecified. Depth-adjusted diagnostics include all 79
unique nonzero circular shifts. Inferential p-values use 10,000 deterministic
phase-randomized surrogates followed by Benjamini-Hochberg correction within
tensor/metric families.

## Limitations

Weight matrices have parameterization symmetries, and intrinsic dimension of their rows
or columns is not intrinsic dimension of the model's learned representation. Compression
results quantify parameter reconstruction only. The four ESMFold2 checkpoints are highly
correlated robustness checks, not independent biological replicates.

## Recommended next step

Use any reproducible block-50 or block-51 anomaly to define a narrow later activation or
causal study. Do not select activation experiments from metrics that fail the depth,
normalization, numerical, or checkpoint-consistency checks reported here.

## Further questions

- Does any candidate anomaly localize to a small number of attention heads?
- Is the anomaly driven by scale, spectral shape, subspace orientation, or compression?
- Does it align consistently with the raw and LayerNorm-scaled ESMFold2 projections?
"""
    report_path = run.output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    artifacts = [report_path, spectral_figure, mixing_figure]
    if alignment_figure is not None:
        artifacts.append(alignment_figure)
    run.write_result(
        "report",
        "complete",
        {"candidate_findings": findings},
        artifacts,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--esmc-checkpoint",
        type=Path,
        help="Local pinned biohub/ESMC-6B safetensors directory.",
    )
    parser.add_argument(
        "--esmfold2-checkpoint",
        action="append",
        type=parse_fold_argument,
        default=[],
        metavar="LABEL=PATH",
        help="Local ESMFold2 safetensors checkpoint. Repeat for all four variants.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache") / "esmc-weight-geometry",
        help="Explicit cache root used only with --download.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the six pinned ESMC-6B shards when no local checkpoint is supplied.",
    )
    parser.add_argument(
        "--download-fold-subsets",
        action="store_true",
        help="Range-download only the four required tensors from each pinned ESMFold2 file.",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip full shard SHA-256 verification for an already verified local snapshot.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda[:index]. Large exact trajectories are intended for an 80 GB GPU.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=(*STAGES, "all"),
        help="Analysis stage. Repeat to run several; defaults to all.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--accumulation-dtype",
        choices=("float64", "float32"),
        default="float64",
    )
    parser.add_argument(
        "--id-max-points",
        type=int,
        default=0,
        help="Deterministic point cap for ID estimation; zero uses all rows and columns.",
    )
    parser.add_argument("--knn-chunk-size", type=int, default=128)
    parser.add_argument(
        "--trajectory-max-gib",
        type=float,
        default=8.0,
        help="Maximum memory allowed for one exact 80-layer matrix-family stack.",
    )
    return parser


def _selected_stages(arguments: argparse.Namespace) -> tuple[str, ...]:
    requested = arguments.stage or ["all"]
    if "all" in requested:
        return STAGES
    return tuple(stage for stage in STAGES if stage in set(requested))


def _resolve_esmc_root(arguments: argparse.Namespace) -> Path:
    if arguments.esmc_checkpoint is not None:
        return arguments.esmc_checkpoint.expanduser().resolve()
    if not arguments.download:
        raise AnalysisError("Pass --esmc-checkpoint PATH or explicitly enable --download")
    return download_esmc_checkpoint(arguments.cache_dir)


def _load_folds(arguments: argparse.Namespace) -> list[FoldWeights]:
    requested = list(arguments.esmfold2_checkpoint)
    if arguments.download_fold_subsets:
        if requested:
            raise AnalysisError(
                "--download-fold-subsets cannot be combined with "
                "--esmfold2-checkpoint"
            )
        requested = download_fold_subsets(arguments.cache_dir)
    seen: set[str] = set()
    folds: list[FoldWeights] = []
    for label, path in requested:
        if label in seen:
            raise AnalysisError(f"Duplicate ESMFold2 checkpoint label: {label!r}")
        seen.add(label)
        folds.append(load_fold_weights(label, path))
    return folds


def _validate_fold_count(stages: Sequence[str], folds: Sequence[FoldWeights]) -> None:
    requiring_folds = {"inventory", "heads", "alignment", "report"}
    if requiring_folds.intersection(stages) and len(folds) != 4:
        raise AnalysisError(
            "Inventory, heads, alignment, and report stages require exactly four "
            "--esmfold2-checkpoint LABEL=PATH arguments"
        )


def _run_manifest(
    run: AnalysisRun,
    *,
    stages: Sequence[str],
    esmc_root: Path,
    folds: Sequence[FoldWeights],
) -> None:
    outputs = sorted(
        str(path.relative_to(run.output_dir))
        for path in run.output_dir.rglob("*")
        if path.is_file() and ".progress" not in path.parts and path.name != "run_manifest.json"
    )
    _atomic_json(
        run.output_dir / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "analysis_fingerprint": run.fingerprint,
            "weights_only": True,
            "selected_stages": list(stages),
            "esmc_checkpoint": esmc_root,
            "esmfold2_checkpoints": {
                fold.label: fold.path
                for fold in folds
            },
            "outputs": outputs,
            "runtime": _runtime_versions(),
        },
    )


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        stages = _selected_stages(arguments)
        output_dir = arguments.output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()) and not arguments.resume:
            raise AnalysisError(
                f"Output directory is not empty: {output_dir}. Pass --resume to reuse it."
            )
        if arguments.id_max_points < 0:
            raise AnalysisError("--id-max-points must be nonnegative")
        if arguments.knn_chunk_size < 1:
            raise AnalysisError("--knn-chunk-size must be positive")
        if arguments.trajectory_max_gib <= 0:
            raise AnalysisError("--trajectory-max-gib must be positive")
        esmc_root = _resolve_esmc_root(arguments)
        verified_hashes = (
            {} if arguments.skip_checksum else verify_esmc_files(esmc_root)
        )
        checkpoint = SafetensorCheckpoint(esmc_root)
        validate_esmc_inventory(checkpoint)
        folds = _load_folds(arguments)
        _validate_fold_count(stages, folds)
        device = _device(arguments.device)
        configuration = {
            "schema_version": SCHEMA_VERSION,
            "esmc_checkpoint": esmc_root,
            "verified_sha256": verified_hashes,
            "esmfold2_checkpoints": {
                fold.label: fold.path
                for fold in folds
            },
            "device": str(device),
            "accumulation_dtype": arguments.accumulation_dtype,
            "id_max_points": arguments.id_max_points,
            "knn_chunk_size": arguments.knn_chunk_size,
            "trajectory_max_gib": arguments.trajectory_max_gib,
        }
        run = AnalysisRun(output_dir, configuration, arguments.resume)
        if "inventory" in stages:
            run_inventory(run, checkpoint, esmc_root, folds, verified_hashes)
        if "spectra" in stages:
            run_spectra(
                run,
                checkpoint,
                device=device,
                accumulation_dtype=arguments.accumulation_dtype,
            )
            run_normalization_metrics(run, checkpoint)
            run_ffn_pair_metrics(run, checkpoint)
        if "dimension" in stages:
            run_intrinsic_dimension(
                run,
                checkpoint,
                device=device,
                maximum_points=arguments.id_max_points,
                chunk_size=arguments.knn_chunk_size,
            )
            run_trajectory(
                run,
                checkpoint,
                device=device,
                maximum_gib=arguments.trajectory_max_gib,
            )
        if "heads" in stages:
            run_heads(run, checkpoint, folds, device=device)
        if "alignment" in stages:
            run_alignment(run, checkpoint, folds, device=device)
        if "compression" in stages:
            run_compression(run, checkpoint, device=device)
        if "report" in stages:
            write_report(run)
        _run_manifest(
            run,
            stages=stages,
            esmc_root=esmc_root,
            folds=folds,
        )
    except (AnalysisError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
