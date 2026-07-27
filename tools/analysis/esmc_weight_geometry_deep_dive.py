"""Exhaustive post hoc and raw-weight analysis for ESMC-6B layer geometry.

This module extends :mod:`tools.analysis.esmc_weight_geometry` without running
model inputs. It treats the checkpoint as the primary source, streams one
block at a time, and preserves all scalar results in CSV plus large arrays in
compressed NPZ files. The baseline analysis directory is immutable input.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from tools.analysis import esmc_weight_geometry as base

SCHEMA_VERSION = 1
TOP_RANK_DEFAULT = 64
SPECTRAL_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
RAW_QUANTILE_SAMPLE = 1_000_000
COHERENCE_SAMPLE = 256
FAMILY_INDEX = {family: index for index, family in enumerate(base.MATRIX_FAMILIES)}
VECTOR_ROLES = (
    "attn_input_norm_weight",
    "attn_input_norm_bias",
    "q_norm_weight",
    "k_norm_weight",
    "ffn_norm_weight",
    "ffn_norm_bias",
)


class DeepDiveError(RuntimeError):
    """Raised when an extended-analysis invariant is violated."""


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    base._atomic_json(path, value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    base._write_csv(path, rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_csv(path).to_dict(orient="records")


def _gini(values: Any) -> float:
    np = base._require_numpy()
    x = np.asarray(values, dtype=np.float64).ravel()
    x = np.abs(x[np.isfinite(x)])
    if x.size == 0 or float(x.sum()) == 0:
        return 0.0
    x.sort()
    indices = np.arange(1, x.size + 1, dtype=np.float64)
    return float((2 * np.sum(indices * x) / (x.size * x.sum())) - (x.size + 1) / x.size)


def _entropy_effective(values: Any) -> tuple[float, float, float]:
    np = base._require_numpy()
    x = np.abs(np.asarray(values, dtype=np.float64).ravel())
    total = float(x.sum())
    if not math.isfinite(total) or total <= 0:
        return 0.0, 0.0, 0.0
    p = x / total
    positive = p[p > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    hhi = float(np.square(p).sum())
    return entropy, float(np.exp(entropy)), hhi


def _safe_corr(left: Any, right: Any, *, method: str = "pearson") -> float | None:
    np = base._require_numpy()
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    scale_x = float(np.std(x))
    scale_y = float(np.std(y))
    floor = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(x))), float(np.max(np.abs(y))))
    if x.size < 3 or scale_x <= floor or scale_y <= floor:
        return None
    if method == "spearman":
        x = base._rankdata(x)
        y = base._rankdata(y)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= np.finfo(np.float64).tiny:
        return None
    return float(np.dot(x, y) / denominator)


def _bh(rows: list[dict[str, Any]], p_field: str, q_field: str) -> None:
    valid = [
        (index, float(row[p_field]))
        for index, row in enumerate(rows)
        if row.get(p_field) is not None
    ]
    adjusted = base._bh_adjust([value for _, value in valid])
    for (index, _), q in zip(valid, adjusted, strict=True):
        rows[index][q_field] = q


def _distribution(values: Any, prefix: str = "") -> dict[str, float | int | None]:
    np = base._require_numpy()
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]

    def key(name: str) -> str:
        return f"{prefix}{name}"

    if x.size == 0:
        return {key("count"): 0}
    mean = float(x.mean())
    sd = float(x.std())
    centered = x - mean
    skew = float(np.mean(centered**3) / sd**3) if sd > 0 else 0.0
    kurtosis = float(np.mean(centered**4) / sd**4 - 3) if sd > 0 else 0.0
    quantiles = np.quantile(x, [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999])
    entropy, effective, hhi = _entropy_effective(x)
    return {
        key("count"): int(x.size),
        key("mean"): mean,
        key("standard_deviation"): sd,
        key("minimum"): float(x.min()),
        key("q001"): float(quantiles[0]),
        key("q01"): float(quantiles[1]),
        key("q05"): float(quantiles[2]),
        key("q25"): float(quantiles[3]),
        key("median"): float(quantiles[4]),
        key("q75"): float(quantiles[5]),
        key("q95"): float(quantiles[6]),
        key("q99"): float(quantiles[7]),
        key("q999"): float(quantiles[8]),
        key("maximum"): float(x.max()),
        key("skewness"): skew,
        key("excess_kurtosis"): kurtosis,
        key("gini_absolute"): _gini(x),
        key("absolute_entropy"): entropy,
        key("absolute_effective_count"): effective,
        key("absolute_hhi"): hhi,
    }


def _robust_z(values: Any) -> Any:
    np = base._require_numpy()
    x = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - median))
    if not math.isfinite(float(mad)) or mad == 0:
        return np.zeros_like(x)
    return 0.6744897501960817 * (x - median) / mad


def _local_robust_z(values: Any, index: int, radius: int = 5) -> float | None:
    np = base._require_numpy()
    x = np.asarray(values, dtype=np.float64)
    neighbors = [
        x[j]
        for j in range(max(0, index - radius), min(x.size, index + radius + 1))
        if j != index and np.isfinite(x[j])
    ]
    if len(neighbors) < 3 or not np.isfinite(x[index]):
        return None
    median = float(np.median(neighbors))
    mad = float(np.median(np.abs(np.asarray(neighbors) - median)))
    if mad == 0:
        return (
            0.0 if float(x[index]) == median else math.copysign(math.inf, float(x[index]) - median)
        )
    return float(0.6744897501960817 * (x[index] - median) / mad)


def _spectrum_shape(values: Any) -> dict[str, Any]:
    np = base._require_numpy()
    singular = np.asarray(values, dtype=np.float64)
    singular = np.maximum(singular[np.isfinite(singular)], 0)
    energy = np.square(singular)
    total = float(energy.sum())
    if singular.size == 0 or total == 0:
        return {"rank": int(singular.size)}
    energy_p = energy / total
    singular_p = singular / singular.sum()
    positive = singular[singular > 0]
    result: dict[str, Any] = {
        "rank": int(singular.size),
        "spectral_gini": _gini(singular),
        "energy_gini": _gini(energy),
        "energy_hhi": float(np.square(energy_p).sum()),
        "singular_hhi": float(np.square(singular_p).sum()),
        "spectral_flatness": float(np.exp(np.log(positive).mean()) / positive.mean())
        if positive.size
        else 0.0,
        "s1_s2_ratio": float(singular[0] / singular[1])
        if singular.size > 1 and singular[1] > 0
        else None,
        "s1_median_ratio": float(singular[0] / np.median(positive)) if positive.size else None,
        "q99_q50_ratio": float(np.quantile(positive, 0.99) / np.quantile(positive, 0.50))
        if positive.size
        else None,
        "q95_q05_ratio": float(np.quantile(positive, 0.95) / np.quantile(positive, 0.05))
        if positive.size and np.quantile(positive, 0.05) > 0
        else None,
    }
    cumulative = np.cumsum(energy_p)
    for rank in SPECTRAL_RANKS:
        if rank <= singular.size:
            result[f"energy_at_{rank}"] = float(cumulative[rank - 1])
            result[f"tail_energy_after_{rank}"] = float(max(0.0, 1 - cumulative[rank - 1]))
    candidate_gaps = [
        rank
        for rank in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        if rank < singular.size and singular[rank] > 0
    ]
    for rank in candidate_gaps:
        result[f"gap_ratio_{rank}"] = float(singular[rank - 1] / singular[rank])
    upper = min(512, singular.size - 1)
    ratios = singular[:upper] / np.maximum(singular[1 : upper + 1], np.finfo(np.float64).tiny)
    result["maximum_gap_rank_top512"] = int(np.argmax(ratios) + 1)
    result["maximum_gap_ratio_top512"] = float(np.max(ratios))

    def fit(start: int, stop: int, log_rank: bool) -> tuple[float | None, float | None]:
        stop = min(stop, positive.size)
        if stop - start < 8:
            return None, None
        ranks = np.arange(start + 1, stop + 1, dtype=np.float64)
        x = np.log(ranks) if log_rank else ranks
        y = np.log(positive[start:stop])
        slope, intercept = np.polyfit(x, y, 1)
        prediction = slope * x + intercept
        denominator = float(np.square(y - y.mean()).sum())
        r2 = 1 - float(np.square(y - prediction).sum()) / denominator if denominator > 0 else 0.0
        return float(slope), float(r2)

    for name, start, stop in (("head", 15, 256), ("bulk", 255, 2048)):
        result[f"powerlaw_{name}_slope"], result[f"powerlaw_{name}_r2"] = fit(start, stop, True)
        result[f"exponential_{name}_slope"], result[f"exponential_{name}_r2"] = fit(
            start, stop, False
        )

    log_s = np.log(np.maximum(positive, np.finfo(np.float64).tiny))
    if log_s.size >= 3:
        x = np.linspace(0, 1, log_s.size)
        line = log_s[0] + x * (log_s[-1] - log_s[0])
        scale = math.hypot(1.0, float(log_s[-1] - log_s[0]))
        distance = np.abs(log_s - line) / scale
        result["log_spectrum_knee_rank"] = int(np.argmax(distance) + 1)
        result["log_spectrum_knee_distance"] = float(distance.max())
    return result


def catalog_baseline(baseline_dir: Path, output_dir: Path) -> None:
    import numpy as np
    import pandas as pd

    artifact_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    array_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    files = sorted(
        path for path in baseline_dir.rglob("*") if path.is_file() and ".progress" not in path.parts
    )
    for path in files:
        relative = path.relative_to(baseline_dir).as_posix()
        kind = path.suffix.lower().lstrip(".") or "none"
        artifact_rows.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path), "kind": kind}
        )
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            quality_rows.extend(
                [
                    {
                        "scope": relative,
                        "check": "row_count",
                        "status": "pass",
                        "value": len(frame),
                        "detail": "",
                    },
                    {
                        "scope": relative,
                        "check": "duplicate_rows",
                        "status": "pass" if not frame.duplicated().any() else "warn",
                        "value": int(frame.duplicated().sum()),
                        "detail": "Exact duplicate rows",
                    },
                    {
                        "scope": relative,
                        "check": "null_cells",
                        "status": "document",
                        "value": int(frame.isna().sum().sum()),
                        "detail": "Nulls are profiled by field below",
                    },
                ]
            )
            for column in frame.columns:
                series = frame[column]
                nonnull = series.dropna()
                row = {
                    "artifact": relative,
                    "field": str(column),
                    "dtype": str(series.dtype),
                    "rows": len(series),
                    "null_count": int(series.isna().sum()),
                    "unique_nonnull": int(nonnull.nunique()),
                    "minimum": None,
                    "maximum": None,
                    "description": describe_field(str(column)),
                }
                if pd.api.types.is_numeric_dtype(series) and len(nonnull):
                    row["minimum"] = float(nonnull.min())
                    row["maximum"] = float(nonnull.max())
                field_rows.append(row)
        elif path.suffix.lower() == ".npz":
            with np.load(path) as archive:
                for name in archive.files:
                    array = archive[name]
                    finite = np.isfinite(array) if np.issubdtype(array.dtype, np.number) else None
                    array_rows.append(
                        {
                            "artifact": relative,
                            "array": name,
                            "shape": "x".join(map(str, array.shape)) or "scalar",
                            "dtype": str(array.dtype),
                            "elements": int(array.size),
                            "bytes_uncompressed": int(array.nbytes),
                            "nonfinite_count": int(array.size - finite.sum())
                            if finite is not None
                            else None,
                            "minimum": float(np.nanmin(array))
                            if finite is not None and array.size
                            else None,
                            "maximum": float(np.nanmax(array))
                            if finite is not None and array.size
                            else None,
                            "description": describe_field(name),
                        }
                    )
    _write_csv(output_dir / "artifact_inventory.csv", artifact_rows)
    _write_csv(output_dir / "field_dictionary.csv", field_rows)
    _write_csv(output_dir / "array_catalog.csv", array_rows)
    _write_csv(output_dir / "data_quality_checks.csv", quality_rows)


def describe_field(name: str) -> str:
    exact = {
        "block": "Zero-based ESMC transformer block index.",
        "produced_state": "One-based ESMC hidden-state index emitted after this block.",
        "state_index": "ESMC hidden-state index, with state 0 denoting the embedding state.",
        "producing_block": "Block that produces the state; null for state 0.",
        "family": "Weight-matrix family: q, k, v, o, gate, value, or down.",
        "checkpoint": "Pinned ESMFold2 checkpoint label.",
        "head": "Zero-based attention-head index.",
        "rank": "Subspace or truncation rank used by the metric.",
        "bh_q": "Benjamini-Hochberg false-discovery-rate adjusted p-value within the declared metric family.",
        "exact_circular_shift_p": "Exact two-sided null p-value over all unique nonzero circular depth shifts.",
    }
    if name in exact:
        return exact[name]
    replacements = {
        "frobenius": "Frobenius",
        "spectral": "spectral",
        "effective_rank": "entropy-based effective rank",
        "stable_rank": "stable rank ||W||_F^2 / ||W||_2^2",
        "participation_ratio": "energy participation ratio",
        "overlap": "normalized squared subspace overlap",
        "angle_degrees": "principal angle in degrees",
        "relative_error": "relative Frobenius reconstruction error",
        "storage_fraction": "estimated stored parameter fraction",
        "mixing_weight": "softmax-normalized ESMFold2 state-mixing coefficient",
    }
    for token, description in replacements.items():
        if token in name:
            return f"{description}; see metric name for qualifiers."
    return "Machine-readable analysis field; definition follows the literal field name and source table grain."


def derive_spectral_shapes(baseline_dir: Path, output_dir: Path) -> None:
    import numpy as np

    rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    by_family_mode: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for path in sorted((baseline_dir / "spectra").glob("block_*_*.npz")):
        stem = path.stem.split("_")
        block = int(stem[1])
        family = stem[2]
        with np.load(path) as archive:
            for mode in ("operator", "rows", "columns"):
                spectrum = archive[f"{mode}_singular_values"].astype(np.float64)
                rows.append(
                    {
                        "block": block,
                        "produced_state": block + 1,
                        "family": family,
                        "geometry": mode,
                        **_spectrum_shape(spectrum),
                    }
                )
                by_family_mode[(family, mode)].append(spectrum)
    for (family, mode), spectra in sorted(by_family_mode.items()):
        if len(spectra) != base.N_BLOCKS:
            raise DeepDiveError(f"Expected 80 spectra for {family}/{mode}, found {len(spectra)}")
        normalized_energy = [np.square(x) / np.square(x).sum() for x in spectra]
        normalized_singular = [x / x.sum() for x in spectra]
        for block in range(1, base.N_BLOCKS):
            left_e = normalized_energy[block - 1]
            right_e = normalized_energy[block]
            midpoint = 0.5 * (left_e + right_e)
            mask_l = left_e > 0
            mask_r = right_e > 0
            js = 0.5 * float((left_e[mask_l] * np.log(left_e[mask_l] / midpoint[mask_l])).sum())
            js += 0.5 * float((right_e[mask_r] * np.log(right_e[mask_r] / midpoint[mask_r])).sum())
            cosine = _safe_corr(normalized_singular[block - 1], normalized_singular[block])
            l1 = float(np.abs(left_e - right_e).sum())
            wasserstein_rank = float(np.abs(np.cumsum(left_e) - np.cumsum(right_e)).sum())
            transition_rows.append(
                {
                    "from_block": block - 1,
                    "to_block": block,
                    "family": family,
                    "geometry": mode,
                    "energy_jensen_shannon": js,
                    "energy_total_variation": 0.5 * l1,
                    "energy_rank_wasserstein": wasserstein_rank,
                    "singular_profile_pearson": cosine,
                    "singular_profile_cosine": float(
                        np.dot(normalized_singular[block - 1], normalized_singular[block])
                        / (
                            np.linalg.norm(normalized_singular[block - 1])
                            * np.linalg.norm(normalized_singular[block])
                        )
                    ),
                }
            )
    _write_csv(output_dir / "spectral_shape_metrics.csv", rows)
    _write_csv(output_dir / "adjacent_spectral_transitions.csv", transition_rows)


def derive_trajectory_metrics(baseline_dir: Path, output_dir: Path) -> None:
    import numpy as np

    rows: list[dict[str, Any]] = []
    eigen_rows: list[dict[str, Any]] = []
    coordinate_archive: dict[str, Any] = {}
    for path in sorted((baseline_dir / "trajectory").glob("*.npz")):
        family = path.stem
        with np.load(path) as archive:
            gram = archive["gram"].astype(np.float64)
            distances = archive["distances"].astype(np.float64)
            cosine = archive["cosine"].astype(np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0)
        eigenvectors = eigenvectors[:, order]
        coordinates = eigenvectors * np.sqrt(eigenvalues)[None, :]
        coordinate_archive[f"{family}_eigenvalues"] = eigenvalues
        coordinate_archive[f"{family}_coordinates"] = coordinates
        total_eigen = float(eigenvalues.sum())
        for component, value in enumerate(eigenvalues, start=1):
            eigen_rows.append(
                {
                    "family": family,
                    "component": component,
                    "eigenvalue": float(value),
                    "explained_fraction": float(value / total_eigen) if total_eigen else 0.0,
                    "cumulative_fraction": float(eigenvalues[:component].sum() / total_eigen)
                    if total_eigen
                    else 0.0,
                }
            )
        speed = np.full(base.N_BLOCKS, np.nan)
        speed[1:] = np.diag(distances, k=1)
        for block in range(base.N_BLOCKS):
            vector = coordinates[block]
            row: dict[str, Any] = {
                "block": block,
                "produced_state": block + 1,
                "family": family,
                "radius_from_trajectory_centroid": float(np.linalg.norm(vector)),
                "previous_frobenius_distance": float(speed[block]) if block else None,
                "previous_cosine_similarity": float(cosine[block - 1, block]) if block else None,
                "next_frobenius_distance": float(speed[block + 1])
                if block + 1 < base.N_BLOCKS
                else None,
                "next_cosine_similarity": float(cosine[block, block + 1])
                if block + 1 < base.N_BLOCKS
                else None,
            }
            for component in range(min(10, coordinates.shape[1])):
                row[f"kpca_coordinate_{component + 1}"] = float(coordinates[block, component])
            nonself = distances[block].copy()
            nonself[block] = np.inf
            nearest = int(np.argmin(nonself))
            row["nearest_layer"] = nearest
            row["nearest_layer_depth_gap"] = abs(nearest - block)
            row["nearest_layer_distance"] = float(nonself[nearest])
            if 0 < block < base.N_BLOCKS - 1:
                before = coordinates[block] - coordinates[block - 1]
                after = coordinates[block + 1] - coordinates[block]
                denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
                turn_cosine = float(np.dot(before, after) / denominator) if denominator else None
                row["turning_cosine"] = turn_cosine
                row["turning_angle_degrees"] = (
                    float(np.degrees(np.arccos(np.clip(turn_cosine, -1, 1))))
                    if turn_cosine is not None
                    else None
                )
                row["acceleration_norm"] = float(np.linalg.norm(after - before))
                row["local_path_to_chord_ratio"] = (
                    float(
                        (np.linalg.norm(before) + np.linalg.norm(after))
                        / np.linalg.norm(coordinates[block + 1] - coordinates[block - 1])
                    )
                    if np.linalg.norm(coordinates[block + 1] - coordinates[block - 1])
                    else None
                )
            rows.append(row)
    np.savez_compressed(output_dir / "trajectory_coordinates.npz", **coordinate_archive)
    _write_csv(output_dir / "trajectory_local_geometry.csv", rows)
    _write_csv(output_dir / "trajectory_eigenspectra.csv", eigen_rows)


def derive_head_metrics(baseline_dir: Path, output_dir: Path) -> None:
    import numpy as np
    import pandas as pd

    heads = pd.read_csv(baseline_dir / "attention_head_metrics.csv")
    metric_columns = [
        column for column in heads.columns if column not in {"block", "produced_state", "head"}
    ]
    summary_rows: list[dict[str, Any]] = []
    for block, group in heads.groupby("block", sort=True):
        row: dict[str, Any] = {"block": int(block), "produced_state": int(block) + 1}
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=np.float64)
            stats = _distribution(values, prefix=f"{metric}__")
            row.update(stats)
            if np.all(values >= 0):
                total = float(values.sum())
                if total > 0:
                    row[f"{metric}__top1_share"] = float(np.max(values) / total)
                    row[f"{metric}__top5_share"] = float(np.sort(values)[-5:].sum() / total)
        summary_rows.append(row)

    transitions = pd.read_csv(baseline_dir / "attention_head_transitions.csv")
    permutation_rows: list[dict[str, Any]] = []
    for record in transitions.to_dict(orient="records"):
        decoded = json.loads(record["head_permutation"])
        if isinstance(decoded, dict):
            permutation = np.asarray(
                [decoded[str(index)] for index in range(len(decoded))], dtype=int
            )
        else:
            permutation = np.asarray(decoded, dtype=int)
        visited = np.zeros(permutation.size, dtype=bool)
        cycles: list[int] = []
        for start in range(permutation.size):
            if visited[start]:
                continue
            current = start
            length = 0
            while not visited[current]:
                visited[current] = True
                current = int(permutation[current])
                length += 1
            cycles.append(length)
        displacement = np.abs(permutation - np.arange(permutation.size))
        permutation_rows.append(
            {
                **record,
                "matching_gain": float(
                    record["hungarian_similarity_mean"] - record["fixed_index_similarity_mean"]
                ),
                "fixed_points": int(np.sum(permutation == np.arange(permutation.size))),
                "cycle_count": len(cycles),
                "maximum_cycle_length": max(cycles),
                "mean_cycle_length": float(np.mean(cycles)),
                "mean_absolute_head_displacement": float(displacement.mean()),
                "maximum_head_displacement": int(displacement.max()),
                "permutation_inversion_count": int(
                    sum(
                        permutation[i] > permutation[j]
                        for i in range(permutation.size)
                        for j in range(i + 1, permutation.size)
                    )
                ),
            }
        )
    _write_csv(output_dir / "attention_head_distribution_summary.csv", summary_rows)
    _write_csv(output_dir / "attention_head_permutation_geometry.csv", permutation_rows)


def derive_mixer_and_compression(baseline_dir: Path, output_dir: Path) -> None:
    import numpy as np
    import pandas as pd

    mixer = pd.read_csv(baseline_dir / "esmfold2_mixing_weights.csv")
    mixer_rows: list[dict[str, Any]] = []
    for checkpoint, group in mixer.groupby("checkpoint", sort=True):
        group = group.sort_values("state_index")
        weights = group["mixing_weight"].to_numpy(dtype=np.float64)
        states = group["state_index"].to_numpy(dtype=np.float64)
        entropy, effective, hhi = _entropy_effective(weights)
        order = np.argsort(weights)[::-1]
        mixer_rows.append(
            {
                "checkpoint": checkpoint,
                "entropy": entropy,
                "effective_state_count": effective,
                "hhi": hhi,
                "gini": _gini(weights),
                "top1_mass": float(weights[order[:1]].sum()),
                "top4_mass": float(weights[order[:4]].sum()),
                "top8_mass": float(weights[order[:8]].sum()),
                "states_77_80_mass": float(weights[77:81].sum()),
                "state_51_mass": float(weights[51]),
                "state_51_rank": int(np.where(order == 51)[0][0] + 1),
                "depth_center_of_mass": float(np.dot(states, weights)),
                "depth_standard_deviation": float(
                    np.sqrt(np.dot(np.square(states - np.dot(states, weights)), weights))
                ),
                "adjacent_total_variation": float(np.abs(np.diff(weights)).sum()),
                "maximum_weight_state": int(np.argmax(weights)),
            }
        )
    pair_rows: list[dict[str, Any]] = []
    for left_name, right_name in combinations(sorted(mixer["checkpoint"].unique()), 2):
        left = (
            mixer[mixer["checkpoint"] == left_name]
            .sort_values("state_index")["mixing_weight"]
            .to_numpy(dtype=float)
        )
        right = (
            mixer[mixer["checkpoint"] == right_name]
            .sort_values("state_index")["mixing_weight"]
            .to_numpy(dtype=float)
        )
        midpoint = 0.5 * (left + right)
        js = 0.5 * float(
            (left[left > 0] * np.log(left[left > 0] / midpoint[left > 0])).sum()
        ) + 0.5 * float((right[right > 0] * np.log(right[right > 0] / midpoint[right > 0])).sum())
        pair_rows.append(
            {
                "checkpoint_left": left_name,
                "checkpoint_right": right_name,
                "pearson": _safe_corr(left, right),
                "spearman": _safe_corr(left, right, method="spearman"),
                "cosine": float(
                    np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
                ),
                "jensen_shannon": js,
                "hellinger": float(np.linalg.norm(np.sqrt(left) - np.sqrt(right)) / math.sqrt(2)),
                "total_variation": float(0.5 * np.abs(left - right).sum()),
                "rank_wasserstein": float(np.abs(np.cumsum(left) - np.cumsum(right)).sum()),
            }
        )
    _write_csv(output_dir / "mixer_profile_summary.csv", mixer_rows)
    _write_csv(output_dir / "mixer_profile_pairwise.csv", pair_rows)

    compression = pd.read_csv(baseline_dir / "compression_metrics.csv")
    compression["effective_storage_fraction"] = compression["storage_fraction"].fillna(
        compression["ideal_value_storage_fraction"]
    )
    pareto_rows: list[dict[str, Any]] = []
    for (block, family), group in compression.groupby(["block", "family"], sort=True):
        valid = group.dropna(
            subset=["effective_storage_fraction", "relative_frobenius_error"]
        ).copy()
        valid = valid.sort_values(["effective_storage_fraction", "relative_frobenius_error"])
        best_error = math.inf
        for record in valid.to_dict(orient="records"):
            is_pareto = float(record["relative_frobenius_error"]) < best_error
            if is_pareto:
                best_error = float(record["relative_frobenius_error"])
            pareto_rows.append(
                {
                    "block": int(block),
                    "produced_state": int(block) + 1,
                    "family": family,
                    "method": record["method"],
                    "reported_storage_fraction": record["storage_fraction"],
                    "ideal_value_storage_fraction": record["ideal_value_storage_fraction"],
                    "effective_storage_fraction": record["effective_storage_fraction"],
                    "relative_frobenius_error": record["relative_frobenius_error"],
                    "spectral_distortion": record.get("spectral_distortion"),
                    "flattened_cosine": record.get("flattened_cosine"),
                    "pareto_optimal": int(is_pareto),
                }
            )
    _write_csv(output_dir / "compression_pareto.csv", pareto_rows)


def _sample_flat(matrix: Any, maximum: int = RAW_QUANTILE_SAMPLE) -> Any:
    torch = base._require_torch()
    flat = matrix.detach().reshape(-1).cpu()
    if flat.numel() <= maximum:
        return flat
    indices = (
        torch.linspace(0, flat.numel() - 1, maximum, dtype=torch.float64).round().to(torch.long)
    )
    return flat[indices]


def _coherence_metrics(matrix: Any, *, orientation: str, device: Any) -> dict[str, Any]:
    torch = base._require_torch()
    vectors = matrix if orientation == "rows" else matrix.T
    count = int(vectors.shape[0])
    sample_count = min(COHERENCE_SAMPLE, count)
    indices = torch.linspace(0, count - 1, sample_count, dtype=torch.float64).round().to(torch.long)
    sample = vectors[indices].to(device=device, dtype=torch.float32)
    sample = torch.nn.functional.normalize(sample, dim=1)
    gram = sample @ sample.T
    mask = ~torch.eye(sample_count, dtype=torch.bool, device=device)
    off = gram[mask]
    absolute = off.abs()
    return {
        f"{orientation}_coherence_sample_size": sample_count,
        f"{orientation}_coherence_mean_absolute": float(absolute.mean().cpu()),
        f"{orientation}_coherence_q95_absolute": float(torch.quantile(absolute, 0.95).cpu()),
        f"{orientation}_coherence_max_absolute": float(absolute.max().cpu()),
        f"{orientation}_mean_squared_cosine": float(torch.square(off).mean().cpu()),
    }


def _raw_matrix_metrics(matrix: Any, *, device: Any) -> dict[str, Any]:
    import numpy as np

    torch = base._require_torch()
    sample = _sample_flat(matrix).numpy().astype(np.float64, copy=False)
    exact = matrix.detach().cpu().numpy()
    mean = float(np.mean(exact, dtype=np.float64))
    square_mean = float(np.mean(np.square(exact, dtype=np.float64), dtype=np.float64))
    sd = math.sqrt(max(0.0, square_mean - mean * mean))
    centered = sample - float(sample.mean())
    sample_sd = float(sample.std())
    absolute = np.abs(sample)
    rms = math.sqrt(square_mean)
    result: dict[str, Any] = {
        "parameter_count": int(matrix.numel()),
        "mean": mean,
        "standard_deviation": sd,
        "rms": rms,
        "minimum_sampled": float(sample.min()),
        "maximum_sampled": float(sample.max()),
        "positive_fraction_sampled": float(np.mean(sample > 0)),
        "zero_fraction_exact": float(np.mean(exact == 0)),
        "near_zero_1e3_rms_fraction_sampled": float(np.mean(absolute <= 1e-3 * rms))
        if rms
        else 1.0,
        "near_zero_1e2_rms_fraction_sampled": float(np.mean(absolute <= 1e-2 * rms))
        if rms
        else 1.0,
        "skewness_sampled": float(np.mean(centered**3) / sample_sd**3) if sample_sd else 0.0,
        "excess_kurtosis_sampled": float(np.mean(centered**4) / sample_sd**4 - 3)
        if sample_sd
        else 0.0,
        "quantile_sample_size": int(sample.size),
    }
    for label, quantile in (
        ("001", 0.001),
        ("01", 0.01),
        ("05", 0.05),
        ("25", 0.25),
        ("50", 0.5),
        ("75", 0.75),
        ("95", 0.95),
        ("99", 0.99),
        ("999", 0.999),
    ):
        result[f"absolute_q{label}_sampled"] = float(np.quantile(absolute, quantile))
    matrix_device = matrix.to(device=device, dtype=torch.float32)
    row_norms = torch.linalg.vector_norm(matrix_device, dim=1).cpu().numpy().astype(np.float64)
    column_norms = torch.linalg.vector_norm(matrix_device, dim=0).cpu().numpy().astype(np.float64)
    result.update(_distribution(row_norms, "row_norm__"))
    result.update(_distribution(column_norms, "column_norm__"))
    result.update(_coherence_metrics(matrix, orientation="rows", device=device))
    result.update(_coherence_metrics(matrix, orientation="columns", device=device))
    del matrix_device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def randomized_svd(
    matrix: Any,
    rank: int,
    *,
    device: Any,
    seed: int,
    oversample: int = 16,
    power_iterations: int = 2,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Deterministic randomized SVD with explicit approximation diagnostics."""

    torch = base._require_torch()
    value = matrix.to(device=device, dtype=torch.float32)
    target = min(int(rank), min(value.shape))
    sketch_rank = min(target + oversample, min(value.shape))
    generator = torch.Generator(device=device).manual_seed(seed)
    omega = torch.randn(
        value.shape[1], sketch_rank, generator=generator, device=device, dtype=torch.float32
    )
    sketch = value @ omega
    for _ in range(power_iterations):
        sketch = value @ (value.T @ sketch)
        sketch, _ = torch.linalg.qr(sketch, mode="reduced")
    q_basis, _ = torch.linalg.qr(sketch, mode="reduced")
    compressed = q_basis.T @ value
    left_small, singular, right_h = torch.linalg.svd(compressed, full_matrices=False)
    left = (q_basis @ left_small[:, :target]).contiguous()
    singular = singular[:target].contiguous()
    right = right_h[:target].T.contiguous()
    frobenius_sq = float(torch.square(value).sum().cpu())
    captured_sq = float(torch.square(singular).sum().cpu())
    diagnostics = {
        "target_rank": target,
        "sketch_rank": sketch_rank,
        "power_iterations": power_iterations,
        "captured_frobenius_energy_fraction": captured_sq / frobenius_sq if frobenius_sq else 0.0,
        "left_orthogonality_error": float(
            torch.linalg.matrix_norm(
                left.T @ left - torch.eye(target, device=device), ord="fro"
            ).cpu()
        ),
        "right_orthogonality_error": float(
            torch.linalg.matrix_norm(
                right.T @ right - torch.eye(target, device=device), ord="fro"
            ).cpu()
        ),
    }
    return left.cpu(), singular.cpu(), right.cpu(), diagnostics


def _basis_localization(basis: Any, prefix: str) -> dict[str, Any]:
    import numpy as np

    x = basis.detach().cpu().numpy().astype(np.float64)
    squared = np.square(x)
    ipr = np.square(squared).sum(axis=0)
    maximum = squared.max(axis=0)
    entropy = -(squared * np.log(np.maximum(squared, np.finfo(np.float64).tiny))).sum(axis=0)
    return {
        f"{prefix}_mean_ipr": float(ipr.mean()),
        f"{prefix}_maximum_ipr": float(ipr.max()),
        f"{prefix}_mean_effective_support": float(np.mean(1 / ipr)),
        f"{prefix}_minimum_effective_support": float(np.min(1 / ipr)),
        f"{prefix}_mean_maximum_coordinate_energy": float(maximum.mean()),
        f"{prefix}_maximum_coordinate_energy": float(maximum.max()),
        f"{prefix}_mean_entropy": float(entropy.mean()),
        f"{prefix}_mean_entropy_effective_support": float(np.exp(entropy).mean()),
    }


def run_raw_matrix_pass(
    checkpoint: base.SafetensorCheckpoint,
    output_dir: Path,
    *,
    device: Any,
    top_rank: int,
    resume: bool,
) -> None:
    import numpy as np

    torch = base._require_torch()
    progress = output_dir / ".progress" / "raw"
    bases_dir = output_dir / "bases"
    ffn_dir = output_dir / "ffn_neurons"
    progress.mkdir(parents=True, exist_ok=True)
    bases_dir.mkdir(parents=True, exist_ok=True)
    ffn_dir.mkdir(parents=True, exist_ok=True)
    for block in range(base.N_BLOCKS):
        marker = progress / f"block_{block:02d}.json"
        if resume and marker.is_file():
            continue
        qkv = checkpoint.tensor(base._block_key(block, "attn_qkv"))
        fc1 = checkpoint.tensor(base._block_key(block, "ffn_fc1"))
        output = checkpoint.tensor(base._block_key(block, "attn_output"))
        down = checkpoint.tensor(base._block_key(block, "ffn_down"))
        matrices = {
            family: base._split_matrix(family, qkv, fc1, output, down)
            for family in base.MATRIX_FAMILIES
        }
        block_rows: list[dict[str, Any]] = []
        sensitivity_rows: list[dict[str, Any]] = []
        for family, matrix in matrices.items():
            row = {"block": block, "produced_state": block + 1, "family": family}
            row.update(_raw_matrix_metrics(matrix, device=device))
            left, singular, right, diagnostics = randomized_svd(
                matrix,
                top_rank,
                device=device,
                seed=17011 + block * 101 + FAMILY_INDEX[family],
            )
            row.update(diagnostics)
            row.update(_basis_localization(left, "left_singular_vectors"))
            row.update(_basis_localization(right, "right_singular_vectors"))
            np.savez_compressed(
                bases_dir / f"block_{block:02d}_{family}.npz",
                left=left.numpy().astype(np.float32),
                singular_values=singular.numpy().astype(np.float32),
                right=right.numpy().astype(np.float32),
            )
            if block in {0, 50, 51, 79}:
                left_check, singular_check, right_check, check_diagnostics = randomized_svd(
                    matrix,
                    top_rank,
                    device=device,
                    seed=29009 + block * 101 + FAMILY_INDEX[family],
                    oversample=32,
                    power_iterations=4,
                )
                singular_relative = torch.abs(singular - singular_check) / torch.clamp(
                    singular_check, min=torch.finfo(singular_check.dtype).tiny
                )
                sensitivity = {
                    "block": block,
                    "produced_state": block + 1,
                    "family": family,
                    "maximum_singular_value_relative_difference": float(singular_relative.max()),
                    "median_singular_value_relative_difference": float(singular_relative.median()),
                    "primary_captured_energy": diagnostics["captured_frobenius_energy_fraction"],
                    "sensitivity_captured_energy": check_diagnostics[
                        "captured_frobenius_energy_fraction"
                    ],
                }
                for side, primary_basis, check_basis in (
                    ("left", left.numpy(), left_check.numpy()),
                    ("right", right.numpy(), right_check.numpy()),
                ):
                    for metrics in _subspace_row(primary_basis, check_basis, (16, 32, 64)):
                        sensitivity_rows.append({**sensitivity, "side": side, **metrics})
                del left_check, singular_check, right_check
            block_rows.append(row)
            del left, singular, right
            if device.type == "cuda":
                torch.cuda.empty_cache()

        gate = matrices["gate"].to(device=device, dtype=torch.float32)
        value = matrices["value"].to(device=device, dtype=torch.float32)
        down_t = matrices["down"].T.to(device=device, dtype=torch.float32)
        gate_norm = torch.linalg.vector_norm(gate, dim=1)
        value_norm = torch.linalg.vector_norm(value, dim=1)
        down_norm = torch.linalg.vector_norm(down_t, dim=1)
        gate_value_cos = torch.nn.functional.cosine_similarity(gate, value, dim=1)
        gate_down_cos = torch.nn.functional.cosine_similarity(gate, down_t, dim=1)
        value_down_cos = torch.nn.functional.cosine_similarity(value, down_t, dim=1)
        triple_strength = gate_norm * value_norm * down_norm
        np.savez_compressed(
            ffn_dir / f"block_{block:02d}.npz",
            neuron=np.arange(base.FFN_WIDTH, dtype=np.int32),
            gate_norm=gate_norm.cpu().numpy().astype(np.float32),
            value_norm=value_norm.cpu().numpy().astype(np.float32),
            down_norm=down_norm.cpu().numpy().astype(np.float32),
            gate_value_cosine=gate_value_cos.cpu().numpy().astype(np.float32),
            gate_down_cosine=gate_down_cos.cpu().numpy().astype(np.float32),
            value_down_cosine=value_down_cos.cpu().numpy().astype(np.float32),
            triple_strength=triple_strength.cpu().numpy().astype(np.float32),
        )
        arrays = [
            x.cpu().numpy().astype(np.float64)
            for x in (
                gate_norm,
                value_norm,
                down_norm,
                gate_value_cos,
                gate_down_cos,
                value_down_cos,
                triple_strength,
            )
        ]
        names = (
            "gate_norm",
            "value_norm",
            "down_norm",
            "gate_value_cosine",
            "gate_down_cosine",
            "value_down_cosine",
            "triple_strength",
        )
        ffn_row: dict[str, Any] = {"block": block, "produced_state": block + 1}
        for name, array in zip(names, arrays, strict=True):
            ffn_row.update(_distribution(array, f"{name}__"))
        for (left_name, left_values), (right_name, right_values) in combinations(
            zip(names[:3], arrays[:3], strict=False), 2
        ):
            ffn_row[f"{left_name}_{right_name}_pearson"] = _safe_corr(left_values, right_values)
            ffn_row[f"{left_name}_{right_name}_spearman"] = _safe_corr(
                left_values, right_values, method="spearman"
            )
        strength = arrays[-1]
        order = np.argsort(strength)[::-1]
        for count in (1, 10, 50, 100, 500, 1000):
            ffn_row[f"triple_strength_top{count}_share"] = float(
                strength[order[:count]].sum() / strength.sum()
            )
        _atomic_json(
            marker,
            {
                "raw_tensor_rows": block_rows,
                "ffn_summary": ffn_row,
                "randomized_svd_sensitivity": sensitivity_rows,
            },
        )
        del qkv, fc1, output, down, matrices, gate, value, down_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_rows: list[dict[str, Any]] = []
    ffn_rows: list[dict[str, Any]] = []
    sensitivity_rows = []
    for marker in sorted(progress.glob("block_*.json")):
        payload = json.loads(marker.read_text(encoding="utf-8"))
        raw_rows.extend(payload["raw_tensor_rows"])
        ffn_rows.append(payload["ffn_summary"])
        sensitivity_rows.extend(payload.get("randomized_svd_sensitivity", []))
    _write_csv(output_dir / "raw_tensor_statistics.csv", raw_rows)
    _write_csv(output_dir / "ffn_neuron_summary.csv", ffn_rows)
    _write_csv(output_dir / "randomized_svd_sensitivity.csv", sensitivity_rows)


def run_normalization_pass(checkpoint: base.SafetensorCheckpoint, output_dir: Path) -> None:
    import numpy as np

    role_arrays: dict[str, Any] = {}
    vector_rows: list[dict[str, Any]] = []
    for role in VECTOR_ROLES:
        vectors = []
        for block in range(base.N_BLOCKS):
            vector = (
                checkpoint.tensor(base._block_key(block, role))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            vectors.append(vector)
            vector_rows.append(
                {"block": block, "produced_state": block + 1, "role": role, **_distribution(vector)}
            )
        role_arrays[role] = np.stack(vectors)
    final_name = checkpoint.find_unique_suffix("transformer.norm.weight")
    role_arrays["final_transformer_norm_weight"] = (
        checkpoint.tensor(final_name).detach().cpu().numpy().astype(np.float64)
    )
    np.savez_compressed(
        output_dir / "normalization_vectors.npz",
        **{key: value.astype(np.float32) for key, value in role_arrays.items()},
    )

    channel_rows: list[dict[str, Any]] = []
    adjacent_rows: list[dict[str, Any]] = []
    cross_role_rows: list[dict[str, Any]] = []
    for role in VECTOR_ROLES:
        matrix = role_arrays[role]
        for channel in range(matrix.shape[1]):
            values = matrix[:, channel]
            z = _robust_z(values)
            channel_rows.append(
                {
                    "role": role,
                    "channel": channel,
                    "depth_mean": float(values.mean()),
                    "depth_standard_deviation": float(values.std()),
                    "depth_minimum": float(values.min()),
                    "depth_maximum": float(values.max()),
                    "lag1_pearson": _safe_corr(values[:-1], values[1:]),
                    "block_50_value": float(values[50]),
                    "block_50_global_robust_z": float(z[50]),
                    "block_51_value": float(values[51]),
                    "block_51_global_robust_z": float(z[51]),
                    "maximum_absolute_step": float(np.max(np.abs(np.diff(values)))),
                    "maximum_step_to_block": int(np.argmax(np.abs(np.diff(values))) + 1),
                }
            )
        for block in range(1, base.N_BLOCKS):
            before = matrix[block - 1]
            after = matrix[block]
            delta = after - before
            adjacent_rows.append(
                {
                    "from_block": block - 1,
                    "to_block": block,
                    "role": role,
                    "cosine": float(
                        np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after))
                    )
                    if np.linalg.norm(before) and np.linalg.norm(after)
                    else None,
                    "relative_l2_change": float(np.linalg.norm(delta) / np.linalg.norm(before))
                    if np.linalg.norm(before)
                    else None,
                    "maximum_absolute_channel_change": float(np.max(np.abs(delta))),
                    "mean_absolute_channel_change": float(np.mean(np.abs(delta))),
                    "changed_channel_fraction_above_1pct_previous_rms": float(
                        np.mean(np.abs(delta) > 0.01 * np.sqrt(np.mean(np.square(before))))
                    ),
                }
            )
    for block in range(base.N_BLOCKS):
        for left, right in combinations(VECTOR_ROLES, 2):
            cross_role_rows.append(
                {
                    "block": block,
                    "produced_state": block + 1,
                    "role_left": left,
                    "role_right": right,
                    "pearson": _safe_corr(role_arrays[left][block], role_arrays[right][block]),
                    "cosine": float(
                        np.dot(role_arrays[left][block], role_arrays[right][block])
                        / (
                            np.linalg.norm(role_arrays[left][block])
                            * np.linalg.norm(role_arrays[right][block])
                        )
                    )
                    if np.linalg.norm(role_arrays[left][block])
                    and np.linalg.norm(role_arrays[right][block])
                    else None,
                }
            )
    _write_csv(output_dir / "normalization_vector_statistics.csv", vector_rows)
    _write_csv(output_dir / "normalization_channel_depth_metrics.csv", channel_rows)
    _write_csv(output_dir / "normalization_adjacent_changes.csv", adjacent_rows)
    _write_csv(output_dir / "normalization_cross_role.csv", cross_role_rows)


def _load_basis(output_dir: Path, block: int, family: str) -> dict[str, Any]:
    import numpy as np

    with np.load(output_dir / "bases" / f"block_{block:02d}_{family}.npz") as archive:
        return {name: archive[name].astype(np.float64) for name in archive.files}


def _subspace_row(left: Any, right: Any, ranks: Iterable[int]) -> list[dict[str, Any]]:
    import numpy as np

    rows = []
    maximum = min(left.shape[1], right.shape[1])
    for rank in ranks:
        use = min(rank, maximum)
        singular = np.linalg.svd(left[:, :use].T @ right[:, :use], compute_uv=False)
        squared = np.square(np.clip(singular, 0, 1))
        angles = np.degrees(np.arccos(np.clip(singular, -1, 1)))
        rows.append(
            {
                "rank": use,
                "normalized_overlap": float(squared.mean()),
                "minimum_angle_degrees": float(angles.min()),
                "median_angle_degrees": float(np.median(angles)),
                "maximum_angle_degrees": float(angles.max()),
                "minimum_canonical_correlation": float(singular.min()),
                "median_canonical_correlation": float(np.median(singular)),
                "maximum_canonical_correlation": float(singular.max()),
                "chordal_distance": float(np.sqrt(np.maximum(0, use - squared.sum()))),
            }
        )
    return rows


def run_subspace_pass(
    output_dir: Path, folds: Sequence[base.FoldWeights], *, top_rank: int
) -> None:
    import numpy as np

    ranks = tuple(rank for rank in (16, 32, 64) if rank <= top_rank)
    if not ranks:
        ranks = (top_rank,)
    within_rows: list[dict[str, Any]] = []
    adjacent_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    read_families = ("q", "k", "v", "gate", "value")
    projection_spaces: dict[tuple[str, str], Any] = {}
    for fold in folds:
        projection = fold.projection.detach().cpu().numpy().astype(np.float64)
        _, _, right_h = np.linalg.svd(projection, full_matrices=False)
        projection_spaces[(fold.label, "raw")] = right_h.T
        scaled_projection = (
            projection * fold.norm_weight.detach().cpu().numpy().astype(np.float64)[None, :]
        )
        _, _, scaled_right_h = np.linalg.svd(scaled_projection, full_matrices=False)
        projection_spaces[(fold.label, "layernorm_scaled_approximation")] = scaled_right_h.T
    for block in range(base.N_BLOCKS):
        bases = {family: _load_basis(output_dir, block, family) for family in base.MATRIX_FAMILIES}
        for left_family, right_family in combinations(read_families, 2):
            for metrics in _subspace_row(
                bases[left_family]["right"], bases[right_family]["right"], ranks
            ):
                within_rows.append(
                    {
                        "block": block,
                        "produced_state": block + 1,
                        "comparison": "residual_read",
                        "family_left": left_family,
                        "family_right": right_family,
                        **metrics,
                    }
                )
        for metrics in _subspace_row(bases["o"]["left"], bases["down"]["left"], ranks):
            within_rows.append(
                {
                    "block": block,
                    "produced_state": block + 1,
                    "comparison": "residual_write",
                    "family_left": "o",
                    "family_right": "down",
                    **metrics,
                }
            )
        for write_family, read_family in (
            ("o", "q"),
            ("o", "k"),
            ("o", "v"),
            ("o", "gate"),
            ("o", "value"),
            ("down", "q"),
            ("down", "k"),
            ("down", "v"),
            ("down", "gate"),
            ("down", "value"),
        ):
            for metrics in _subspace_row(
                bases[write_family]["left"], bases[read_family]["right"], ranks
            ):
                within_rows.append(
                    {
                        "block": block,
                        "produced_state": block + 1,
                        "comparison": "write_to_read",
                        "family_left": write_family,
                        "family_right": read_family,
                        **metrics,
                    }
                )

        for fold in folds:
            for variant in ("raw", "layernorm_scaled_approximation"):
                projection_space = projection_spaces[(fold.label, variant)]
                for family in base.MATRIX_FAMILIES:
                    side = "left" if family in {"o", "down"} else "right"
                    for metrics in _subspace_row(bases[family][side], projection_space, ranks):
                        projection_rows.append(
                            {
                                "block": block,
                                "produced_state": block + 1,
                                "checkpoint": fold.label,
                                "projection_variant": variant,
                                "family": family,
                                "side": side,
                                **metrics,
                            }
                        )
    for family in base.MATRIX_FAMILIES:
        previous = _load_basis(output_dir, 0, family)
        for block in range(1, base.N_BLOCKS):
            current = _load_basis(output_dir, block, family)
            for side in ("left", "right"):
                for metrics in _subspace_row(previous[side], current[side], ranks):
                    adjacent_rows.append(
                        {
                            "from_block": block - 1,
                            "to_block": block,
                            "family": family,
                            "side": side,
                            **metrics,
                        }
                    )
            singular_left = previous["singular_values"]
            singular_right = current["singular_values"]
            adjacent_rows.append(
                {
                    "from_block": block - 1,
                    "to_block": block,
                    "family": family,
                    "side": "singular_values",
                    "rank": len(singular_left),
                    "normalized_overlap": float(
                        np.dot(singular_left, singular_right)
                        / (np.linalg.norm(singular_left) * np.linalg.norm(singular_right))
                    ),
                    "minimum_angle_degrees": None,
                    "median_angle_degrees": None,
                    "maximum_angle_degrees": None,
                    "minimum_canonical_correlation": None,
                    "median_canonical_correlation": None,
                    "maximum_canonical_correlation": None,
                    "chordal_distance": None,
                }
            )
            previous = current
    _write_csv(output_dir / "within_block_subspace_geometry.csv", within_rows)
    _write_csv(output_dir / "adjacent_subspace_turnover.csv", adjacent_rows)
    _write_csv(output_dir / "projection_alignment_randomized_svd.csv", projection_rows)


def run_projection_checkpoint_pass(folds: Sequence[base.FoldWeights], output_dir: Path) -> None:
    import numpy as np

    rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    bases: dict[tuple[str, str], Any] = {}
    projections: dict[tuple[str, str], Any] = {}
    for fold in folds:
        projection = fold.projection.detach().cpu().numpy().astype(np.float64)
        norm_weight = fold.norm_weight.detach().cpu().numpy().astype(np.float64)
        norm_bias = fold.norm_bias.detach().cpu().numpy().astype(np.float64)
        for role, vector in (("layernorm_weight", norm_weight), ("layernorm_bias", norm_bias)):
            vector_rows.append({"checkpoint": fold.label, "role": role, **_distribution(vector)})
        for variant, value in (
            ("raw", projection),
            ("layernorm_scaled_approximation", projection * norm_weight[None, :]),
        ):
            _, singular, right_h = np.linalg.svd(value, full_matrices=False)
            bases[(fold.label, variant)] = right_h.T
            projections[(fold.label, variant)] = value
            vector_rows.append(
                {
                    "checkpoint": fold.label,
                    "role": f"projection_{variant}",
                    **_distribution(value.ravel()),
                    **{
                        f"spectrum_{key}": metric
                        for key, metric in _spectrum_shape(singular).items()
                    },
                }
            )
    for left, right in combinations(sorted(fold.label for fold in folds), 2):
        for variant in ("raw", "layernorm_scaled_approximation"):
            left_matrix = projections[(left, variant)]
            right_matrix = projections[(right, variant)]
            row = {
                "checkpoint_left": left,
                "checkpoint_right": right,
                "projection_variant": variant,
                "frobenius_cosine": float(
                    np.vdot(left_matrix, right_matrix)
                    / (np.linalg.norm(left_matrix) * np.linalg.norm(right_matrix))
                ),
                "relative_frobenius_distance": float(
                    np.linalg.norm(left_matrix - right_matrix)
                    / math.sqrt(np.linalg.norm(left_matrix) * np.linalg.norm(right_matrix))
                ),
            }
            for metrics in _subspace_row(
                bases[(left, variant)], bases[(right, variant)], (16, 32, 64, 128, 256)
            ):
                rows.append({**row, **metrics})
        left_fold = next(fold for fold in folds if fold.label == left)
        right_fold = next(fold for fold in folds if fold.label == right)
        for role, left_vector, right_vector in (
            ("layernorm_weight", left_fold.norm_weight, right_fold.norm_weight),
            ("layernorm_bias", left_fold.norm_bias, right_fold.norm_bias),
        ):
            x = left_vector.detach().cpu().numpy().astype(np.float64)
            y = right_vector.detach().cpu().numpy().astype(np.float64)
            rows.append(
                {
                    "checkpoint_left": left,
                    "checkpoint_right": right,
                    "projection_variant": role,
                    "rank": None,
                    "frobenius_cosine": float(
                        np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
                    )
                    if np.linalg.norm(x) and np.linalg.norm(y)
                    else None,
                    "relative_frobenius_distance": float(
                        np.linalg.norm(x - y) / math.sqrt(np.linalg.norm(x) * np.linalg.norm(y))
                    )
                    if np.linalg.norm(x) and np.linalg.norm(y)
                    else None,
                }
            )
    _write_csv(output_dir / "projection_checkpoint_pairwise.csv", rows)
    _write_csv(output_dir / "projection_checkpoint_statistics.csv", vector_rows)


def run_adjacent_raw_pass(
    checkpoint: base.SafetensorCheckpoint, output_dir: Path, *, device: Any
) -> None:
    import numpy as np

    torch = base._require_torch()
    rows: list[dict[str, Any]] = []
    for family in base.MATRIX_FAMILIES:
        previous = base.load_block_matrix(checkpoint, 0, family).detach().cpu()
        for block in range(1, base.N_BLOCKS):
            current = base.load_block_matrix(checkpoint, block, family).detach().cpu()
            left = previous.to(device=device, dtype=torch.float32)
            right = current.to(device=device, dtype=torch.float32)
            delta = right - left
            left_norm = torch.linalg.vector_norm(left)
            right_norm = torch.linalg.vector_norm(right)
            dot = torch.sum(left * right)
            row_cosines = (
                torch.nn.functional.cosine_similarity(left, right, dim=1)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            column_cosines = (
                torch.nn.functional.cosine_similarity(left, right, dim=0)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            scale = dot / torch.sum(left * left)
            rows.append(
                {
                    "from_block": block - 1,
                    "to_block": block,
                    "family": family,
                    "frobenius_distance": float(torch.linalg.vector_norm(delta).cpu()),
                    "relative_frobenius_distance_geometric": float(
                        (torch.linalg.vector_norm(delta) / torch.sqrt(left_norm * right_norm)).cpu()
                    ),
                    "flattened_cosine": float((dot / (left_norm * right_norm)).cpu()),
                    "optimal_scalar": float(scale.cpu()),
                    "scale_adjusted_relative_error": float(
                        (torch.linalg.vector_norm(right - scale * left) / right_norm).cpu()
                    ),
                    "sign_agreement_fraction": float(
                        torch.mean((torch.sign(left) == torch.sign(right)).float()).cpu()
                    ),
                    **_distribution(row_cosines, "fixed_row_cosine__"),
                    **_distribution(column_cosines, "fixed_column_cosine__"),
                }
            )
            previous = current
            del left, right, delta
            if device.type == "cuda":
                torch.cuda.empty_cache()
    _write_csv(output_dir / "adjacent_raw_matrix_changes.csv", rows)


def _identifier(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    return "|".join(
        f"{field}={row[field]}" for field in fields if field in row and row[field] is not None
    )


def collect_layer_metrics(output_dir: Path) -> list[dict[str, Any]]:
    import pandas as pd

    specifications = {
        "spectral_shape_metrics.csv": ("block", ("family", "geometry")),
        "adjacent_spectral_transitions.csv": ("to_block", ("family", "geometry")),
        "trajectory_local_geometry.csv": ("block", ("family",)),
        "attention_head_distribution_summary.csv": ("block", ()),
        "attention_head_permutation_geometry.csv": ("to_block", ()),
        "mixer_profile_summary.csv": (None, ("checkpoint",)),
        "raw_tensor_statistics.csv": ("block", ("family",)),
        "ffn_neuron_summary.csv": ("block", ()),
        "normalization_vector_statistics.csv": ("block", ("role",)),
        "normalization_adjacent_changes.csv": ("to_block", ("role",)),
        "normalization_cross_role.csv": ("block", ("role_left", "role_right")),
        "within_block_subspace_geometry.csv": (
            "block",
            ("comparison", "family_left", "family_right", "rank"),
        ),
        "adjacent_subspace_turnover.csv": ("to_block", ("family", "side", "rank")),
        "projection_alignment_randomized_svd.csv": (
            "block",
            ("checkpoint", "projection_variant", "family", "side", "rank"),
        ),
        "adjacent_raw_matrix_changes.csv": ("to_block", ("family",)),
    }
    rows: list[dict[str, Any]] = []
    excluded = {
        "block",
        "to_block",
        "from_block",
        "produced_state",
        "rank",
        "nearest_layer",
        "nearest_layer_depth_gap",
        "quantile_sample_size",
        "parameter_count",
        "target_rank",
        "sketch_rank",
        "power_iterations",
    }
    for filename, (block_field, identifiers) in specifications.items():
        path = output_dir / filename
        if not path.is_file() or block_field is None:
            continue
        frame = pd.read_csv(path)
        numeric = [
            column
            for column in frame.select_dtypes(include="number").columns
            if column not in excluded
        ]
        for record in frame.to_dict(orient="records"):
            block = int(record[block_field])
            if not 0 <= block < base.N_BLOCKS:
                continue
            family = _identifier(record, identifiers) or "all"
            for metric in numeric:
                value = record.get(metric)
                if value is None or not math.isfinite(float(value)):
                    continue
                rows.append(
                    {
                        "block": block,
                        "produced_state": block + 1,
                        "source_table": filename,
                        "family": family,
                        "metric": metric,
                        "value": float(value),
                    }
                )
    _write_csv(output_dir / "derived_layer_metrics_long.csv", rows)
    return rows


def _poly_detrend(values: Any, degree: int = 3) -> tuple[Any, Any]:
    import numpy as np

    y = np.asarray(values, dtype=np.float64)
    x = np.linspace(-1, 1, y.size)
    use_degree = min(degree, y.size - 1)
    trend = np.polyval(np.polyfit(x, y, use_degree), x)
    return trend, y - trend


def analyze_layer_metrics(output_dir: Path, baseline_dir: Path) -> None:
    import numpy as np
    import pandas as pd

    rows = collect_layer_metrics(output_dir)
    frame = pd.DataFrame(rows)
    mixer = pd.read_csv(baseline_dir / "esmfold2_mixing_weights.csv")
    profiles = {
        name: group.sort_values("state_index")["mixing_weight"].to_numpy(dtype=np.float64)[1:]
        for name, group in mixer.groupby("checkpoint")
    }
    anomaly_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    usable_series: list[tuple[str, str, str, Any]] = []
    for (source, family, metric), group in frame.groupby(
        ["source_table", "family", "metric"], sort=True
    ):
        if group["block"].nunique() != base.N_BLOCKS or len(group) != base.N_BLOCKS:
            continue
        ordered = group.sort_values("block")
        values = ordered["value"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)) or float(np.std(values)) == 0:
            continue
        trend, residual = _poly_detrend(values)
        residual_z = _robust_z(residual)
        usable_series.append((source, family, metric, residual_z))
        absolute = np.abs(residual - np.median(residual))
        for block in range(base.N_BLOCKS):
            empirical_p = float((1 + np.sum(absolute >= absolute[block])) / (base.N_BLOCKS + 1))
            local_z = _local_robust_z(residual, block)
            global_z = float(residual_z[block])
            anomaly_rows.append(
                {
                    "block": block,
                    "produced_state": block + 1,
                    "source_table": source,
                    "family": family,
                    "metric": metric,
                    "value": float(values[block]),
                    "depth_trend": float(trend[block]),
                    "depth_residual": float(residual[block]),
                    "depth_adjusted_global_robust_z": global_z,
                    "local_robust_z": local_z,
                    "global_percentile": float(
                        (np.sum(values <= values[block]) - 0.5) / base.N_BLOCKS
                    ),
                    "empirical_outlier_p": empirical_p,
                    "depth_adjusted_normal_approximation_p": math.erfc(abs(global_z) / math.sqrt(2))
                    if math.isfinite(global_z)
                    else 0.0,
                    "local_normal_approximation_p": math.erfc(abs(local_z) / math.sqrt(2))
                    if local_z is not None and math.isfinite(local_z)
                    else (0.0 if local_z is not None else None),
                    "prespecified_candidate": int(block in {50, 51}),
                }
            )
        for checkpoint, profile in profiles.items():
            for lag in range(-2, 3):
                if lag < 0:
                    x = values[-lag:]
                    y = profile[: lag or None]
                elif lag > 0:
                    x = values[:-lag]
                    y = profile[lag:]
                else:
                    x = values
                    y = profile
                _, x_residual = _poly_detrend(x)
                _, y_residual = _poly_detrend(y)
                observed = _safe_corr(x_residual, y_residual)
                shift_values = []
                if observed is not None:
                    for shift in range(1, len(y_residual)):
                        shifted = np.roll(y_residual, shift)
                        candidate = _safe_corr(x_residual, shifted)
                        if candidate is not None:
                            shift_values.append(candidate)
                p = (
                    float(
                        (1 + np.sum(np.abs(shift_values) >= abs(observed)))
                        / (len(shift_values) + 1)
                    )
                    if observed is not None and shift_values
                    else None
                )
                correlation_rows.append(
                    {
                        "source_table": source,
                        "family": family,
                        "metric": metric,
                        "checkpoint": checkpoint,
                        "lag": lag,
                        "n": len(x),
                        "pearson": _safe_corr(x, y),
                        "spearman": _safe_corr(x, y, method="spearman"),
                        "depth_detrended_pearson": observed,
                        "exact_circular_shift_p": p,
                        "unique_nonzero_shifts": len(shift_values),
                        "primary_indexing": int(lag == 0),
                    }
                )
    anomaly_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in anomaly_rows:
        anomaly_groups[(row["source_table"], row["family"], row["metric"])].append(row)
    for selected in anomaly_groups.values():
        _bh(selected, "empirical_outlier_p", "empirical_bh_q_within_series")
        _bh(selected, "depth_adjusted_normal_approximation_p", "depth_adjusted_bh_q_within_series")
        _bh(selected, "local_normal_approximation_p", "local_bh_q_within_series")
    for source in sorted({row["source_table"] for row in correlation_rows}):
        selected = [
            row
            for row in correlation_rows
            if row["source_table"] == source and row["exact_circular_shift_p"] is not None
        ]
        _bh(selected, "exact_circular_shift_p", "source_wide_bh_q")
    correlation_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in correlation_rows:
        correlation_groups[(row["source_table"], row["family"], row["metric"], row["lag"])].append(
            row
        )
    for selected in correlation_groups.values():
        _bh(selected, "exact_circular_shift_p", "checkpoint_consistency_bh_q")
    _write_csv(output_dir / "derived_layer_anomalies.csv", anomaly_rows)
    _write_csv(output_dir / "derived_mixer_correlations.csv", correlation_rows)

    source_groups: dict[str, list[Any]] = defaultdict(list)
    all_series: list[Any] = []
    for source, _, _, z in usable_series:
        clipped = np.clip(np.asarray(z, dtype=np.float64), -10, 10)
        source_groups[source].append(clipped)
        all_series.append(clipped)
    omnibus_rows: list[dict[str, Any]] = []
    for scope, series in [(source, values) for source, values in source_groups.items()] + [
        ("all_sources", all_series)
    ]:
        if not series:
            continue
        matrix = np.stack(series, axis=1)
        for block in range(base.N_BLOCKS):
            absolute_z = np.abs(matrix[block])
            omnibus_rows.append(
                {
                    "block": block,
                    "produced_state": block + 1,
                    "scope": scope,
                    "metric_count": matrix.shape[1],
                    "median_absolute_depth_adjusted_z": float(np.median(absolute_z)),
                    "mean_absolute_depth_adjusted_z": float(np.mean(absolute_z)),
                    "rms_depth_adjusted_z": float(np.sqrt(np.mean(np.square(matrix[block])))),
                    "maximum_absolute_depth_adjusted_z": float(np.max(absolute_z)),
                    "fraction_above_2": float(np.mean(absolute_z >= 2)),
                    "fraction_above_3": float(np.mean(absolute_z >= 3)),
                    "positive_fraction": float(np.mean(matrix[block] > 0)),
                }
            )
    _write_csv(output_dir / "layer_omnibus_scores.csv", omnibus_rows)

    balanced_rows: list[dict[str, Any]] = []
    source_matrices = {
        source: np.stack(values, axis=1) for source, values in source_groups.items() if values
    }
    for block in range(base.N_BLOCKS):
        source_scores = []
        for source, matrix in sorted(source_matrices.items()):
            absolute = np.abs(matrix[block])
            source_rms = float(np.sqrt(np.mean(np.square(matrix[block]))))
            source_scores.append(source_rms)
            balanced_rows.append(
                {
                    "block": block,
                    "produced_state": block + 1,
                    "scope": source,
                    "source_count": 1,
                    "source_rms_median": source_rms,
                    "source_rms_mean": source_rms,
                    "source_rms_maximum": source_rms,
                    "within_source_median_absolute_z": float(np.median(absolute)),
                }
            )
        balanced_rows.append(
            {
                "block": block,
                "produced_state": block + 1,
                "scope": "balanced_across_sources",
                "source_count": len(source_scores),
                "source_rms_median": float(np.median(source_scores)),
                "source_rms_mean": float(np.mean(source_scores)),
                "source_rms_maximum": float(np.max(source_scores)),
                "within_source_median_absolute_z": None,
            }
        )
    _write_csv(output_dir / "layer_omnibus_balanced.csv", balanced_rows)


def make_figures(output_dir: Path, baseline_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFD",
        }
    )

    spectral = pd.read_csv(output_dir / "spectral_shape_metrics.csv")
    operator = spectral[spectral["geometry"] == "operator"]
    metrics = ["energy_at_16", "spectral_gini", "powerlaw_head_slope", "log_spectrum_knee_rank"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, metrics, strict=True):
        for family, group in operator.groupby("family"):
            axis.plot(group["block"], group[metric], label=family, linewidth=1.2)
        axis.axvspan(48, 53, color="#C69C3C", alpha=0.12)
        axis.axvline(50, color="#C96B36", linewidth=1, linestyle="--")
        axis.set(title=metric.replace("_", " "), xlabel="ESMC block", ylabel="Value")
    axes.flat[0].legend(ncol=4, fontsize=7)
    fig.suptitle("Operator spectral-shape profiles across all 80 ESMC-6B blocks")
    fig.savefig(figure_dir / "spectral_shape_profiles.png", dpi=300)
    plt.close(fig)

    trajectory = pd.read_csv(output_dir / "trajectory_local_geometry.csv")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for family, group in trajectory.groupby("family"):
        axes[0].plot(
            group["block"], group["previous_frobenius_distance"], label=family, linewidth=1.2
        )
        axes[1].plot(group["block"], group["turning_angle_degrees"], label=family, linewidth=1.2)
    for axis in axes:
        axis.axvspan(48, 53, color="#C69C3C", alpha=0.12)
        axis.axvline(50, color="#C96B36", linewidth=1, linestyle="--")
    axes[0].set(ylabel="Frobenius step distance", title="Adjacent trajectory speed")
    axes[1].set(
        ylabel="Turning angle (degrees)",
        xlabel="ESMC block",
        title="Parameter-trajectory curvature",
    )
    axes[0].legend(ncol=7, fontsize=7)
    fig.savefig(figure_dir / "trajectory_speed_curvature.png", dpi=300)
    plt.close(fig)

    if (output_dir / "raw_tensor_statistics.csv").is_file():
        raw = pd.read_csv(output_dir / "raw_tensor_statistics.csv")
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        raw_metrics = [
            "standard_deviation",
            "excess_kurtosis_sampled",
            "row_norm__gini_absolute",
            "right_singular_vectors_mean_ipr",
        ]
        for axis, metric in zip(axes.flat, raw_metrics, strict=True):
            for family, group in raw.groupby("family"):
                axis.plot(group["block"], group[metric], label=family, linewidth=1.1)
            axis.axvspan(48, 53, color="#C69C3C", alpha=0.12)
            axis.axvline(50, color="#C96B36", linewidth=1, linestyle="--")
            axis.set(title=metric.replace("_", " "), xlabel="ESMC block", ylabel="Value")
        axes.flat[0].legend(ncol=4, fontsize=7)
        fig.suptitle("Raw parameter distributions and singular-vector localization")
        fig.savefig(figure_dir / "raw_parameter_profiles.png", dpi=300)
        plt.close(fig)

    if (output_dir / "layer_omnibus_scores.csv").is_file():
        omnibus = pd.read_csv(output_dir / "layer_omnibus_scores.csv")
        scope = omnibus[omnibus["scope"] == "all_sources"]
        fig, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
        axis.plot(
            scope["block"],
            scope["rms_depth_adjusted_z"],
            color="#3568A6",
            linewidth=1.5,
            label="RMS",
        )
        axis.plot(
            scope["block"],
            scope["median_absolute_depth_adjusted_z"],
            color="#C69C3C",
            linewidth=1.5,
            label="Median absolute",
        )
        axis.axvspan(48, 53, color="#C69C3C", alpha=0.12)
        axis.axvline(50, color="#C96B36", linewidth=1, linestyle="--")
        axis.set(
            title="Omnibus depth-adjusted weight anomaly scores",
            xlabel="ESMC block",
            ylabel="Robust standardized score",
        )
        axis.legend()
        fig.savefig(figure_dir / "layer_omnibus_anomalies.png", dpi=300)
        plt.close(fig)


def refresh_output_catalog(output_dir: Path) -> None:
    import numpy as np
    import pandas as pd

    artifact_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    array_rows: list[dict[str, Any]] = []
    excluded_names = {
        "output_artifact_inventory.csv",
        "output_field_dictionary.csv",
        "output_array_catalog.csv",
    }
    for path in sorted(
        p
        for p in output_dir.rglob("*")
        if p.is_file() and ".progress" not in p.parts and p.name not in excluded_names
    ):
        relative = path.relative_to(output_dir).as_posix()
        artifact_rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "kind": path.suffix.lower().lstrip("."),
            }
        )
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            for column in frame.columns:
                series = frame[column]
                nonnull = series.dropna()
                field_rows.append(
                    {
                        "artifact": relative,
                        "field": column,
                        "dtype": str(series.dtype),
                        "rows": len(series),
                        "null_count": int(series.isna().sum()),
                        "unique_nonnull": int(nonnull.nunique()),
                        "minimum": float(nonnull.min())
                        if len(nonnull) and pd.api.types.is_numeric_dtype(series)
                        else None,
                        "maximum": float(nonnull.max())
                        if len(nonnull) and pd.api.types.is_numeric_dtype(series)
                        else None,
                        "description": describe_field(column),
                    }
                )
        elif path.suffix.lower() == ".npz":
            with np.load(path) as archive:
                for name in archive.files:
                    array = archive[name]
                    finite = np.isfinite(array) if np.issubdtype(array.dtype, np.number) else None
                    array_rows.append(
                        {
                            "artifact": relative,
                            "array": name,
                            "shape": "x".join(map(str, array.shape)) or "scalar",
                            "dtype": str(array.dtype),
                            "elements": int(array.size),
                            "bytes_uncompressed": int(array.nbytes),
                            "nonfinite_count": int(array.size - finite.sum())
                            if finite is not None
                            else None,
                            "minimum": float(np.nanmin(array))
                            if finite is not None and array.size
                            else None,
                            "maximum": float(np.nanmax(array))
                            if finite is not None and array.size
                            else None,
                            "description": describe_field(name),
                        }
                    )
    _write_csv(output_dir / "output_artifact_inventory.csv", artifact_rows)
    _write_csv(output_dir / "output_field_dictionary.csv", field_rows)
    _write_csv(output_dir / "output_array_catalog.csv", array_rows)


def validate_outputs(output_dir: Path, baseline_dir: Path) -> None:
    import numpy as np
    import pandas as pd

    checks: list[dict[str, Any]] = []

    def add(
        scope: str,
        check: str,
        passed: bool,
        value: Any,
        expected: Any,
        severity: str,
        detail: str = "",
    ) -> None:
        checks.append(
            {
                "scope": scope,
                "check": check,
                "status": "pass" if passed else "fail",
                "severity": severity if not passed else "none",
                "value": value,
                "expected": expected,
                "detail": detail,
            }
        )

    expected_rows = {
        "spectral_shape_metrics.csv": 80 * 7 * 3,
        "adjacent_spectral_transitions.csv": 79 * 7 * 3,
        "trajectory_local_geometry.csv": 80 * 7,
        "attention_head_distribution_summary.csv": 80,
        "attention_head_permutation_geometry.csv": 79,
        "mixer_profile_summary.csv": 4,
        "mixer_profile_pairwise.csv": 6,
        "compression_pareto.csv": 3920,
        "raw_tensor_statistics.csv": 80 * 7,
        "ffn_neuron_summary.csv": 80,
        "randomized_svd_sensitivity.csv": 4 * 7 * 2 * 3,
        "normalization_vector_statistics.csv": 80 * len(VECTOR_ROLES),
        "normalization_channel_depth_metrics.csv": len(VECTOR_ROLES) * base.D_MODEL,
        "normalization_adjacent_changes.csv": len(VECTOR_ROLES) * 79,
        "normalization_cross_role.csv": math.comb(len(VECTOR_ROLES), 2) * 80,
        "adjacent_raw_matrix_changes.csv": 79 * 7,
    }
    for filename, expected in expected_rows.items():
        path = output_dir / filename
        if not path.is_file():
            checks.append(
                {
                    "scope": filename,
                    "check": "presence",
                    "status": "not_run",
                    "severity": "none",
                    "value": None,
                    "expected": expected,
                    "detail": "Stage not selected or not yet completed",
                }
            )
            continue
        frame = pd.read_csv(path)
        add(filename, "row_count", len(frame) == expected, len(frame), expected, "high")
        numeric = frame.select_dtypes(include="number")
        infinite = int(np.isinf(numeric.to_numpy(dtype=np.float64)).sum())
        add(
            filename,
            "no_infinite_values",
            infinite == 0,
            infinite,
            0,
            "high",
            "Null values are assessed separately because some fields are structurally inapplicable",
        )
        add(
            filename,
            "no_exact_duplicate_rows",
            not frame.duplicated().any(),
            int(frame.duplicated().sum()),
            0,
            "medium",
        )

    mixer = pd.read_csv(baseline_dir / "esmfold2_mixing_weights.csv")
    for checkpoint, group in mixer.groupby("checkpoint"):
        total = float(group["mixing_weight"].sum())
        add(
            "esmfold2_mixing_weights.csv",
            f"{checkpoint}_softmax_sum",
            abs(total - 1) < 1e-12,
            total,
            1.0,
            "critical",
        )
        states = sorted(group["state_index"].astype(int).tolist())
        add(
            "esmfold2_mixing_weights.csv",
            f"{checkpoint}_state_coverage",
            states == list(range(81)),
            len(states),
            81,
            "critical",
        )

    monotonic_failures = 0
    negative_failures = 0
    for path in sorted((baseline_dir / "spectra").glob("*.npz")):
        with np.load(path) as archive:
            for name in (
                "operator_singular_values",
                "rows_singular_values",
                "columns_singular_values",
            ):
                values = archive[name]
                monotonic_failures += int(np.any(np.diff(values) > 1e-10))
                negative_failures += int(np.any(values < -1e-12))
    add(
        "baseline/spectra",
        "monotone_nonincreasing",
        monotonic_failures == 0,
        monotonic_failures,
        0,
        "critical",
    )
    add("baseline/spectra", "nonnegative", negative_failures == 0, negative_failures, 0, "critical")

    trajectory_failures = 0
    for path in sorted((baseline_dir / "trajectory").glob("*.npz")):
        with np.load(path) as archive:
            for name in ("gram", "distances", "cosine"):
                matrix = archive[name]
                if not np.allclose(matrix, matrix.T, atol=1e-8, rtol=1e-8):
                    trajectory_failures += 1
            if not np.allclose(np.diag(archive["distances"]), 0, atol=1e-8):
                trajectory_failures += 1
            if not np.allclose(np.diag(archive["cosine"]), 1, atol=1e-8):
                trajectory_failures += 1
    add(
        "baseline/trajectory",
        "symmetric_with_valid_diagonals",
        trajectory_failures == 0,
        trajectory_failures,
        0,
        "critical",
    )

    spectral = pd.read_csv(output_dir / "spectral_shape_metrics.csv")
    baseline_tensor = pd.read_csv(baseline_dir / "tensor_metrics.csv")
    merged = spectral[spectral["geometry"] == "operator"].merge(
        baseline_tensor[["block", "family", "operator_leading_energy_fraction"]],
        on=["block", "family"],
        validate="one_to_one",
    )
    maximum_difference = float(
        np.max(np.abs(merged["energy_at_1"] - merged["operator_leading_energy_fraction"]))
    )
    add(
        "spectral_shape_metrics.csv",
        "energy_at_1_reconciles_to_baseline",
        maximum_difference < 1e-12,
        maximum_difference,
        "<1e-12",
        "critical",
    )

    basis_files = (
        sorted((output_dir / "bases").glob("*.npz")) if (output_dir / "bases").is_dir() else []
    )
    if basis_files:
        add("bases", "file_count", len(basis_files) == 560, len(basis_files), 560, "critical")
        maximum_orthogonality = 0.0
        for path in basis_files:
            with np.load(path) as archive:
                for name in ("left", "right"):
                    basis_array = archive[name].astype(np.float64)
                    error = float(
                        np.linalg.norm(basis_array.T @ basis_array - np.eye(basis_array.shape[1]))
                    )
                    maximum_orthogonality = max(maximum_orthogonality, error)
        add(
            "bases",
            "maximum_saved_orthogonality_error",
            maximum_orthogonality < 1e-3,
            maximum_orthogonality,
            "<1e-3",
            "high",
        )
    ffn_files = (
        sorted((output_dir / "ffn_neurons").glob("*.npz"))
        if (output_dir / "ffn_neurons").is_dir()
        else []
    )
    if ffn_files:
        add("ffn_neurons", "file_count", len(ffn_files) == 80, len(ffn_files), 80, "critical")
        shape_failures = 0
        for path in ffn_files:
            with np.load(path) as archive:
                shape_failures += sum(
                    int(archive[name].shape != (base.FFN_WIDTH,)) for name in archive.files
                )
        add("ffn_neurons", "array_shapes", shape_failures == 0, shape_failures, 0, "critical")
    _write_csv(output_dir / "output_data_quality_checks.csv", checks)


def write_evidence_index(
    output_dir: Path, baseline_dir: Path, stages: Sequence[str], checkpoint: Path | None
) -> None:
    import pandas as pd

    inventory = pd.read_csv(output_dir / "output_artifact_inventory.csv")
    csv_rows = []
    for record in inventory[inventory["kind"] == "csv"].to_dict(orient="records"):
        path = output_dir / record["path"]
        frame = pd.read_csv(path)
        csv_rows.append(
            f"| `{record['path']}` | {len(frame):,} | {len(frame.columns):,} | `{record['sha256']}` |"
        )
    text = f"""# ESMC-6B weights-only evidence archive

This is a pre-synthesis evidence index. It documents the full analysis output without converting correlations or weight geometry into biological or causal claims.

## Scope

- ESMC blocks: 0 through 79; block 50 produces state 51 and block 51 consumes it.
- Inputs or activations: none.
- Baseline evidence directory: `{baseline_dir}`.
- Raw checkpoint: `{checkpoint if checkpoint is not None else "not used in selected stages"}`.
- Completed stages: {", ".join(stages)}.
- Scalar tables: CSV. Large vectors, bases, spectra, pairwise matrices, and neuron-level arrays: compressed NPZ.
- Numeric conventions: zero-based blocks, one-based produced states, FP64 accumulation where stated, deterministic randomized SVD for new singular vectors.

## CSV tables

| File | Rows | Columns | SHA-256 |
| --- | ---: | ---: | --- |
{os.linesep.join(csv_rows)}

## Large arrays

See `output_array_catalog.csv` for every NPZ member, shape, dtype, element count, non-finite count, extrema, definition, and containing artifact. The original checkpoint is not duplicated into the result directory.

## Field definitions

See `output_field_dictionary.csv` for every CSV field and its dtype, row count, null count, distinct count, numeric range, and definition. See `field_dictionary.csv` and `array_catalog.csv` for the frozen baseline bundle.

## Interpretation boundary

These files support weight-space hypotheses only. Approximate top singular vectors are explicitly labeled by their randomized-SVD diagnostics. Four ESMFold2 checkpoints are highly related checkpoints and therefore consistency checks, not four independent experiments. No result here establishes effects on P@L, folding accuracy, perplexity, or biological information retention.
"""
    (output_dir / "ALL_DATA.md").write_text(text, encoding="utf-8")


def write_methodology_docs(output_dir: Path) -> None:
    definitions = """# Metric definitions and data conventions

## Indexing and scope

- `block` is zero-based from 0 to 79.
- `produced_state = block + 1`. ESMC state 51 is produced by block 50 and consumed by block 51.
- State 0 is the embedding state and has no producing transformer block.
- No sequences, activations, hidden states, model outputs, or forward passes are used.
- Matrix families are Q, K, V, attention output O, SwiGLU gate, SwiGLU value, and FFN down projection.

## Linear spectra

- Singular energy is `s_i^2 / sum_j s_j^2`.
- Stable rank is `||W||_F^2 / ||W||_2^2`.
- Participation ratio is `(sum_i s_i^2)^2 / sum_i s_i^4`.
- Spectral effective rank is `exp(-sum_i p_i log p_i)` for energy probabilities `p_i`.
- Spectral flatness is the geometric mean divided by the arithmetic mean of positive singular values.
- Spectral and energy Gini coefficients quantify concentration on `[0, 1]`.
- HHI is the sum of squared normalized mass and effective count is the exponential entropy.
- Energy-at-rank and tail-energy fields are exact functions of saved complete spectra.
- Gap ratio at rank `k` is `s_k / s_(k+1)`.
- Power-law slopes fit log singular value against log rank; exponential slopes fit log singular value against rank. Head and bulk windows are recorded in the code.
- The log-spectrum knee is the maximum perpendicular deviation from the line connecting the first and final positive log singular values. It is a descriptive knee, not a model-selection criterion.
- Adjacent spectral Jensen-Shannon, total variation, rank-Wasserstein, correlation, and cosine compare normalized full spectral profiles.

## Raw parameter distributions

- Mean, variance, RMS, and exact zero fraction scan the complete tensor with FP64 NumPy accumulation.
- Quantiles, skewness, kurtosis, sign fraction, and relative near-zero rates use a deterministic evenly spaced sample of at most 1,000,000 parameters. Fields are suffixed `sampled`.
- Row and column norm distributions are complete, not sampled.
- Coherence uses 256 deterministic evenly spaced row or column vectors, unit normalizes them, and summarizes off-diagonal cosine magnitudes.

## Randomized singular vectors

- New singular-vector bases use deterministic Gaussian randomized SVD at target rank 64 by default, 16 oversamples, and two power iterations.
- Every tensor records captured Frobenius energy and left/right orthogonality error.
- Basis arrays are saved in `bases/*.npz` as float32. Subspace metrics are accumulated in float64.
- Singular-vector inverse participation ratio is `sum_i v_i^4`; its reciprocal is effective coordinate support.
- Chordal distance is `sqrt(r - sum_i cos(theta_i)^2)`.
- Normalized overlap is the mean squared canonical correlation between equal-rank subspaces.

## Layer trajectories

- Baseline trajectory Gram, cosine, and Frobenius-distance matrices use all parameters of each matrix family.
- Kernel-PCA coordinates come from the positive eigenspectrum of the centered 80 by 80 layer Gram matrix.
- Speed is adjacent Frobenius distance, acceleration is the norm of the second coordinate difference, and turning angle is the angle between consecutive trajectory steps.
- Nearest layer excludes self. Depth gap records whether geometric proximity is local or nonlocal in depth.

## Attention heads

- Q, K, and V use 64 by 2560 per-head matrices. O uses the matching 2560 by 64 block.
- Q-K and V-O overlaps are canonical subspace overlaps. Fold projection overlaps compare the head output space with the fold projection row space.
- Head distribution rows preserve mean, spread, quantiles, skewness, kurtosis, Gini, entropy, and top-head concentration for every baseline head metric.
- Hungarian permutations are decomposed into fixed points, cycles, displacements, inversions, and gain over fixed-index matching.

## FFN neurons and normalization channels

- FFN neuron arrays preserve gate norm, value norm, matching down-column norm, three pairwise cosines, and the descriptive triple-strength proxy `||g_i|| ||v_i|| ||d_i||` for all 6,912 neurons in every block.
- Triple strength is a weight-scale proxy only. It is not an activation, attribution, or functional importance score.
- Normalization vectors are preserved in `normalization_vectors.npz`; channel-depth rows summarize all 2,560 channels for each of six vector roles.
- Cross-role correlations compare matching channel indices. Adjacent changes report cosine, relative L2 movement, maximum channel movement, and thresholded change fractions.

## Compression

- Effective storage uses reported quantization storage when available and ideal nonzero-value storage for sparsity methods. Index and metadata overhead for sparse formats is not modeled.
- Pareto optimality means no earlier configuration at equal or lower effective storage has lower Frobenius error within the same block and family.
- All compression metrics are parameter reconstruction metrics, not functional performance measurements.

## Anomalies, correlations, and multiple testing

- Depth trends are cubic least-squares fits over normalized block depth. Residuals are standardized by median absolute deviation.
- Local robust z-scores compare a block against the five blocks on each side, excluding itself and truncating at endpoints.
- Empirical outlier p-values rank absolute residual deviations among 80 blocks. Normal-approximation p-values from robust z-scores are labeled as approximations.
- Benjamini-Hochberg q-values are reported within each complete 80-layer metric series for layer anomalies.
- Mixer associations include Pearson, Spearman, cubic depth-detrended Pearson, lags -2 through +2, and exact circular shifts over every unique nonzero shift.
- Exact shift p-values have minimum resolution 1/80 = 0.0125. Checkpoint-consistency q-values adjust across the four related ESMFold2 profiles for one metric and lag.
- The four folds are consistency checks, not independent statistical replicates.

## Null and non-finite semantics

- Null means structurally inapplicable, unavailable because a stage was not selected, or undefined because a denominator is zero. The field dictionary gives counts by column.
- Infinity is never accepted in shipped numerical tables. Validation checks enforce this.
- Tiny negative Gram eigenvalues are numerical roundoff only when they lie below the recorded tolerance; the baseline preserves clamped negative mass.
"""
    coverage = """# Analysis coverage ledger

| Lens | Status | Preserved outputs | Evidence boundary |
| --- | --- | --- | --- |
| Exact uncentered operator spectra | complete | baseline spectra NPZ, tensor metrics | Full singular values, no vectors |
| Centered row and column PCA | complete | baseline spectra NPZ, tensor metrics | Weight-neuron geometry only |
| Spectral concentration, gaps, tails, slopes, knees | complete | spectral shape CSV | Descriptive shape diagnostics |
| Low-rank reconstruction | complete | tensor metrics | Parameter error only |
| INT8, INT4, unstructured and 2:4 reconstruction | complete | compression and Pareto CSV | No performance claim |
| Raw value distributions | complete after raw stage | raw tensor statistics | Sampled higher moments are labeled |
| Row and column norm concentration | complete after raw stage | raw tensor statistics | Full vector norms |
| Row and column coherence | complete after raw stage | raw tensor statistics | Deterministic 256-vector diagnostic |
| Singular-vector localization | complete after raw stage | bases NPZ, raw statistics | Randomized rank-64 approximation |
| Weight-vector intrinsic dimension | complete in baseline | intrinsic-dimension CSV | Raw and unit-normalized geometries |
| Full-matrix layer trajectory | complete | trajectory NPZ and local geometry | 80-point parameter path |
| Adjacent raw matrix change | complete after raw stage | adjacent raw changes | Fixed indices plus scale adjustment |
| Adjacent singular-subspace turnover | complete after subspace stage | adjacent subspace CSV | Randomized top subspaces |
| Within-block residual read/write circuits | complete after subspace stage | within-block subspace CSV | Linear weight-space alignment |
| Attention-head spectra and coupling | complete | baseline head CSV, distributions | All 40 heads by 80 blocks |
| Head permutation and reorganization | complete | transition and permutation CSV | Hungarian QKVO matching |
| FFN neuron norms, cosines, strength concentration | complete after raw stage | FFN NPZ and summary CSV | All 552,960 neuron-block pairs |
| Normalization vector distributions | complete | baseline and extended normalization CSV | All blocks and final norm |
| Normalization channel depth persistence | complete after raw stage | normalization NPZ and channel CSV | All 15,360 role-channel trajectories |
| ESMFold2 mixer concentration and distances | complete | mixer summary and pairwise CSV | Four related checkpoints |
| ESMFold2 projection pair geometry | complete after subspace stage | projection pairwise CSV | Raw and LN-scaled approximation |
| Projection-to-block/head alignment | complete | baseline and randomized alignment CSV | Shared projection per fold |
| Local and depth-adjusted anomaly tests | complete | anomaly CSVs | Multiple-testing scopes explicit |
| Mixer correlations and lag tests | complete | correlation CSVs | Exact circular-shift null |
| Cross-source omnibus scores | complete | omnibus CSVs | Both metric-weighted and source-balanced |
| Inputs, activations, hidden states, outputs | intentionally excluded | none | Outside weights-only scope |
| Functional ablation, P@L, folding accuracy | intentionally excluded | none | Requires model execution |
| SAE activation analysis | intentionally excluded | none | Requires activations |
"""
    (output_dir / "METRIC_DEFINITIONS.md").write_text(definitions, encoding="utf-8")
    (output_dir / "ANALYSIS_COVERAGE.md").write_text(coverage, encoding="utf-8")


def build_notebook(output_dir: Path) -> Path:
    import asyncio

    import nbformat as nbf
    from nbclient import NotebookClient

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# ESMC-6B weights-only deep dive\n\n## tl;dr\n\nThis notebook is an executable audit companion. It inventories the saved evidence and reproduces focused block 50/51 comparisons without loading model inputs or running a forward pass."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\nAll calculations consume checkpoint weights or saved weight-derived tables. Block 50 produces state 51; block 51 consumes it. Correlations are descriptive, and related ESMFold2 checkpoints are not independent replicates.\n\n### Key Assumptions\n\nThe baseline artifact bundle is frozen input. Randomized singular-vector calculations use deterministic seeds and preserve approximation diagnostics."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nROOT = Path.cwd()\nprint(ROOT)"
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. Inventory all scalar and array artifacts"),
        nbf.v4.new_code_cell(
            "artifacts = pd.read_csv(ROOT / 'output_artifact_inventory.csv')\nfields = pd.read_csv(ROOT / 'output_field_dictionary.csv')\narrays = pd.read_csv(ROOT / 'output_array_catalog.csv')\ndisplay(artifacts.groupby('kind').agg(files=('path','count'), bytes=('bytes','sum')))\ndisplay(pd.DataFrame({'csv_fields':[len(fields)], 'npz_arrays':[len(arrays)], 'npz_elements':[arrays['elements'].sum()]}))"
        ),
        nbf.v4.new_markdown_cell(
            "## Results\n\n### 2. Inspect prespecified block 50 and block 51 anomalies"
        ),
        nbf.v4.new_code_cell(
            "anomalies = pd.read_csv(ROOT / 'derived_layer_anomalies.csv')\nfocus = anomalies[(anomalies.block.isin([50,51])) & ((anomalies.local_bh_q_within_series <= 0.05) | (anomalies.depth_adjusted_bh_q_within_series <= 0.05))].sort_values(['block','local_bh_q_within_series'])\ndisplay(focus.head(100))\nprint('Significant prespecified rows:', len(focus))"
        ),
        nbf.v4.new_markdown_cell("### 3. Check mixer relationships under exact circular shifts"),
        nbf.v4.new_code_cell(
            "correlations = pd.read_csv(ROOT / 'derived_mixer_correlations.csv')\nprimary = correlations[(correlations.lag == 0) & (correlations.checkpoint_consistency_bh_q <= 0.05)].sort_values('checkpoint_consistency_bh_q')\ndisplay(primary.head(100))\nprint('Significant primary-index correlations:', len(primary))"
        ),
        nbf.v4.new_markdown_cell("### 4. Compare omnibus layer scores"),
        nbf.v4.new_code_cell(
            "omnibus = pd.read_csv(ROOT / 'layer_omnibus_scores.csv')\noverall = omnibus[omnibus.scope == 'all_sources'].copy()\ndisplay(overall.sort_values('rms_depth_adjusted_z', ascending=False).head(15))\nax = overall.plot(x='block', y=['rms_depth_adjusted_z','median_absolute_depth_adjusted_z'], figsize=(12,4), color=['#3568A6','#C69C3C'])\nax.axvspan(48,53,color='#C69C3C',alpha=.12); ax.axvline(50,color='#C96B36',ls='--'); ax.set_ylabel('Robust standardized score'); plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\nThis notebook intentionally defers the final synthesis. Use the executed tables above together with `ALL_DATA.md`, the field dictionary, array catalog, validation ledger, and 300 dpi figures. Any eventual report should separate robust scale-invariant findings from norm-driven or approximation-sensitive diagnostics."
        ),
    ]
    path = output_dir / "weights_only_deep_dive.ipynb"
    nbf.write(notebook, path)
    previous_path = os.environ.get("PATH", "")
    previous_ipython_dir = os.environ.get("IPYTHONDIR")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + previous_path
    os.environ["IPYTHONDIR"] = str(output_dir / ".ipython")
    try:
        client = NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(output_dir)}},
        )
        executed = client.execute()
    finally:
        os.environ["PATH"] = previous_path
        if previous_ipython_dir is None:
            os.environ.pop("IPYTHONDIR", None)
        else:
            os.environ["IPYTHONDIR"] = previous_ipython_dir
    nbf.write(executed, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--esmc-checkpoint", type=Path)
    parser.add_argument(
        "--esmfold2-checkpoint",
        action="append",
        default=[],
        type=base.parse_fold_argument,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-rank", type=int, default=TOP_RANK_DEFAULT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stage",
        action="append",
        choices=(
            "catalog",
            "derived",
            "raw",
            "normalization",
            "subspaces",
            "statistics",
            "figures",
            "notebook",
            "all",
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    baseline_dir = arguments.baseline_dir.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = arguments.stage or ["all"]
    stages = (
        [
            "catalog",
            "derived",
            "raw",
            "normalization",
            "subspaces",
            "statistics",
            "figures",
            "notebook",
        ]
        if "all" in requested
        else list(dict.fromkeys(requested))
    )
    provenance_path = output_dir / "deep_dive_provenance.json"
    previous_payload: dict[str, Any] = {}
    previous_stages: list[str] = []
    if provenance_path.is_file():
        previous_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        previous_stages = [str(stage) for stage in previous_payload.get("stages", [])]
    stage_evidence = {
        "catalog": output_dir / "artifact_inventory.csv",
        "derived": output_dir / "spectral_shape_metrics.csv",
        "raw": output_dir / "raw_tensor_statistics.csv",
        "normalization": output_dir / "normalization_vectors.npz",
        "subspaces": output_dir / "within_block_subspace_geometry.csv",
        "statistics": output_dir / "derived_layer_anomalies.csv",
        "figures": output_dir / "figures" / "spectral_shape_profiles.png",
        "notebook": output_dir / "weights_only_deep_dive.ipynb",
    }
    evidenced_stages = {stage for stage, path in stage_evidence.items() if path.is_file()}
    completed_stage_union = list(
        dict.fromkeys(
            stage
            for stage in (
                "catalog",
                "derived",
                "raw",
                "normalization",
                "subspaces",
                "statistics",
                "figures",
                "notebook",
            )
            if stage in set(previous_stages).union(stages).union(evidenced_stages)
        )
    )
    device = base._device(arguments.device)
    checkpoint = None
    verified_checkpoint_hashes: dict[str, str] = dict(
        previous_payload.get("verified_checkpoint_sha256", {})
    )
    resolved_checkpoint = (
        arguments.esmc_checkpoint.expanduser().resolve()
        if arguments.esmc_checkpoint is not None
        else None
    )
    if resolved_checkpoint is not None:
        verified_checkpoint_hashes = base.verify_esmc_files(resolved_checkpoint)
    if any(stage in stages for stage in ("raw", "normalization", "subspaces")):
        if resolved_checkpoint is None:
            raise DeepDiveError("Raw, normalization, and subspace stages require --esmc-checkpoint")
        checkpoint = base.SafetensorCheckpoint(resolved_checkpoint)
        base.validate_esmc_inventory(checkpoint)
    folds = [base.load_fold_weights(label, path) for label, path in arguments.esmfold2_checkpoint]
    if any(stage in stages for stage in ("subspaces",)) and len(folds) != 4:
        raise DeepDiveError("Subspace stage requires exactly four ESMFold2 checkpoint subsets")
    torch = base._require_torch()
    np = base._require_numpy()
    checkpoint_provenance = (
        str(resolved_checkpoint)
        if resolved_checkpoint is not None
        else previous_payload.get("checkpoint")
    )
    fold_provenance = (
        {fold.label: str(fold.path) for fold in folds}
        if folds
        else dict(previous_payload.get("folds", {}))
    )
    fold_hashes = (
        {
            fold.label: _sha256(
                fold.path if fold.path.is_file() else fold.path / "model.safetensors"
            )
            for fold in folds
        }
        if folds
        else dict(previous_payload.get("fold_subset_sha256", {}))
    )
    run_history = list(previous_payload.get("run_history", []))
    if not run_history and previous_payload.get("command_line"):
        run_history.append(
            {
                "command_line": previous_payload["command_line"],
                "device": previous_payload.get("device"),
                "stages": previous_payload.get("stages", []),
            }
        )
    run_history.append({"command_line": sys.argv, "device": str(device), "stages": stages})
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "baseline_dir": str(baseline_dir),
        "output_dir": str(output_dir),
        "checkpoint": checkpoint_provenance,
        "folds": fold_provenance,
        "stages": completed_stage_union,
        "device": str(device),
        "top_rank": arguments.top_rank,
        "verified_checkpoint_sha256": verified_checkpoint_hashes,
        "fold_subset_sha256": fold_hashes,
        "command_line": sys.argv,
        "run_history": run_history,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "pid": os.getpid(),
    }
    _atomic_json(provenance_path, provenance)
    if "catalog" in stages:
        catalog_baseline(baseline_dir, output_dir)
    if "derived" in stages:
        derive_spectral_shapes(baseline_dir, output_dir)
        derive_trajectory_metrics(baseline_dir, output_dir)
        derive_head_metrics(baseline_dir, output_dir)
        derive_mixer_and_compression(baseline_dir, output_dir)
    if "raw" in stages:
        assert checkpoint is not None
        run_raw_matrix_pass(
            checkpoint,
            output_dir,
            device=device,
            top_rank=arguments.top_rank,
            resume=arguments.resume,
        )
        run_adjacent_raw_pass(checkpoint, output_dir, device=device)
    if "normalization" in stages:
        assert checkpoint is not None
        run_normalization_pass(checkpoint, output_dir)
    if "subspaces" in stages:
        run_subspace_pass(output_dir, folds, top_rank=arguments.top_rank)
        run_projection_checkpoint_pass(folds, output_dir)
    if "statistics" in stages:
        analyze_layer_metrics(output_dir, baseline_dir)
    if "figures" in stages:
        make_figures(output_dir, baseline_dir)
    validate_outputs(output_dir, baseline_dir)
    write_methodology_docs(output_dir)
    refresh_output_catalog(output_dir)
    write_evidence_index(
        output_dir,
        baseline_dir,
        completed_stage_union,
        Path(checkpoint_provenance) if checkpoint_provenance else None,
    )
    refresh_output_catalog(output_dir)
    if "notebook" in stages:
        build_notebook(output_dir)
        validate_outputs(output_dir, baseline_dir)
        write_methodology_docs(output_dir)
        refresh_output_catalog(output_dir)
        write_evidence_index(
            output_dir,
            baseline_dir,
            completed_stage_union,
            Path(checkpoint_provenance) if checkpoint_provenance else None,
        )
        refresh_output_catalog(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
