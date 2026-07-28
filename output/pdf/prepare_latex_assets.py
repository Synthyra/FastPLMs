#!/usr/bin/env python3
"""Prepare deterministic LaTeX plot inputs from the audited report tables."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "artifacts" / "analysis" / "esmc_layer51_synthesis_20260726"
TABLES = REPORT / "report_tables"
DEEP_DIVE = (
    REPO
    / "artifacts"
    / "analysis"
    / "esmc_weight_geometry_deep_dive_workstation_20260725"
    / "figures"
)
OUT = Path(__file__).resolve().parent / "figures"

# P@L marker values read from the vector paths embedded in panel D of Figure 1
# in the ESMC paper. The 80 markers correspond to transformer blocks 0 through
# 79. The axis mapping uses the plotted range 0.00 through 0.75. These values
# are figure-derived approximations because the underlying source-data table was
# not distributed with the checkpoint analysis.
PAL_FIGURE_1D = (
    0.04513409, 0.06909114, 0.08623446, 0.07313395, 0.06855513,
    0.05716713, 0.06923650, 0.08091068, 0.05872520, 0.05530017,
    0.06279526, 0.07377898, 0.06522095, 0.06289974, 0.06525729,
    0.06052402, 0.05830275, 0.05672651, 0.06082382, 0.06382186,
    0.06600225, 0.06581147, 0.04765971, 0.05075769, 0.05340595,
    0.04284468, 0.04666037, 0.05269278, 0.05358311, 0.05198416,
    0.04999909, 0.05311978, 0.04494331, 0.04293553, 0.04609710,
    0.04736899, 0.05088488, 0.04848645, 0.04950396, 0.04698288,
    0.04691475, 0.05113471, 0.05475507, 0.05116651, 0.05352860,
    0.05514572, 0.05825278, 0.06135529, 0.07430591, 0.08340904,
    0.09223054, 0.09969838, 0.16047678, 0.05631768, 0.08679773,
    0.08050185, 0.30980631, 0.16398357, 0.08777891, 0.14227051,
    0.17562595, 0.09716367, 0.06823715, 0.16345665, 0.18769533,
    0.32673159, 0.19883349, 0.34964387, 0.21769842, 0.33516244,
    0.29348063, 0.36944909, 0.40359946, 0.51343666, 0.56407624,
    0.59736809, 0.64068065, 0.52933534, 0.67235082, 0.70106385,
)


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_dat(name: str, columns: list[str], rows: list[dict[str, object]]) -> None:
    path = OUT / name
    with path.open("w", newline="\n", encoding="ascii") as handle:
        handle.write(" ".join(columns) + "\n")
        for row in rows:
            handle.write(" ".join(str(row[column]) for column in columns) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    mixer = read_csv("mixer_profile.csv")
    checkpoint_files = {
        "esmfold2": "mixer_esmfold2.dat",
        "esmfold2_experimental_cutoff2025": "mixer_experimental.dat",
        "esmfold2_experimental_fast_cutoff2025": "mixer_experimental_fast.dat",
        "esmfold2_fast": "mixer_fast.dat",
    }
    for checkpoint, filename in checkpoint_files.items():
        rows = [
            {
                "state": int(row["state_index"]),
                "weight": f"{float(row['mixing_weight_pct']):.10f}",
            }
            for row in mixer
            if row["checkpoint"] == checkpoint
        ]
        write_dat(filename, ["state", "weight"], rows)

    weights_by_checkpoint = {
        checkpoint: {
            int(row["state_index"]): float(row["mixing_weight_pct"])
            for row in mixer
            if row["checkpoint"] == checkpoint
        }
        for checkpoint in checkpoint_files
    }
    comparison_rows = []
    for block, pal in enumerate(PAL_FIGURE_1D):
        state = block + 1
        checkpoint_weights = {
            checkpoint: weights[state]
            for checkpoint, weights in weights_by_checkpoint.items()
        }
        comparison_rows.append(
            {
                "block": block,
                "state": state,
                "pal": f"{pal:.8f}",
                "pal_pct": f"{100.0 * pal:.6f}",
                "esmfold2": f"{checkpoint_weights['esmfold2']:.10f}",
                "experimental": f"{checkpoint_weights['esmfold2_experimental_cutoff2025']:.10f}",
                "experimental_fast": f"{checkpoint_weights['esmfold2_experimental_fast_cutoff2025']:.10f}",
                "fast": f"{checkpoint_weights['esmfold2_fast']:.10f}",
                "mean": f"{sum(checkpoint_weights.values()) / 4.0:.10f}",
            }
        )
    write_dat(
        "pal_mixer_comparison.dat",
        [
            "block",
            "state",
            "pal",
            "pal_pct",
            "esmfold2",
            "experimental",
            "experimental_fast",
            "fast",
            "mean",
        ],
        comparison_rows,
    )

    projection = read_csv("projection_gate_value_rank16_profile.csv")
    for family in ("gate", "value"):
        rows = [
            {
                "block": int(row["block"]),
                "mean": f"{float(row['mean_overlap']):.10f}",
                "minimum": f"{float(row['minimum_overlap']):.10f}",
                "maximum": f"{float(row['maximum_overlap']):.10f}",
            }
            for row in projection
            if row["family"] == family
        ]
        write_dat(f"projection_{family}.dat", ["block", "mean", "minimum", "maximum"], rows)

    attention = read_csv("attention_head_threshold_profile.csv")
    threshold_files = {
        "V-O overlap > 0.5": "attention_gt_05.dat",
        "V-O overlap > 0.7": "attention_gt_07.dat",
    }
    for threshold, filename in threshold_files.items():
        rows = [
            {"block": int(row["block"]), "count": int(row["head_count"])}
            for row in attention
            if row["threshold"] == threshold
        ]
        write_dat(filename, ["block", "count"], rows)

    q_rank = read_csv("spectral_q_effective_rank_profile.csv")
    write_dat(
        "q_effective_rank.dat",
        ["block", "effective_rank", "depth_residual", "local_robust_z"],
        [
            {
                "block": int(row["block"]),
                "effective_rank": f"{float(row['effective_rank']):.10f}",
                "depth_residual": f"{float(row['depth_residual']):.10f}",
                "local_robust_z": f"{float(row['local_robust_z']):.10f}",
            }
            for row in q_rank
        ],
    )

    copied = {
        "layer_omnibus_anomalies.png": "layer_omnibus_anomalies.png",
        "raw_parameter_profiles.png": "raw_parameter_profiles.png",
        "spectral_shape_profiles.png": "spectral_shape_profiles.png",
        "trajectory_speed_curvature.png": "trajectory_speed_curvature.png",
    }
    for source_name, target_name in copied.items():
        shutil.copy2(DEEP_DIVE / source_name, OUT / target_name)

    manifest = {
        "schema_version": 1,
        "source_report": str(REPORT),
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(OUT.iterdir())
            if path.is_file() and path.name != "asset_manifest.json"
        ],
    }
    (OUT / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUT), "files": len(manifest["files"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
