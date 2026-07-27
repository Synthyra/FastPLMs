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
