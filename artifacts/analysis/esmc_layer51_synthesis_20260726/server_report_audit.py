#!/usr/bin/env python3
"""Build the bounded evidence payload for the ESMC state-51 synthesis report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.replace([np.inf, -np.inf], np.nan)
    return json.loads(clean.to_json(orient="records"))


def _local_robust_z(values: pd.Series, block: int, radius: int = 5) -> float | None:
    indexed = values.sort_index()
    if block not in indexed.index:
        return None
    neighbors = indexed.loc[
        (indexed.index >= block - radius) & (indexed.index <= block + radius)
    ].drop(index=block, errors="ignore").dropna()
    if len(neighbors) < 3:
        return None
    median = float(neighbors.median())
    mad = float((neighbors - median).abs().median())
    value = float(indexed.loc[block])
    if mad == 0:
        return 0.0 if value == median else math.copysign(math.inf, value - median)
    return 0.6744897501960817 * (value - median) / mad


def _rank_desc(values: pd.Series, block: int) -> int:
    return int(values.rank(method="min", ascending=False).loc[block])


def _rank_asc(values: pd.Series, block: int) -> int:
    return int(values.rank(method="min", ascending=True).loc[block])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--deep-dive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline_dir.resolve()
    deep = args.deep_dive_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def read(root: Path, name: str) -> pd.DataFrame:
        return pd.read_csv(root / name)

    mixing = read(baseline, "esmfold2_mixing_weights.csv")
    mixer_summary = read(deep, "mixer_profile_summary.csv")
    mixer_pairwise = read(deep, "mixer_profile_pairwise.csv")
    alignment = read(baseline, "projection_alignment.csv")
    correlations = read(baseline, "mixing_correlations.csv")
    heads = read(baseline, "attention_head_metrics.csv")
    head_transitions = read(baseline, "attention_head_transitions.csv")
    head_permutations = read(deep, "attention_head_permutation_geometry.csv")
    ffn = read(baseline, "ffn_pair_metrics.csv")
    ffn_neurons = read(deep, "ffn_neuron_summary.csv")
    layer_anomalies = read(baseline, "layer_anomalies.csv")
    dimensions = read(baseline, "intrinsic_dimension_metrics.csv")
    compression = read(baseline, "compression_metrics.csv")
    quality = read(deep, "output_data_quality_checks.csv")
    svd_sensitivity = read(deep, "randomized_svd_sensitivity.csv")
    within_subspaces = read(deep, "within_block_subspace_geometry.csv")
    turnover = read(deep, "adjacent_subspace_turnover.csv")
    projection_pairwise = read(deep, "projection_checkpoint_pairwise.csv")
    projection_stats = read(deep, "projection_checkpoint_statistics.csv")
    normalization_changes = read(deep, "normalization_adjacent_changes.csv")
    raw_stats = read(deep, "raw_tensor_statistics.csv")
    spectral_shapes = read(deep, "spectral_shape_metrics.csv")
    spectral_transitions = read(deep, "adjacent_spectral_transitions.csv")
    trajectory = read(deep, "trajectory_local_geometry.csv")
    omnibus = read(deep, "layer_omnibus_balanced.csv")
    derived_anomalies = read(deep, "derived_layer_anomalies.csv")

    checkpoint_labels = {
        "esmfold2": "ESMFold2",
        "esmfold2_fast": "ESMFold2 Fast",
        "esmfold2_experimental_cutoff2025": "Experimental 2025",
        "esmfold2_experimental_fast_cutoff2025": "Experimental Fast 2025",
    }
    mixing["checkpoint_label"] = mixing.checkpoint.map(checkpoint_labels)
    mixer_summary["checkpoint_label"] = mixer_summary.checkpoint.map(checkpoint_labels)

    head_layer = (
        heads.groupby("block", as_index=False)
        .agg(
            heads_vo_gt_05=("vo_overlap", lambda x: int((x > 0.5).sum())),
            heads_vo_gt_07=("vo_overlap", lambda x: int((x > 0.7).sum())),
            vo_overlap_mean=("vo_overlap", "mean"),
            vo_overlap_sd=("vo_overlap", "std"),
            o_stable_rank_mean=("o_stable_rank", "mean"),
            output_redundancy_mean=("within_layer_output_redundancy_mean", "mean"),
        )
        .assign(produced_state=lambda d: d.block + 1)
    )
    head_threshold_profile = pd.concat(
        [
            head_layer[["block", "produced_state", "heads_vo_gt_05"]]
            .rename(columns={"heads_vo_gt_05": "head_count"})
            .assign(threshold="V-O overlap > 0.5"),
            head_layer[["block", "produced_state", "heads_vo_gt_07"]]
            .rename(columns={"heads_vo_gt_07": "head_count"})
            .assign(threshold="V-O overlap > 0.7"),
        ],
        ignore_index=True,
    )

    align_rank16 = alignment[
        alignment.family.isin(["q", "k", "v", "gate", "value"])
        & (alignment["rank"] == 16)
        & (alignment.projection_variant == "raw")
    ]
    alignment_profile = (
        align_rank16.groupby(["block", "produced_state", "family"], as_index=False)
        .normalized_overlap.agg(["mean", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_overlap",
                "min": "minimum_overlap",
                "max": "maximum_overlap",
            }
        )
    )

    spectral_profile = layer_anomalies[
        (layer_anomalies.metric == "operator_effective_rank")
        & layer_anomalies.family.isin(["q", "k", "v", "gate", "value", "down"])
    ][["block", "produced_state", "family", "value", "depth_residual", "local_robust_z", "bh_q"]]
    spectral_profile = spectral_profile.rename(columns={"value": "effective_rank"})

    ffn_profile = ffn.merge(
        ffn_neurons[
            [
                "block",
                "triple_strength_top10_share",
                "triple_strength_top100_share",
                "gate_down_cosine__standard_deviation",
                "value_down_cosine__standard_deviation",
            ]
        ],
        on="block",
        how="left",
    ).assign(produced_state=lambda d: d.block + 1)

    summary: dict[str, object] = {
        "server_evidence": {
            "baseline_dir": str(baseline),
            "deep_dive_dir": str(deep),
            "quality_checks_passed": int(quality["status"].eq("pass").sum()),
            "quality_checks_total": int(len(quality)),
        },
        "mixer": {
            "state_51": _records(mixer_summary[["checkpoint", "state_51_mass", "state_51_rank"]]),
            "states_77_80": _records(mixer_summary[["checkpoint", "states_77_80_mass"]]),
            "pairwise_pearson_min": float(mixer_pairwise.pearson.min()),
            "pairwise_pearson_max": float(mixer_pairwise.pearson.max()),
            "pairwise_spearman_min": float(mixer_pairwise.spearman.min()),
            "pairwise_spearman_max": float(mixer_pairwise.spearman.max()),
        },
    }

    projection_targets: list[dict[str, object]] = []
    for family in ["gate", "value"]:
        for rank in [16, 32, 64, 128, 256]:
            for variant in ["raw", "layernorm_scaled_approximation"]:
                subset = alignment[
                    (alignment.family == family)
                    & (alignment["rank"] == rank)
                    & (alignment.projection_variant == variant)
                ]
                for checkpoint, frame in subset.groupby("checkpoint"):
                    series = frame.set_index("block").normalized_overlap
                    for block in [50, 51]:
                        value = float(series.loc[block])
                        local = series.loc[
                            (series.index >= block - 5) & (series.index <= block + 5)
                        ].drop(index=block)
                        projection_targets.append(
                            {
                                "checkpoint": checkpoint,
                                "family": family,
                                "rank": rank,
                                "projection_variant": variant,
                                "block": block,
                                "produced_state": block + 1,
                                "normalized_overlap": value,
                                "local_median_delta": value - float(local.median()),
                                "local_robust_z": _local_robust_z(series, block),
                                "global_rank_desc": _rank_desc(series, block),
                            }
                        )
    summary["projection_targets"] = projection_targets

    q_corr = correlations[
        (correlations.family == "q")
        & (correlations.metric.isin(["operator_effective_rank", "operator_participation_ratio", "operator_stable_rank"]))
        & (correlations.lag == 0)
    ]
    summary["q_spectrum_mixer_correlations"] = _records(q_corr)

    head50 = heads[heads.block == 50].sort_values("vo_overlap", ascending=False)
    head51 = heads[heads.block == 51].sort_values("vo_overlap", ascending=False)
    summary["attention"] = {
        "block_50": _records(head_layer[head_layer.block == 50]),
        "block_51": _records(head_layer[head_layer.block == 51]),
        "block_50_top_heads": _records(
            head50[
                [
                    "head",
                    "vo_overlap",
                    "q_spectral_norm",
                    "q_frobenius_norm",
                    "o_stable_rank",
                ]
            ].head(10)
        ),
        "block_51_top_heads": _records(
            head51[["head", "vo_overlap", "q_spectral_norm", "q_frobenius_norm", "o_stable_rank"]].head(10)
        ),
        "transition_49_52": _records(
            head_transitions[head_transitions.to_block.isin([50, 51, 52])]
            .merge(
                head_permutations[
                    [
                        "from_block",
                        "to_block",
                        "matching_gain",
                        "fixed_points",
                        "cycle_count",
                        "maximum_cycle_length",
                        "mean_absolute_head_displacement",
                        "permutation_inversion_count",
                    ]
                ],
                on=["from_block", "to_block"],
                how="left",
            )
        ),
    }

    summary["ffn"] = {
        "block_50": _records(ffn_profile[ffn_profile.block == 50]),
        "block_51": _records(ffn_profile[ffn_profile.block == 51]),
    }
    for metric in ["gate_down_cosine_mean", "gate_down_cosine_sd", "value_down_cosine_sd"]:
        series = ffn.set_index("block")[metric]
        summary["ffn"][f"block_50_{metric}_audit"] = {
            "value": float(series.loc[50]),
            "local_robust_z": _local_robust_z(series, 50),
            "rank_low": _rank_asc(series, 50),
            "rank_high": _rank_desc(series, 50),
        }

    summary["numerical_validation"] = {
        "randomized_svd_max_relative_singular_difference": float(
            svd_sensitivity.maximum_singular_value_relative_difference.max()
        ),
        "randomized_svd_median_relative_singular_difference": float(
            svd_sensitivity.median_singular_value_relative_difference.median()
        ),
        "randomized_svd_minimum_overlap": float(svd_sensitivity.normalized_overlap.min()),
        "selected_rows": _records(
            svd_sensitivity[svd_sensitivity.block.isin([0, 50, 51, 79])]
        ),
    }

    summary["projection_checkpoint_pairwise"] = _records(projection_pairwise)
    summary["projection_checkpoint_statistics"] = _records(projection_stats)
    summary["within_block_subspaces_50_51"] = _records(
        within_subspaces[
            within_subspaces.block.isin([50, 51])
            & within_subspaces["rank"].isin([16, 32, 64])
        ]
    )
    summary["adjacent_subspace_turnover_49_52"] = _records(
        turnover[turnover.to_block.isin([50, 51, 52]) & turnover["rank"].isin([16, 32, 64])]
    )
    summary["normalization_changes_49_52"] = _records(
        normalization_changes[normalization_changes.to_block.isin([50, 51, 52])]
    )
    summary["spectral_transitions_49_52"] = _records(
        spectral_transitions[spectral_transitions.to_block.isin([50, 51, 52])]
    )
    summary["trajectory_50_51"] = _records(trajectory[trajectory.block.isin([50, 51])])
    summary["omnibus_50_51"] = _records(omnibus[omnibus.block.isin([50, 51])])

    selected_raw_columns = [
        "block",
        "produced_state",
        "family",
        "rms",
        "standard_deviation",
        "skewness_sampled",
        "excess_kurtosis_sampled",
        "row_norm__gini_absolute",
        "column_norm__gini_absolute",
        "rows_coherence_mean_absolute",
        "columns_coherence_mean_absolute",
        "captured_frobenius_energy_fraction",
        "left_singular_vectors_mean_ipr",
        "right_singular_vectors_mean_ipr",
    ]
    summary["raw_weight_statistics_50_51"] = _records(
        raw_stats[raw_stats.block.isin([50, 51])][selected_raw_columns]
    )
    summary["spectral_shape_50_51"] = _records(
        spectral_shapes[
            spectral_shapes.block.isin([50, 51])
            & (spectral_shapes.geometry == "operator")
        ]
    )

    significant = derived_anomalies[
        derived_anomalies.block.isin([50, 51])
        & (derived_anomalies.local_bh_q_within_series < 0.05)
        & (derived_anomalies.depth_adjusted_bh_q_within_series < 0.05)
    ]
    summary["significant_deep_dive_metrics_50_51"] = {
        "count": int(len(significant)),
        "by_source": significant.groupby("source_table").size().sort_values(ascending=False).to_dict(),
        "rows": _records(significant),
    }

    dimension_targets = dimensions[dimensions.block.isin([50, 51])]
    summary["intrinsic_dimension_50_51"] = _records(dimension_targets)
    summary["compression_50_51"] = _records(compression[compression.block.isin([50, 51])])

    datasets = {
        "mixer_profile": _records(
            mixing[["checkpoint", "checkpoint_label", "state_index", "producing_block", "mixing_weight_pct"]]
        ),
        "mixer_summary": _records(mixer_summary),
        "projection_rank16_profile": _records(alignment_profile),
        "projection_gate_value_rank16_profile": _records(
            alignment_profile[alignment_profile.family.isin(["gate", "value"])]
        ),
        "spectral_effective_rank_profile": _records(spectral_profile),
        "spectral_q_effective_rank_profile": _records(
            spectral_profile[spectral_profile.family == "q"]
        ),
        "attention_head_layer_profile": _records(head_layer),
        "attention_head_threshold_profile": _records(head_threshold_profile),
        "ffn_profile": _records(ffn_profile),
        "checkpoint_projection_similarity": _records(projection_pairwise),
        "key_evidence": [
            {"finding": "State 51 mixer mass", "value": float(mixer_summary.state_51_mass.mean()), "unit": "fraction"},
            {"finding": "States 77-80 mixer mass", "value": float(mixer_summary.states_77_80_mass.mean()), "unit": "fraction"},
            {"finding": "Block 50 heads with VO overlap > 0.5", "value": int(head_layer.set_index("block").loc[50, "heads_vo_gt_05"]), "unit": "heads"},
            {"finding": "Block 50 heads with VO overlap > 0.7", "value": int(head_layer.set_index("block").loc[50, "heads_vo_gt_07"]), "unit": "heads"},
            {"finding": "Block 50 gate-down cosine mean", "value": float(ffn.set_index("block").loc[50, "gate_down_cosine_mean"]), "unit": "cosine"},
        ],
    }

    (out / "server_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "report_datasets.json").write_text(
        json.dumps(datasets, indent=2, sort_keys=True), encoding="utf-8"
    )
    sql_specs = {
        "key_evidence": "SELECT finding, value, unit FROM read_csv_auto('report_tables/key_evidence.csv') ORDER BY finding",
        "mixer_profile": "SELECT checkpoint, checkpoint_label, state_index, producing_block, mixing_weight_pct FROM read_csv_auto('report_tables/mixer_profile.csv') ORDER BY checkpoint, state_index",
        "mixer_summary": "SELECT * FROM read_csv_auto('report_tables/mixer_summary.csv') ORDER BY state_51_mass DESC",
        "projection_gate_value_rank16_profile": "SELECT block, produced_state, family, mean_overlap, minimum_overlap, maximum_overlap FROM read_csv_auto('report_tables/projection_gate_value_rank16_profile.csv') ORDER BY family, block",
        "attention_head_threshold_profile": "SELECT block, produced_state, threshold, head_count FROM read_csv_auto('report_tables/attention_head_threshold_profile.csv') ORDER BY threshold, block",
        "spectral_q_effective_rank_profile": "SELECT block, produced_state, family, effective_rank, depth_residual, local_robust_z, bh_q FROM read_csv_auto('report_tables/spectral_q_effective_rank_profile.csv') ORDER BY block",
    }
    table_dir = out / "report_tables"
    table_dir.mkdir(exist_ok=True)
    for dataset_name in sql_specs:
        pd.DataFrame(datasets[dataset_name]).to_csv(table_dir / f"{dataset_name}.csv", index=False)
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("duckdb is required to validate report SQL") from exc
    previous_cwd = Path.cwd()
    sql_validation: dict[str, object] = {}
    try:
        os.chdir(out)
        connection = duckdb.connect(database=":memory:")
        for dataset_name, sql in sql_specs.items():
            (out / f"{dataset_name}.sql").write_text(sql + ";\n", encoding="utf-8")
            frame = connection.execute(sql).fetchdf()
            expected = pd.DataFrame(datasets[dataset_name])
            if len(frame) != len(expected):
                raise RuntimeError(
                    f"SQL row-count mismatch for {dataset_name}: {len(frame)} != {len(expected)}"
                )
            sql_validation[dataset_name] = {
                "columns": list(frame.columns),
                "row_count": len(frame),
                "sql": sql,
            }
        connection.close()
    finally:
        os.chdir(previous_cwd)
    (out / "sql_validation.json").write_text(
        json.dumps(sql_validation, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest_rows = []
    for path in sorted(deep.rglob("*")):
        if path.is_file():
            manifest_rows.append(
                {
                    "path": str(path.relative_to(deep)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path) if path.stat().st_size < 16_000_000 else None,
                }
            )
    pd.DataFrame(manifest_rows).to_csv(out / "server_artifact_manifest.csv", index=False)
    print(json.dumps({"outputs": sorted(p.name for p in out.iterdir()), "quality": summary["server_evidence"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
