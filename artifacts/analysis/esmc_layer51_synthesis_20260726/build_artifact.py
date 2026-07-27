#!/usr/bin/env python3
"""Assemble the validated portable report from server-produced evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GENERATED_AT = "2026-07-26T11:30:00Z"
TITLE = "ESMC-6B State 51 Weight Geometry"


def split_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    title, body = markdown.split("\n", 1)
    sections: list[tuple[str, str]] = []
    for match in re.finditer(r"(?m)^## (.+)$", body):
        start = match.end()
        next_match = re.search(r"(?m)^## .+$", body[start:])
        end = start + next_match.start() if next_match else len(body)
        sections.append((match.group(1).strip(), body[start:end].strip()))
    return title.strip(), sections


def main() -> int:
    datasets = json.loads((ROOT / "report_datasets.json").read_text(encoding="utf-8"))
    markdown = (ROOT / "REPORT_SOURCE.md").read_text(encoding="utf-8")
    title_line, sections = split_sections(markdown)
    if title_line != f"# {TITLE}":
        raise ValueError(f"Unexpected report title: {title_line!r}")

    sources = [
        {
            "id": "server_audit",
            "label": "Server-generated ESMC weight-geometry audit",
            "path": "server_audit_summary.json",
        },
        {
            "id": "deep_dive_archive",
            "label": "Complete weights-only deep-dive provenance and data dictionary",
            "path": "deep_dive_provenance.json",
        },
        {
            "id": "paper_s26",
            "label": "Language Modeling Materializes a World Model of Protein Biology, Figure S26",
            "href": "https://doi.org/10.64898/2026.06.03.729735",
        },
        {
            "id": "fastplms_implementation",
            "label": "FastPLMs ESMFold2 LanguageModelShim implementation",
            "path": "modeling_esmfold2_common.py",
        },
        {"id": "sql_key_evidence", "label": "Executed DuckDB query for headline evidence", "path": "key_evidence.sql"},
        {"id": "sql_mixer_profile", "label": "Executed DuckDB query for the 81-state mixer profile", "path": "mixer_profile.sql"},
        {"id": "sql_mixer_summary", "label": "Executed DuckDB query for the checkpoint mixer table", "path": "mixer_summary.sql"},
        {"id": "sql_projection_profile", "label": "Executed DuckDB query for rank-16 projection overlap", "path": "projection_gate_value_rank16_profile.sql"},
        {"id": "sql_attention_profile", "label": "Executed DuckDB query for attention-head coupling counts", "path": "attention_head_threshold_profile.sql"},
        {"id": "sql_q_spectrum", "label": "Executed DuckDB query for Q effective-rank depth profile", "path": "spectral_q_effective_rank_profile.sql"},
    ]

    cards = [
        {
            "id": "state51_mass",
            "description": "Mean of the four checkpoint state-51 softmax weights.",
            "dataset": "key_evidence",
            "filter": {"finding": "State 51 mixer mass"},
            "metrics": [{"label": "Mean state-51 mass", "field": "value", "format": "percent"}],
            "sourceId": "sql_key_evidence",
        },
        {
            "id": "late_state_mass",
            "description": "Mean mass assigned to states 77-80 across the four checkpoints.",
            "dataset": "key_evidence",
            "filter": {"finding": "States 77-80 mixer mass"},
            "metrics": [{"label": "Mean states 77-80 mass", "field": "value", "format": "percent"}],
            "sourceId": "sql_key_evidence",
        },
        {
            "id": "high_vo_heads",
            "description": "Number of block-50 heads with V-O subspace overlap above 0.5.",
            "dataset": "key_evidence",
            "filter": {"finding": "Block 50 heads with VO overlap > 0.5"},
            "metrics": [{"label": "High-coupling heads", "field": "value", "format": "number"}],
            "sourceId": "sql_key_evidence",
        },
        {
            "id": "extreme_vo_heads",
            "description": "Number of block-50 heads with V-O subspace overlap above 0.7.",
            "dataset": "key_evidence",
            "filter": {"finding": "Block 50 heads with VO overlap > 0.7"},
            "metrics": [{"label": "Extreme-coupling heads", "field": "value", "format": "number"}],
            "sourceId": "sql_key_evidence",
        },
    ]

    charts = [
        {
            "id": "mixer_profile",
            "title": "ESMFold2 mixing weight by ESMC state",
            "subtitle": "All 81 states and four supported checkpoints; values are percentage points of softmax mass.",
            "showDescription": True,
            "intent": "trend",
            "question": "How is folding-model mixing mass distributed across ESMC depth, and is state 51 consistently elevated?",
            "rationale": "A multi-series line chart preserves the ordered 81-state profile and makes both the late-layer concentration and the shared state-51 bump visible.",
            "comparisonContext": {"grain": "ESMC hidden state", "denominator": "81-state softmax mass", "unit": "%"},
            "type": "line",
            "dataset": "mixer_profile",
            "encodings": {
                "x": {"field": "state_index", "type": "quantitative", "label": "ESMC state"},
                "y": {"field": "mixing_weight_pct", "type": "quantitative", "label": "Mixing mass", "unit": "%"},
                "color": {"field": "checkpoint_label", "type": "nominal", "label": "Checkpoint"},
                "tooltip": [
                    {"field": "checkpoint_label", "type": "nominal", "label": "Checkpoint"},
                    {"field": "state_index", "type": "quantitative", "label": "State"},
                    {"field": "mixing_weight_pct", "type": "quantitative", "label": "Mass", "unit": "%"},
                ],
            },
            "xAxisTitle": "ESMC hidden-state index",
            "yAxisTitle": "Softmax mixing mass (%)",
            "valueFormat": "number",
            "unit": "%",
            "layout": "full",
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "labelAsc"},
            "labels": {"values": "none"},
            "referenceLines": [{"axis": "x", "value": 51, "label": "State 51", "color": "neutral", "lineStyle": "dashed"}],
            "maxRows": 400,
            "sourceId": "sql_mixer_profile",
        },
        {
            "id": "projection_alignment",
            "title": "Rank-16 overlap with the ESMFold2 projection",
            "subtitle": "Four-checkpoint mean raw overlap for SwiGLU gate and value input subspaces; isotropic expectation is 0.10.",
            "showDescription": True,
            "intent": "trend",
            "question": "Where in depth do compact FFN subspaces align with the shared folding projection?",
            "rationale": "A two-series depth profile reveals the localized block-50 peak while retaining the global baseline and neighboring layers.",
            "comparisonContext": {"grain": "transformer block", "baseline": "isotropic expectation 0.10", "normalization": "mean squared canonical correlation"},
            "type": "line",
            "dataset": "projection_gate_value_rank16_profile",
            "encodings": {
                "x": {"field": "block", "type": "quantitative", "label": "Block"},
                "y": {"field": "mean_overlap", "type": "quantitative", "label": "Normalized overlap"},
                "color": {"field": "family", "type": "nominal", "label": "FFN family"},
                "tooltip": [
                    {"field": "family", "type": "nominal", "label": "Family"},
                    {"field": "block", "type": "quantitative", "label": "Block"},
                    {"field": "produced_state", "type": "quantitative", "label": "Produced state"},
                    {"field": "mean_overlap", "type": "quantitative", "label": "Mean overlap"},
                    {"field": "minimum_overlap", "type": "quantitative", "label": "Checkpoint minimum"},
                    {"field": "maximum_overlap", "type": "quantitative", "label": "Checkpoint maximum"},
                ],
            },
            "xAxisTitle": "ESMC transformer block",
            "yAxisTitle": "Normalized subspace overlap",
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "labelAsc"},
            "labels": {"values": "none"},
            "referenceLines": [
                {"axis": "x", "value": 50, "label": "Produces state 51", "color": "neutral", "lineStyle": "dashed"},
                {"axis": "y", "value": 0.1, "label": "Isotropic expectation", "color": "neutral", "lineStyle": "dotted"},
            ],
            "maxRows": 200,
            "sourceId": "sql_projection_profile",
        },
        {
            "id": "attention_head_counts",
            "title": "High V-O coupling heads by block",
            "subtitle": "Counts among 40 heads at overlap thresholds 0.5 and 0.7.",
            "showDescription": True,
            "intent": "trend",
            "question": "Is the block-50 V-O event confined to one head or distributed across the layer?",
            "rationale": "Two same-unit count series across all 80 blocks show whether the number of strongly coupled heads is locally and globally exceptional.",
            "comparisonContext": {"grain": "transformer block", "denominator": "40 attention heads", "unit": "heads"},
            "type": "line",
            "dataset": "attention_head_threshold_profile",
            "encodings": {
                "x": {"field": "block", "type": "quantitative", "label": "Block"},
                "y": {"field": "head_count", "type": "quantitative", "label": "Head count"},
                "color": {"field": "threshold", "type": "nominal", "label": "Coupling threshold"},
                "tooltip": [
                    {"field": "block", "type": "quantitative", "label": "Block"},
                    {"field": "threshold", "type": "nominal", "label": "Threshold"},
                    {"field": "head_count", "type": "quantitative", "label": "Heads"},
                ],
            },
            "xAxisTitle": "ESMC transformer block",
            "yAxisTitle": "Number of heads",
            "valueFormat": "number",
            "unit": "heads",
            "layout": "full",
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "spec"},
            "labels": {"values": "none"},
            "referenceLines": [{"axis": "x", "value": 50, "label": "Produces state 51", "color": "neutral", "lineStyle": "dashed"}],
            "maxRows": 100,
            "sourceId": "sql_attention_profile",
        },
        {
            "id": "q_effective_rank",
            "title": "Q operator effective rank by block",
            "subtitle": "Entropy-based effective rank from the uncentered singular spectrum of each 2560 x 2560 Q matrix.",
            "showDescription": True,
            "intent": "trend",
            "question": "How does Q spectral dimension change through depth around the state-51 producer?",
            "rationale": "A single ordered layer profile shows the local block-50 spectrum in the context of the strong global depth trend that motivates detrending.",
            "comparisonContext": {"grain": "transformer block", "normalization": "exp entropy of squared singular-value energy"},
            "type": "line",
            "dataset": "spectral_q_effective_rank_profile",
            "encodings": {
                "x": {"field": "block", "type": "quantitative", "label": "Block"},
                "y": {"field": "effective_rank", "type": "quantitative", "label": "Effective rank"},
                "tooltip": [
                    {"field": "block", "type": "quantitative", "label": "Block"},
                    {"field": "effective_rank", "type": "quantitative", "label": "Effective rank"},
                    {"field": "depth_residual", "type": "quantitative", "label": "Cubic-depth residual"},
                    {"field": "local_robust_z", "type": "quantitative", "label": "Local robust z"},
                ],
            },
            "xAxisTitle": "ESMC transformer block",
            "yAxisTitle": "Q effective rank",
            "valueFormat": "number",
            "layout": "full",
            "palette": {"kind": "sequential"},
            "labels": {"values": "none"},
            "referenceLines": [{"axis": "x", "value": 50, "label": "Produces state 51", "color": "neutral", "lineStyle": "dashed"}],
            "maxRows": 100,
            "sourceId": "sql_q_spectrum",
        },
    ]

    tables: list[dict[str, object]] = []

    section_chart = {
        "State 51 is consistently preferred by all four folding checkpoints": "mixer_profile",
        "Projection alignment is the strongest direct explanation": "projection_alignment",
        "Attention geometry indicates a coordinated multi-head event": "attention_head_counts",
        "Spectra point to compact dominant directions, but not a low-rank block": "q_effective_rank",
    }
    chart_notes = {
        "mixer_profile": "**How to read the profile.** The shared late-depth rise is the dominant pattern, but all four curves retain a separate state-51 peak. The dashed reference identifies the state produced by block 50; the exact checkpoint values follow in the table.",
        "projection_alignment": "**How to read the overlap.** Values above 0.10 exceed the isotropic expectation. Gate and value both form a local peak at block 50, and the minimum-to-maximum checkpoint range remains positive relative to neighboring blocks.",
        "attention_head_counts": "**How to read the head event.** Block 50 is the only layer with 10 heads above 0.5 and 6 above 0.7. The immediate collapse at block 51 favors a transient, coordinated write event over a persistent single-head specialization.",
        "q_effective_rank": "**How to read the spectrum.** The global depth shape is substantial, which is why correlations with the mixer were detrended. Block 50 is not a singular rank collapse; its relevance comes from compact leading directions combined with the projection and head/FFN evidence.",
    }

    blocks: list[dict[str, object]] = [
        {
            "id": "report_title",
            "type": "markdown",
            "body": title_line,
            "layout": "full",
        },
    ]
    for index, (heading, body) in enumerate(sections):
        source_id = "server_audit" if heading not in {
            "The apparent contradiction is between objectives, not measurements of one quantity",
            "Scope and reproducibility",
            "Further questions",
        } else None
        block: dict[str, object] = {
            "id": f"section_{index:02d}",
            "type": "markdown",
            "body": f"## {heading}\n\n{body}",
            "layout": "full",
        }
        if source_id:
            block["sourceId"] = source_id
        blocks.append(block)
        if heading == "Technical summary":
            blocks.append({"id": "headline_metrics", "type": "metric-strip", "cardIds": ["state51_mass", "late_state_mass", "high_vo_heads", "extreme_vo_heads"], "layout": "full"})
        chart_id = section_chart.get(heading)
        if chart_id:
            blocks.append({"id": f"note_{chart_id}", "type": "markdown", "body": chart_notes[chart_id], "layout": "full", "sourceId": "server_audit"})
            blocks.append({"id": f"block_{chart_id}", "type": "chart", "chartId": chart_id, "layout": "full"})

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "Weights-only synthesis of the ESMC-6B state-51 ESMFold2 mixing outlier.",
            "generatedAt": GENERATED_AT,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": datasets,
            "accessIssues": [],
        },
        "sources": sources,
    }
    (ROOT / "artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"blocks": len(blocks), "charts": len(charts), "tables": len(tables), "datasets": {key: len(value) for key, value in datasets.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
