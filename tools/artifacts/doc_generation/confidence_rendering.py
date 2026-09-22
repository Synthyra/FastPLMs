"""Render confidence-head identities and scientifically scoped evaluation evidence."""

from __future__ import annotations

import math
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from fastplms.registry import ModelSpec
from tools.artifacts.doc_generation.confidence_evidence import (
    CONFIDENCE_RESEARCH_EVIDENCE,
    ConfidenceEvidence,
    load_confidence_evidence,
)
from tools.artifacts.doc_generation.evidence_links import evidence_reference


def _withheld_metrics_notice(evidence: ConfidenceEvidence) -> str:
    return (
        f"Confidence metric review status: `{evidence.review_status}`. "
        "Metrics and acceptance claims are withheld until the evidence is validated."
    )


def _live_confidence_section(spec: ModelSpec) -> str:
    if spec.id not in {"esmfold2_300", "esmfold2_600"} or spec.confidence_adaptation is not None:
        return ""
    return f"""## Current v2 confidence head

The v2 reproduction publishes current EMA heads during training to the public
[artifact dataset](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/main/confidence/v2/v2-reproduction-20260922/{spec.id}).
After the first trained checkpoint is uploaded, the Hub config binds this model
to its latest published head. `AutoModel.from_pretrained` then loads that head
and enables pLDDT, PAE, pTM and iPTM by default. These training checkpoints have
pending evaluation; the historical results below do not validate them.

Each load resolves an immutable dataset revision and verifies the head's hash
and native state. `model.config.confidence_head_resolved` records the revision,
training update and head identity. An already loaded model keeps its head;
reload to obtain a newer publication. `save_pretrained` embeds the exact loaded
head, so the saved model reloads without fetching a newer one.

Pass `load_confidence_head=False` to load the unchanged base without its external
head. This option does not remove a head already embedded in a saved model.
External loading supports one resident model device and cached offline loads;
split-device and disk-offloaded loading are unsupported. See the
[confidence training guide](https://github.com/Synthyra/FastPLMs/blob/main/docs/confidence_training.md#ongoing-hugging-face-checkpoints).

"""


def _confidence_adaptation_section(spec: ModelSpec, root: Path | None) -> str:
    """Render identity and measured evidence for a separately trained head."""

    adaptation = spec.confidence_adaptation
    if adaptation is None:
        return ""
    evidence: Mapping[str, object] = {}
    if root is not None:
        report = load_confidence_evidence(
            root / adaptation.evidence_path, protocol="pilot", evidence_root=root
        )
        if not report.can_report_metrics:
            return f"""## Adapted confidence head

This artifact adds a separately trained Synthyra native confidence head to the
frozen base checkpoint. Its evaluation evidence is under review.

{_withheld_metrics_notice(report)}

- Adapted head SHA-256: `{adaptation.head_sha256}`
- Evaluation record: {adaptation.evaluation_url}
- Evidence: {evidence_reference(adaptation.evidence_path, root)}

"""
        evidence = report.payload
    candidate = evidence.get("candidate")
    if not isinstance(candidate, dict):
        candidate = {}

    def metric(name: str) -> str:
        value = candidate.get(name)
        return f"{value:.3f}" if isinstance(value, (int, float)) else "recorded"

    def ranking(value: object) -> str:
        if not isinstance(value, dict):
            return "recorded"
        selection = value.get("selection_accuracy")
        count = value.get("comparable_count")
        if isinstance(selection, (int, float)) and isinstance(count, int):
            return f"{selection:.1%} ({count} comparable targets)"
        return "recorded"

    candidate_ranking = evidence.get("candidate_sample_ranking")
    if not isinstance(candidate_ranking, dict):
        candidate_ranking = {}
    metrics = (
        f"atom MAE {metric('atom_mae')}; Cα MAE {metric('ca_mae')}; "  # noqa: RUF001
        f"calibration error {metric('calibration_error_10bin')}; "
        f"pLDDT Spearman {metric('target_plddt_spearman')}; "
        f"iPTM/DockQ Spearman {metric('iptm_dockq_spearman')}; "
        f"pLDDT CE {metric('plddt_ce')}; PAE CE {metric('pae_ce')}; "
        f"PAE overflow {metric('pae_overflow_fraction')}"
    )
    return f"""## Adapted confidence head

This artifact keeps the frozen official base checkpoint and adds a separately
trained Synthyra native confidence head. Only the confidence head was trained;
the backbone and folding parameters remain frozen. The head supplies pLDDT and
PAE, with pTM and iPTM derived by the existing runtime. Call
`model.infer_protein(..., calculate_confidence=False)` to skip confidence while
retaining structure inference.

The held-out evaluation used 128 targets: 64 monomers and 64 dimers, with
64-384 total residues per target.

The head was initialized from the pinned experimental Fast donor and trained
on 1,024 experimental AtlasFold targets: 512 monomers and 512 dimers. The
training recipe used atom-specific heavy-atom lDDT targets and native-order PAE
targets, with pLDDT cross-entropy plus `0.1 * PAE cross-entropy`. The full
recipe and split rules are documented in the
[confidence training guide](https://github.com/Synthyra/FastPLMs/blob/main/docs/confidence_training.md).

- Donor: `{adaptation.donor_repo}@{adaptation.donor_revision}`
- Donor weight SHA-256: `{adaptation.donor_weight_sha256}`
- Base weight SHA-256: `{adaptation.base_weight_sha256}`
- Adapted head SHA-256: `{adaptation.head_sha256}`
- Training record: {adaptation.training_url}
- Evaluation record: {adaptation.evaluation_url}
- Evidence: {evidence_reference(adaptation.evidence_path, root)}
- Held-out evidence: {metrics}
- Two-seed pLDDT sample selection: {ranking(candidate_ranking.get("plddt_selection"))}
- Two-seed interface sample selection: {ranking(candidate_ranking.get("interface_selection"))}

Both models show near-chance within-target sample selection with two seeds only.
These are weak diagnostics and do not support a general sample-ranking claim.

"""


CONFIDENCE_RESEARCH_ROWS = (
    ("pLDDT against all-atom lDDT, Spearman", "plddt_lddt_spearman", 3),
    ("pTM against TM-score, Spearman", "ptm_tm_spearman", 3),
    ("ipTM against DockQ, Spearman", "iptm_dockq_spearman", 3),
    ("Atom pLDDT mean absolute error", "atom_plddt_mae", 4),
    ("Calibration error, 10 bins", "calibration_error_10bin", 4),
    ("pLDDT cross-entropy", "plddt_ce", 3),
    ("PAE cross-entropy", "pae_ce", 3),
    ("Within-target lDDT selection accuracy", "within_target_plddt_accuracy", 3),
    ("Within-target ipTM against DockQ selection accuracy", "within_target_iptm_dockq_accuracy", 3),
    ("Top-1 selection regret", "top1_regret", 4),
    ("Random-choice regret", "random_selection_regret", 4),
    ("Unresolved against resolved residue AUROC", "disorder_auroc", 3),
    ("Resolved residue mean pLDDT", "resolved_residue_mean_plddt", 3),
    ("Unresolved residue mean pLDDT", "unresolved_residue_mean_plddt", 3),
    ("Resolved residues below pLDDT 50", "resolved_fraction_below_50", 3),
    ("Unresolved residues below pLDDT 50", "unresolved_fraction_below_50", 3),
)


CONFIDENCE_AGREEMENT_ROWS = (("Mean pLDDT", "mean_plddt"), ("pTM", "ptm"), ("ipTM", "iptm"))


def _learning_rate(value: float) -> str:
    """Format a learning rate as `1e-4` rather than `0.0001`."""
    return f"{value:.0e}".replace("e-0", "e-")


def _interval_cell(interval: Sequence[float] | None, digits: int) -> str:
    """Format one bootstrap interval, or an empty cell for a measurement without one."""
    if interval is None:
        return ""
    low, high = interval
    return f"{low:.{digits}f} to {high:.{digits}f}"


def _metric_cell(value: object, digits: int, *, signed: bool = False) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Confidence metric must be finite or undefined: {value!r}")
    return format(value, f"{'+' if signed else ''}.{digits}f")


def _fill_paragraphs(paragraphs: Iterable[str]) -> str:
    """Wrap interpolated card prose to the card width."""
    return "\n\n".join(
        textwrap.fill(paragraph, width=79, break_long_words=False, break_on_hyphens=False)
        for paragraph in paragraphs
    )


def _confidence_research_section(spec: ModelSpec, root: Path | None) -> str:
    """Render measured results for a confidence head trained separately and not published."""

    evidence_path = CONFIDENCE_RESEARCH_EVIDENCE.get(spec.id)
    if spec.confidence_adaptation is not None or evidence_path is None or root is None:
        return ""
    report = load_confidence_evidence(root / evidence_path, protocol="v2", evidence_root=root)
    evidence, review = report.payload, report.review
    if not report.can_report_metrics:
        audit = review.get("training_history_audit")
        recovery = review.get("raw_prediction_recovery")
        if not (
            report.review_status == "requires_recomputation"
            and isinstance(audit, Mapping)
            and isinstance(recovery, Mapping)
            and recovery.get("status") == "unavailable"
            and audit.get("maximum") == 0
            and isinstance(audit.get("history_rows"), int)
            and isinstance(audit.get("wandb_url"), str)
        ):
            return f"""## Separately trained confidence head

The pinned base checkpoint has its confidence head disabled. The historical
GH200 confidence weights were not recovered and remain unpublished.

{_withheld_metrics_notice(report)}

"""
        return f"""## Separately trained confidence head

The pinned base checkpoint has its confidence head disabled. The historical
GH200 confidence weights were not recovered and remain unpublished.

The archived correlations, bootstrap intervals, and acceptance gates require
recomputation after correcting tied ranks in Spearman correlation. Raw test
predictions were not recovered from the closed GH200 workstation or W&B, so
the historical numbers are withheld here pending raw prediction recovery.
They do not establish corrected quality or acceptance results.

The [W&B training history audit]({audit["wandb_url"]}) inspected all
{audit["history_rows"]} update rows and found zero skipped training targets.
The skipped-target gradient bug therefore did not affect this recorded run's
training weights. This audit does not validate the archived correlations.

The [confidence training guide](https://github.com/Synthyra/FastPLMs/blob/main/docs/confidence_training.md)
preserves the historical results and their review status.

"""
    data, training, test = evidence["data"], evidence["training"], evidence["test"]
    config = training["config"]
    head, production = test["heads"]["v2"], test["heads"]["production"]
    agreement = evidence["production_agreement"]["v2"]
    configurations = " and ".join(f"`{name}`" for name in data["configurations"])
    comparison = "\n".join(
        f"| {label} | {_metric_cell(head[key], digits)} | "
        f"{_interval_cell(head['interval_95'].get(key), digits)} | "
        f"{_metric_cell(production[key], digits)} |"
        for label, key, digits in CONFIDENCE_RESEARCH_ROWS
    )
    correlations = "\n".join(
        f"| {label} | {_metric_cell(agreement[f'{key}_spearman'], 3)} | "
        f"{_metric_cell(agreement[f'{key}_mean_difference'], 3, signed=True)} |"
        for label, key in CONFIDENCE_AGREEMENT_ROWS
    )
    evaluation_scope = (
        f"Evaluation folded {test['targets']} held-out targets with "
        f"{test['samples_per_target']} samples each at the same settings and scored "
        "every sample with the trained head. "
    )
    if report.review_status == "corrected_test_metrics":
        evaluation_scope = (
            f"These corrected metrics reuse saved predictions for {test['targets']} targets "
            f"with {test['samples_per_target']} samples each from the original evaluation. "
            "The test split is spent; this recomputation is not a new held-out evaluation. "
            "The training validation correlation remains unverified pending cache rescoring. "
        )
    introduction = _fill_paragraphs(
        (
            "The pinned base checkpoint has its confidence head disabled. The historical results below come "
            "from a "
            "confidence head trained separately for this backbone. Those weights are not published "
            "and are not part of this artifact.",
            f"The head was initialized from `{training['donor_repo']}` at revision "
            f"`{training['donor_revision']}`, and the backbone, folding trunk, and diffusion "
            "module "
            f"stayed frozen. Training targets come from the {configurations} configurations of "
            f"[{data['dataset']}](https://huggingface.co/datasets/{data['dataset']}), limited to "
            f"structures resolved to {data['maximum_resolution_angstrom']} Å or better. "
            "Chains were "
            "clustered at 40% sequence identity, and test targets share no cluster with a training "
            "target.",
            f"Each update folded {config['targets_per_update']} new targets with "
            f"{config['samples_per_target']} diffusion samples each, at {config['num_loops']} "
            f"recycling loops and {config['num_sampling_steps']} diffusion steps, then minimized "
            f"pLDDT cross-entropy plus PAE cross-entropy plus {config['ranking_weight']} times a "
            "pairwise loss that ranks the samples of one target. Optimization used AdamW at "
            f"`{_learning_rate(config['learning_rate'])}` with cosine decay to "
            f"`{_learning_rate(config['minimum_learning_rate'])}` and an "
            "exponential moving average "
            f"of the weights. The run completed {training['updates']} updates in "
            f"{training['elapsed_hours']:.1f} hours on one GH200 and kept its final moving-average "
            "weights.",
            evaluation_scope + "Production `esmfold2`, which uses the 6B ESMC backbone and its own "
            "released confidence "
            f"head, folded and scored its own {test['samples_per_target']} samples of the same "
            "targets. Intervals are 95% intervals from one bootstrap over test targets.",
        )
    )
    definitions = _fill_paragraphs(
        (
            "Selection accuracy counts sample pairs whose measured quality differs by "
            "at least 0.01 "
            f"lDDT or 0.05 DockQ: {head['within_target_plddt_pairs']:.0f} lDDT pairs and "
            f"{head['within_target_iptm_dockq_pairs']:.0f} ipTM pairs for this head, "
            f"{production['within_target_plddt_pairs']:.0f} and "
            f"{production['within_target_iptm_dockq_pairs']:.0f} for production. Regret is the "
            "measured quality lost by taking the top-ranked sample instead of the best "
            "one, next to "
            "the loss from an average sample.",
            "Residues whose C-alpha atom is missing from the experimental structure stand in for "
            "disordered regions, and no head receives pLDDT labels on those atoms. "
            "The AUROC is the "
            "probability that an unresolved residue receives a lower pLDDT than a resolved one.",
            "Agreement with production uses the per-target mean of each model's samples, with ipTM "
            f"over the {agreement['multi_chain_targets']:.0f} multi-chain targets. "
            "Each model folds "
            "its own samples, so these compare per-target scores rather than two scores of one "
            "structure.",
        )
    )
    return f"""## Separately trained confidence head

{introduction}

| Measurement | This head | 95% interval | Production `esmfold2` |
| --- | ---: | ---: | ---: |
{comparison}

| Agreement with production `esmfold2` | Spearman | Mean difference |
| --- | ---: | ---: |
{correlations}

{definitions}

The recipe, the split rules, the per-stratum results, and the acceptance gates are in the
[confidence training guide](https://github.com/Synthyra/FastPLMs/blob/main/docs/confidence_training.md).

"""
