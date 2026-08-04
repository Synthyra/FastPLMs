"""Generate model support data and model cards from the typed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
import textwrap
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from fastplms.registry import ModelFamily, ModelRegistry, ModelSpec, get_model_registry
from tools.artifacts.license_metadata import (
    render_checkpoint_terms,
    render_hub_license_yaml,
)
from tools.remote.biohub_reference_environment import (
    BiohubReferenceEnvironmentError,
    validate_biohub_reference_environment_evidence,
)


GENERATED_MARKER = "<!-- Generated from src/fastplms/models.toml. Do not edit. -->"
BINDER_IMAGE_URL = (
    "https://raw.githubusercontent.com/Synthyra/FastPLMs/main/"
    "docs/assets/egfr_fastplms_binder_design.png"
)
ESMC_RELEASE_DOCUMENTATION = """\
Detailed backend measurements, release guardrails, and the GH200 package
compatibility exception are maintained in the
[attention backend guide](https://github.com/Synthyra/FastPLMs/blob/main/docs/attention_backends.md)
and
[release evidence manifest](https://github.com/Synthyra/FastPLMs/blob/main/docs/generated/capability_evidence.md).
"""

FAMILY_DOCUMENTATION = {
    "esm2": ("../models.md#esm2", "../../examples/embedding_and_retrieval.py"),
    "esm_plusplus": (
        "../models.md#esm-and-esmc",
        "../../examples/attention_switching.py",
    ),
    "esm3": ("../models.md#esm3", "../../examples/generation.py"),
    "e1": ("../models.md#e1", "../../examples/e1_rag.py"),
    "dplm": ("../models.md#dplm", "../../examples/generation.py"),
    "dplm2": ("../models.md#dplm2", "../../examples/generation.py"),
    "ankh": ("../models.md#ankh", "../../examples/ankh_embeddings.py"),
    "boltz2": ("../models.md#boltz2", "../../examples/structure_preparation.py"),
    "esmfold": ("../models.md#esmfold", "../../examples/structure_preparation.py"),
    "esmfold2": ("../esmfold2.md", "../../examples/structure_preparation.py"),
}

AUTO_CLASS_STATUS = {
    "AutoConfig": "FastPLMs extension",
    "AutoModel": "pretrained",
    "AutoModelForMaskedLM": "pretrained",
    "AutoModelForProteinFolding": "pretrained",
    "AutoModelForSeq2SeqLM": "pretrained",
    "AutoModelForSequenceClassification": "base weights + untrained task head",
    "AutoModelForTokenClassification": "base weights + untrained task head",
}

EMBEDDING_FAMILIES = frozenset(
    {
        "ankh",
        "dplm",
        "dplm2",
        "e1",
        "esm2",
        "esm3",
        "esm_plusplus",
        "esmfold2",
    }
)

SEQUENCE_TTT_AUTO_CLASSES = {
    "ankh": "AutoModelForMaskedLM",
    "dplm": "AutoModelForMaskedLM",
    "dplm2": "AutoModelForMaskedLM",
    "e1": "AutoModelForMaskedLM",
    "esm2": "AutoModelForMaskedLM",
    "esm3": "AutoModel",
    "esm_plusplus": "AutoModelForMaskedLM",
}


@dataclass(frozen=True)
class EvidenceSelector:
    """One executable validation selector and the exact scope it supports."""

    tier: str
    targets: tuple[str, ...]
    scope: str


@dataclass(frozen=True)
class CapabilityEvidenceRow:
    """A documented capability backed by explicit evidence selectors."""

    capability: str
    guide: str
    example: str
    evidence: tuple[str, ...]


class EsmcReportError(ValueError):
    """Raised when frozen ESMC release evidence is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class EsmcRuntimeIdentity:
    """Artifact-derived runtime identity required by every ESMC report."""

    runtime_revision: str
    source_tree_sha256: str
    runtime_bundle_sha256: str


@dataclass(frozen=True, slots=True)
class EsmcReportSet:
    """One complete, cross-device-consistent 30-record ESMC evidence set."""

    reports: tuple[dict[str, object], ...]
    runtime_identity: EsmcRuntimeIdentity
    candidate_environment: dict[str, object]
    reference_environment: dict[str, object]

    def select(self, model_id: str) -> tuple[dict[str, object], ...]:
        """Return the ten backend/panel reports for one checkpoint."""

        return tuple(report for report in self.reports if report["model_id"] == model_id)

    def get(self, model_id: str, backend: str, panel: str) -> dict[str, object]:
        """Return one uniquely keyed report from the complete evidence set."""

        matches = tuple(
            report
            for report in self.reports
            if report["model_id"] == model_id
            and report["configured_backend"] == backend
            and isinstance(report["panel"], Mapping)
            and report["panel"]["kind"] == panel
        )
        if len(matches) != 1:
            raise EsmcReportError(
                f"ESMC evidence key {(model_id, backend, panel)!r} resolved to "
                f"{len(matches)} reports"
            )
        return matches[0]


ESMC_DIAGNOSTIC_SCHEMA_VERSION = 3
ESMC_MODEL_IDS = ("esmc_small", "esmc_large", "esmc_6b")
ESMC_SAE_EXAMPLES = {
    "esmc_small": ("biohub/ESMC-300M-sae-layer23-k64-codebook65536", 23),
    "esmc_large": ("biohub/ESMC-600M-sae-layer27-k64-codebook65536", 27),
    "esmc_6b": ("biohub/ESMC-6B-sae-layer60-k64-codebook16384", 60),
}
ESMC_PANEL_KINDS = ("generated_kernel_boundary", "real_biological_holdout")
ESMC_REFERENCE_SOURCE_NAMES = ("biohub-esm", "biohub-transformers")
ESMC_BACKENDS = (
    "eager",
    "sdpa",
    "flex_attention",
    "flash_attention_2",
    "flash_attention_3",
)
ESMC_MEASURED_BACKENDS = ("eager", "sdpa", "flex_attention")
ESMC_UNAVAILABLE_BACKENDS = ("flash_attention_2", "flash_attention_3")
ESMC_REPORT_COUNT = len(ESMC_MODEL_IDS) * len(ESMC_BACKENDS) * len(ESMC_PANEL_KINDS)
ESMC_REPORT_MAX_BYTES = 16 * 1024 * 1024
ESMC_RELEASE_GATE_MODES = {
    "sdpa": "exact",
    "eager": "strict_numeric",
    "flex_attention": "diagnostic_with_catastrophe_gate",
}
ESMC_CATASTROPHE_UPPER = {
    "relative_l2": 0.25,
    "relative_q999": 0.50,
}
ESMC_CATASTROPHE_LOWER = {
    "residue_cosine_p01": 0.90,
    "pooled_cosine_min": 0.95,
}
ESMC_TOP_LEVEL_FIELDS = {
    "schema_version",
    "model_id",
    "candidate",
    "reference",
    "record_status",
    "unavailability",
    "configured_backend",
    "effective_backend",
    "dtype",
    "panel",
    "environment",
    "kernel",
    "panel_tensor_metrics",
    "panel_logits_metrics",
    "cases",
    "published_band_violations",
    "catastrophic_gate",
    "release_gate",
    "report_sha256",
}


CAPABILITY_EVIDENCE_SELECTORS: dict[str, EvidenceSelector] = {
    "cpu:autoclass-runtime": EvidenceSelector(
        tier="cpu_contract",
        targets=(
            "tests/cpu/test_autoclass_evidence_matrix.py::"
            "test_autoclass_runtime_evidence_matrix_exactly_matches_all_37_entries",
            "tests/cpu/test_autoclass_evidence_matrix.py::"
            "test_autoclass_runtime_evidence_targets_are_collected_cpu_tests",
        ),
        scope="Every family-level AutoClass entry and its explicit tiny runtime contracts.",
    ),
    "artifact:checkpoint-autoclasses": EvidenceSelector(
        tier="artifact",
        targets=(
            "tests/release/test_published_automodel.py::"
            "test_local_artifact_offline_autoclass_parity",
        ),
        scope="Every advertised AutoClass for every built checkpoint, grouped by checkpoint.",
    ),
    "compliance:sequence-primary-head": EvidenceSelector(
        tier="compliance",
        targets=(
            "tests/parity/test_native_results.py::test_native_exact_checkpoint_contract",
            "tests/parity/test_native_results.py::test_native_every_checkpoint_bf16_inference",
        ),
        scope=(
            "The official-parity head only: AutoModel for ANKH or a family without MaskedLM; "
            "otherwise AutoModelForMaskedLM."
        ),
    ),
    "compliance:ankh-seq2seq": EvidenceSelector(
        tier="compliance",
        targets=(
            "tests/parity/test_native_results.py::"
            "test_native_ankh_explicit_decoder_prompt_generation",
        ),
        scope="ANKH AutoModelForSeq2SeqLM explicit-prompt generation only.",
    ),
    "compliance:structure-automodel": EvidenceSelector(
        tier="compliance",
        targets=(
            "tests/structure/test_esmfold_folding_compliance.py",
            "tests/structure/test_esmfold2_folding_compliance.py",
        ),
        scope="ESMFold and ESMFold2 AutoModel folding paths only.",
    ),
    "benchmark:claim-eligible-primary-head": EvidenceSelector(
        tier="benchmark",
        targets=("benchmarks/suite.py::benchmark_cases[claim_eligible=True]",),
        scope=(
            "The benchmark-selected head for representative sequence checkpoints and "
            "ESMFold2 projection cases; startup and embedding cases are excluded."
        ),
    ),
    "cpu:attention-contracts": EvidenceSelector(
        tier="cpu_contract",
        targets=("tests/cpu/test_attention_contracts.py",),
        scope=(
            "Portable dispatch, masks, fallback, fake FA2/FA3, ESMC Flex/FA3, and "
            "eager/SDPA gradient contracts."
        ),
    ),
    "nightly:sequence-backends": EvidenceSelector(
        tier="nightly",
        targets=("tests/integration/test_backend_consistency.py",),
        scope=(
            "Current GH200 eager, SDPA, and Flex forward/backward paths. Flash kernels "
            "are not downloaded, built, or executed in the current locked environment."
        ),
    ),
    "historical:fa2-focused": EvidenceSelector(
        tier="historical",
        targets=("tools/remote/run.py::_kernel_capability_preflight",),
        scope=(
            "Policy records prior real FlashAttention 2 focused execution, but the immutable "
            "execution report is not bundled in this repository and no current GH200 "
            "numerical claim is inferred from it."
        ),
    ),
    "compliance:flash-unavailable-gh200": EvidenceSelector(
        tier="compliance",
        targets=(
            "tests/parity/test_native_results.py::"
            "test_esmc_bf16_calibration_and_biological_holdout",
        ),
        scope=(
            "Complete report-bound FA2/FA3 unavailability records and fail-closed "
            "dispatch on the frozen release environment."
        ),
    ),
    "compliance:deep-backends": EvidenceSelector(
        tier="compliance",
        targets=("tests/parity/test_native_results.py::test_native_representatives_all_backends",),
        scope="Every advertised backend on the pinned deep sequence representative per family.",
    ),
    "benchmark:claim-eligible-backends": EvidenceSelector(
        tier="benchmark",
        targets=("benchmarks/suite.py::benchmark_cases[claim_eligible=True]",),
        scope="Backends emitted by claim-eligible sequence and ESMFold2 benchmark cases.",
    ),
    "cpu:embedding-contracts": EvidenceSelector(
        tier="cpu_contract",
        targets=("tests/cpu/test_embedding_contracts.py",),
        scope="Ordered inputs, biological masking, pooling, streaming, and persistence.",
    ),
    "cpu:e1-embeddings": EvidenceSelector(
        tier="cpu_contract",
        targets=("tests/cpu/test_e1_contracts.py",),
        scope="E1 raw-sequence and MSA embedding persistence.",
    ),
    "feature:e1-rag": EvidenceSelector(
        tier="feature",
        targets=("tests/integration/test_e1_rag.py",),
        scope="E1 retrieval, MSA preparation, cache, scoring, and embedding flows.",
    ),
    "cpu:ankh-contracts": EvidenceSelector(
        tier="cpu_contract",
        targets=("tests/cpu/test_ankh_contracts.py",),
        scope="ANKH encoder and explicit-decoder embeddings, layers, masks, and T5 views.",
    ),
    "cpu:generation-contracts": EvidenceSelector(
        tier="cpu_contract",
        targets=("tests/cpu/test_generation_contracts.py",),
        scope="Tiny deterministic DPLM, DPLM2, and ESM3 generation contracts.",
    ),
    "feature:generation": EvidenceSelector(
        tier="feature",
        targets=(
            "tests/integration/test_dplm_generation.py",
            "tests/integration/test_esm3.py",
        ),
        scope="DPLM, DPLM2, and ESM3 generation behavior in the feature suite.",
    ),
    "cpu:peft": EvidenceSelector(
        tier="cpu_contract",
        targets=("tests/cpu/test_peft_contracts.py",),
        scope="Real initializer, collators, one optimizer step, and adapter/classifier reload.",
    ),
    "nightly:peft": EvidenceSelector(
        tier="nightly",
        targets=("tests/unit/test_fine_tuning_example.py",),
        scope="Fine-tuning example contracts in the nightly feature job.",
    ),
    "cpu:ttt": EvidenceSelector(
        tier="cpu_contract",
        targets=("tests/cpu/test_ttt_contracts.py",),
        scope="Seeded TTT initialization, update, reset, save, reload, and family isolation.",
    ),
    "feature:ttt": EvidenceSelector(
        tier="feature",
        targets=("tests/integration/test_ttt.py",),
        scope="TTT integration behavior in the feature suite.",
    ),
    "cpu:structure-contracts": EvidenceSelector(
        tier="cpu_contract",
        targets=("tests/cpu/test_structure_contracts.py",),
        scope="Tiny injected structure cores, public outputs, save/reload, and binder batching.",
    ),
    "structure:public-contracts": EvidenceSelector(
        tier="structure",
        targets=("tests/structure/test_structure_public_helpers.py",),
        scope="Seeded Boltz helper, linker masking, real features, losses, and binder gradients.",
    ),
    "structure:full-suite": EvidenceSelector(
        tier="structure",
        targets=("tests/structure",),
        scope="The declared GPU structure suite for folding and preparation behavior.",
    ),
    "feature:binder": EvidenceSelector(
        tier="feature",
        targets=("tests/integration/test_binder_design.py",),
        scope="Seeded binder workflow, atom padding, critic ranking, and traceability.",
    ),
    "cpu:artifact-example": EvidenceSelector(
        tier="cpu_contract",
        targets=(
            "tests/cpu/test_documentation_contracts.py::"
            "test_artifact_loading_example_executes_local_only_autoconfig",
        ),
        scope="The offline local-artifact example with AutoConfig.",
    ),
    "cpu:task-head-example": EvidenceSelector(
        tier="cpu_contract",
        targets=(
            "tests/cpu/test_documentation_contracts.py::"
            "test_task_head_example_executes_all_advertised_heads_offline",
        ),
        scope=(
            "Offline ESM2 masked-LM scoring, contacts, sequence classification, "
            "and token classification through the documented example."
        ),
    ),
}


EMBEDDING_CAPABILITY_ROWS = (
    CapabilityEvidenceRow(
        "Sequence list or streaming FASTA",
        "[embedding API](../embedding_api.md)",
        "[embedding and retrieval](../../examples/embedding_and_retrieval.py)",
        ("cpu:embedding-contracts",),
    ),
    CapabilityEvidenceRow(
        "Ordered mapping or one-shot generator",
        "[embedding API](../embedding_api.md)",
        "[runnable API contracts](../../tests/cpu/test_embedding_contracts.py)",
        ("cpu:embedding-contracts",),
    ),
    CapabilityEvidenceRow(
        "Biological-residue `max_length`, bounded token windows, and stable order",
        "[embedding API](../embedding_api.md#bounded-streaming-and-length-policy)",
        "[runnable API contracts](../../tests/cpu/test_embedding_contracts.py)",
        ("cpu:embedding-contracts",),
    ),
    CapabilityEvidenceRow(
        "Mean and standard-deviation pooling",
        "[embedding API](../embedding_api.md#pooling)",
        "[embedding and retrieval](../../examples/embedding_and_retrieval.py)",
        ("cpu:embedding-contracts",),
    ),
    CapabilityEvidenceRow(
        "Max/norm/median/variance/CLS/PARTI pooling",
        "[embedding API](../embedding_api.md#pooling)",
        "[runnable pooler contract](../../tests/cpu/test_embedding_contracts.py)",
        ("cpu:embedding-contracts",),
    ),
    CapabilityEvidenceRow(
        "Full-residue and all-selected-layer output",
        "[embedding API](../embedding_api.md#full-residue-embeddings)",
        "[ANKH layers](../../examples/ankh_embeddings.py)",
        ("cpu:embedding-contracts", "cpu:ankh-contracts"),
    ),
    CapabilityEvidenceRow(
        "Transactional sharded safetensors and exact resume",
        "[embedding API](../embedding_api.md#safetensors-storage)",
        "[embedding and retrieval](../../examples/embedding_and_retrieval.py)",
        ("cpu:embedding-contracts",),
    ),
    CapabilityEvidenceRow(
        "Read-only SQLite and ordered duplicate-preserving filters",
        "[embedding API](../embedding_api.md#sqlite-streaming-retrieval-and-resume)",
        "[embedding and retrieval](../../examples/embedding_and_retrieval.py)",
        ("cpu:embedding-contracts",),
    ),
    CapabilityEvidenceRow(
        "Legacy SQLite conversion without pickle deserialization",
        "[embedding API](../embedding_api.md#sqlite-streaming-retrieval-and-resume)",
        "[runnable converter contract](../../tests/cpu/test_embedding_contracts.py)",
        ("cpu:embedding-contracts",),
    ),
    CapabilityEvidenceRow(
        "E1 raw-sequence and MSA-aware ordered embeddings",
        "[E1 guide](../models.md#e1)",
        "[E1 RAG](../../examples/e1_rag.py)",
        ("cpu:e1-embeddings", "feature:e1-rag"),
    ),
    CapabilityEvidenceRow(
        "ANKH encoder/explicit-decoder hidden-state selection",
        "[ANKH guide](../models.md#ankh)",
        "[ANKH layers](../../examples/ankh_embeddings.py)",
        ("cpu:ankh-contracts",),
    ),
)


GENERATION_CAPABILITY_ROWS = (
    CapabilityEvidenceRow(
        "ESM2 pretrained masked-LM scoring and contact prediction",
        "[ESM2](../models.md#esm2)",
        "[task heads](../../examples/task_heads.py)",
        ("cpu:task-head-example", "cpu:autoclass-runtime"),
    ),
    CapabilityEvidenceRow(
        "ESM2 sequence/token classification with explicitly untrained task heads",
        "[ESM2](../models.md#esm2)",
        "[task heads](../../examples/task_heads.py)",
        ("cpu:task-head-example", "cpu:autoclass-runtime"),
    ),
    CapabilityEvidenceRow(
        "DPLM amino-acid diffusion generation",
        "[DPLM](../models.md#dplm)",
        "[generation](../../examples/generation.py)",
        ("cpu:generation-contracts", "feature:generation"),
    ),
    CapabilityEvidenceRow(
        "DPLM2 modality-aware sequence/structure co-generation",
        "[DPLM2](../models.md#dplm2)",
        "[generation](../../examples/generation.py)",
        ("cpu:generation-contracts", "feature:generation"),
    ),
    CapabilityEvidenceRow(
        "ESM3 multimodal-conditioned generation",
        "[ESM3](../models.md#esm3)",
        "[generation](../../examples/generation.py)",
        ("cpu:generation-contracts", "feature:generation"),
    ),
    CapabilityEvidenceRow(
        "ANKH task-prompted sequence-to-sequence generation",
        "[ANKH](../models.md#ankh)",
        "[ANKH embeddings and generation](../../examples/ankh_embeddings.py)",
        ("cpu:ankh-contracts", "compliance:ankh-seq2seq"),
    ),
    CapabilityEvidenceRow(
        "Trainer/PEFT LoRA with immutable inputs and verified save/reload",
        "[fine-tuning](../finetuning.md)",
        "[fine-tuning](../../examples/fine_tuning.py)",
        ("cpu:peft", "nightly:peft"),
    ),
    CapabilityEvidenceRow(
        "Seeded TTT adapter initialize/update/reset/save/reload",
        "[TTT](../ttt.md)",
        "[TTT](../../examples/ttt.py)",
        ("cpu:ttt", "feature:ttt"),
    ),
)


STRUCTURE_CAPABILITY_ROWS = (
    CapabilityEvidenceRow(
        "ESMFold single-chain folding and multimer-linker confidence masking",
        "[models](../models.md#esmfold)",
        "[structure preparation](../../examples/structure_preparation.py)",
        (
            "cpu:structure-contracts",
            "structure:public-contracts",
            "structure:full-suite",
            "compliance:structure-automodel",
        ),
    ),
    CapabilityEvidenceRow(
        "Seed-scoped Boltz2 protein helper and BF16 execution policy",
        "[Boltz2](../models.md#boltz2)",
        "[structure preparation](../../examples/structure_preparation.py)",
        (
            "cpu:structure-contracts",
            "structure:public-contracts",
            "structure:full-suite",
        ),
    ),
    CapabilityEvidenceRow(
        "Atom-dense binder optimization and critic reporting",
        "[binder design](../binder_design.md)",
        "[binder design](../../examples/binder_design_fastplms.py)",
        (
            "cpu:structure-contracts",
            "structure:public-contracts",
            "feature:binder",
        ),
    ),
    CapabilityEvidenceRow(
        "Offline local artifact AutoClass loading",
        "[artifacts](../artifacts.md)",
        "[artifact loading](../../examples/artifact_loading.py)",
        ("cpu:artifact-example", "artifact:checkpoint-autoclasses"),
    ),
)


def _esmfold2_structure_capability_rows(
    registry: ModelRegistry,
) -> tuple[CapabilityEvidenceRow, ...]:
    rows: list[CapabilityEvidenceRow] = []
    for spec in registry.by_family("esmfold2"):
        if spec.msa_conditioning is None:
            raise ValueError(f"{spec.id}: ESMFold2 MSA conditioning is undeclared")
        if spec.msa_conditioning:
            capability = (
                f"`{spec.id}` 48-block full ESMFold2: single-sequence or optional "
                "MSA-conditioned protein inputs, typed complexes, ligands, nucleic acids, "
                "modifications, bonds, and distograms; pocket requests fail closed"
            )
        else:
            capability = (
                f"`{spec.id}` 24-block Fast ESMFold2: inference-optimized "
                "single-sequence conditioning with typed multichain and multimolecule "
                "inputs; every protein must have `msa=None` and MSA inputs fail closed"
            )
        rows.append(
            CapabilityEvidenceRow(
                capability,
                "[ESMFold2](../esmfold2.md)",
                "[structure preparation](../../examples/structure_preparation.py)",
                (
                    "cpu:structure-contracts",
                    "structure:full-suite",
                    "compliance:structure-automodel",
                ),
            )
        )
    return tuple(rows)


CURATED_EXAMPLE_CPU_CASES: dict[str, tuple[str, ...]] = {
    "embedding_and_retrieval.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_embedding_and_retrieval_example_executes_with_ordered_sqlite",
    ),
    "attention_switching.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_attention_switching_main_executes_optimized_and_masked_fallback",
    ),
    "ankh_embeddings.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_ankh_embedding_example_executes_encoder_and_decoder_layers",
    ),
    "generation.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_generation_example_executes_seeded_dplm_branch_offline",
        "tests/cpu/test_documentation_contracts.py::"
        "test_generation_example_executes_seeded_dplm2_branch_offline",
        "tests/cpu/test_documentation_contracts.py::"
        "test_generation_example_executes_seeded_esm3_trace",
    ),
    "e1_rag.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_e1_rag_example_executes_local_msa_and_shared_persistence",
    ),
    "ttt.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_ttt_example_executes_seeded_adapt_save_and_reset",
    ),
    "structure_preparation.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_structure_preparation_example_executes_each_public_branch",
    ),
    "artifact_loading.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_artifact_loading_example_executes_local_only_autoconfig",
    ),
    "task_heads.py": (
        "tests/cpu/test_documentation_contracts.py::"
        "test_task_head_example_executes_all_advertised_heads_offline",
    ),
    "fine_tuning.py": (
        "tests/cpu/test_peft_contracts.py::"
        "test_fine_tuning_main_wires_both_tasks_without_external_io",
        "tests/cpu/test_peft_contracts.py::"
        "test_shipped_collators_create_tokenizer_aware_sequence_and_pair_batches",
        "tests/cpu/test_peft_contracts.py::"
        "test_shipped_initializer_drives_one_peft_step_and_atomic_final_reload",
    ),
    "binder_design_fastplms.py": (
        "tests/cpu/test_structure_contracts.py::"
        "test_public_binder_workflow_pads_heterogeneous_prepared_atoms_without_truncation",
        "tests/cpu/test_structure_contracts.py::"
        "test_binder_example_main_wires_explicit_offline_cli_arguments",
        "tests/cpu/test_structure_contracts.py::"
        "test_binder_structure_loss_is_finite_and_differentiable",
    ),
}


def _code(values: Iterable[str]) -> str:
    return ", ".join(f"`{value}`" for value in values)


def _table_row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _append_rows(lines: list[str], rows: Iterable[tuple[str, ...]]) -> None:
    lines.extend(_table_row(*row) for row in rows)


def _render_evidence_keys(keys: Iterable[str]) -> str:
    values = tuple(keys)
    missing = sorted(set(values).difference(CAPABILITY_EVIDENCE_SELECTORS))
    if missing:
        raise ValueError("Unknown capability evidence selectors: " + ", ".join(missing))
    if not values:
        raise ValueError("Every advertised capability requires at least one evidence selector.")
    return _code(values)


def _append_capability_rows(
    lines: list[str],
    rows: Iterable[CapabilityEvidenceRow],
) -> None:
    for row in rows:
        lines.append(
            _table_row(
                row.capability,
                row.guide,
                row.example,
                _render_evidence_keys(row.evidence),
            )
        )


def _primary_sequence_auto_class(family: ModelFamily) -> str:
    advertised = set(family.auto_map)
    if family.id == "ankh" or "AutoModelForMaskedLM" not in advertised:
        selected = "AutoModel"
    else:
        selected = "AutoModelForMaskedLM"
    if selected not in advertised:
        raise ValueError(f"{family.id} does not advertise required primary class {selected}.")
    return selected


def benchmark_autoclass_evidence_pairs(
    registry: ModelRegistry,
) -> frozenset[tuple[str, str]]:
    """Return only family/AutoClass pairs emitted by claim-eligible benchmarks."""

    pairs: set[tuple[str, str]] = set()
    for spec in registry.values():
        family = spec.family
        if "benchmark" not in family.test_tiers:
            continue
        if not (spec.is_deep_reference or family.id == "esmfold2"):
            continue
        if family.tokenizer_mode == "structure" and family.id != "esmfold2":
            continue
        pairs.add((family.id, _primary_sequence_auto_class(family)))
    return frozenset(pairs)


def benchmark_backend_evidence(registry: ModelRegistry) -> frozenset[str]:
    """Return only backends emitted by claim-eligible benchmark cases."""

    from benchmarks.suite import benchmark_cases

    backends = frozenset(
        str(case.backend)
        for case in benchmark_cases(
            family=None,
            quick=False,
            local_files_only=True,
        )
        if case.claim_eligible
    )
    advertised = {backend for family in registry.families.values() for backend in family.attention}
    unexpected = sorted(backends.difference(advertised))
    if unexpected:
        raise ValueError(
            "Claim-eligible benchmark cases advertise unknown backends: " + ", ".join(unexpected)
        )
    return backends


def autoclass_evidence_keys(
    registry: ModelRegistry,
    family_id: str,
    auto_class: str,
) -> tuple[str, ...]:
    """Map one advertised AutoClass to its actually executable evidence."""

    family = registry.families[family_id]
    if auto_class not in family.auto_map:
        raise ValueError(f"{family_id} does not advertise {auto_class}.")

    evidence = ["cpu:autoclass-runtime", "artifact:checkpoint-autoclasses"]
    if family.id == "esm2" and auto_class in {
        "AutoModelForMaskedLM",
        "AutoModelForSequenceClassification",
        "AutoModelForTokenClassification",
    }:
        evidence.append("cpu:task-head-example")
    if family.tokenizer_mode != "structure":
        if auto_class == _primary_sequence_auto_class(family):
            evidence.append("compliance:sequence-primary-head")
        if family.id == "ankh" and auto_class == "AutoModelForSeq2SeqLM":
            evidence.append("compliance:ankh-seq2seq")
    elif family.id in {"esmfold", "esmfold2"} and auto_class == "AutoModel":
        evidence.append("compliance:structure-automodel")

    if (family_id, auto_class) in benchmark_autoclass_evidence_pairs(registry):
        evidence.append("benchmark:claim-eligible-primary-head")
    return tuple(evidence)


def _autoclass_workflow_example(family: ModelFamily, auto_class: str) -> str:
    if family.id == "esm2" and auto_class in {
        "AutoModelForMaskedLM",
        "AutoModelForSequenceClassification",
        "AutoModelForTokenClassification",
    }:
        return "../../examples/task_heads.py"
    return FAMILY_DOCUMENTATION[family.id][1]


def attention_backend_evidence_keys(
    registry: ModelRegistry,
    backend: str,
) -> tuple[str, ...]:
    """Map an advertised backend to scoped CPU, GPU, parity, and benchmark evidence."""

    advertising_families = tuple(
        family for family in registry.families.values() if backend in family.attention
    )
    if not advertising_families:
        raise ValueError(f"No family advertises attention backend {backend!r}.")

    evidence = ["cpu:attention-contracts"]
    sequence_families = tuple(
        family for family in advertising_families if family.tokenizer_mode != "structure"
    )
    if sequence_families and backend in ESMC_MEASURED_BACKENDS:
        evidence.append("nightly:sequence-backends")
    if backend == "flash_attention_2":
        evidence.append("historical:fa2-focused")
        evidence.append("compliance:flash-unavailable-gh200")
    elif backend == "flash_attention_3":
        evidence.append("compliance:flash-unavailable-gh200")
    elif any(
        spec.is_deep_reference
        and spec.family.tokenizer_mode != "structure"
        and backend in spec.family.attention
        for spec in registry.values()
    ):
        evidence.append("compliance:deep-backends")
    if backend in ESMC_MEASURED_BACKENDS and backend in benchmark_backend_evidence(registry):
        evidence.append("benchmark:claim-eligible-backends")
    return tuple(evidence)


def _render_evidence_selector_catalog() -> list[str]:
    lines = [
        "## Executable evidence selectors",
        "",
        "Only the selectors below are claimed. Their scopes are intentionally narrower",
        "than a whole family or validation tier. A tier appearing on another row does not",
        "automatically apply to the capability in this row.",
        "",
        _table_row("Selector", "Tier/job", "Executable target", "Scope"),
        _table_row("---", "---", "---", "---"),
    ]
    for key, selector in CAPABILITY_EVIDENCE_SELECTORS.items():
        targets = "<br>".join(f"`{target}`" for target in selector.targets)
        lines.append(_table_row(f"`{key}`", f"`{selector.tier}`", targets, selector.scope))
    lines.append("")
    return lines


def _render_curated_example_cpu_evidence() -> list[str]:
    lines = [
        "## Curated offline example execution",
        "",
        "Every curated example is routed to the exact collected CPU test nodes below.",
        "These tests run under the required offline `cpu_contract` gate.",
        "",
        _table_row("Example", "Tier", "Exact executable CPU node"),
        _table_row("---", "---", "---"),
    ]
    for example_name, nodeids in CURATED_EXAMPLE_CPU_CASES.items():
        for nodeid in nodeids:
            lines.append(
                _table_row(
                    f"[`{example_name}`](../../examples/{example_name})",
                    "`cpu_contract`",
                    f"`{nodeid}`",
                )
            )
    lines.append("")
    return lines


def _precision_contract(family: ModelFamily) -> str:
    experimental = set(family.experimental_precisions)
    return ", ".join(
        f"`{value}` (experimental)" if value in experimental else f"`{value}`"
        for value in family.precisions
    )


def _hub_license_label(family: ModelFamily) -> str:
    label = f"`{family.hub_license}`"
    if family.hub_license == "other":
        label += f" ({render_checkpoint_terms(family)})"
    return label


def _tokenizer_class_label(family: ModelFamily) -> str:
    if family.tokenizer_class is None:
        return "`n/a`"
    return f"`{family.tokenizer_class}`"


def _auto_class_status(family: ModelFamily, auto_class: str) -> str:
    """Describe whether an advertised entry point has trained checkpoint state."""

    if family.id == "ankh" and auto_class == "AutoModelForMaskedLM":
        return "FastPLMs extension"
    try:
        return AUTO_CLASS_STATUS[auto_class]
    except KeyError as error:
        raise ValueError(f"No model-card weight status is defined for {auto_class!r}.") from error


def _platform_requirements(family: ModelFamily) -> str:
    requirements = ["Python 3.11-3.14, PyTorch 2.13, and Transformers 5.13 are required."]
    if family.tokenizer_mode == "structure":
        requirements.append(
            "The artifact requirements include the direct structure dependencies. "
            "The published execution contract requires a CUDA device. The current "
            "validated release target is the exact NVIDIA GH200 on Linux aarch64; "
            "Linux x86-64, CPU-only, Windows, and macOS structure runs are not "
            "current release evidence."
        )
    elif any(name.startswith("flash_attention_") for name in family.attention):
        requirements.append(
            "The artifact requirements include the direct FlashAttention loader "
            "dependency. FlashAttention also requires compatible CUDA hardware and "
            "BF16 execution."
        )
    else:
        requirements.append(
            "The declared CPU gate covers tiny offline contracts; published "
            "checkpoint throughput and parity require the documented device tier."
        )
    return " ".join(requirements)


def _installation_section(spec: ModelSpec) -> str:
    return f"""\
## Install and platform requirements

Install the direct dependencies published with this model:

```bash
python -m pip install -r \\
  "https://huggingface.co/{spec.fast.repo_id}/resolve/main/requirements.txt"
```

The FastPLMs implementation itself is embedded in the model repository and loaded
by Transformers through `trust_remote_code=True`.

{_platform_requirements(spec.family)} The Hub quick start below requires network
access on first download. For an air-gapped run, first build the manifest-pinned
local artifact and use the offline form shown in the example.

"""


def _esmc_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EsmcReportError(f"ESMC JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _esmc_reject_json_constant(value: str) -> object:
    raise EsmcReportError(f"ESMC JSON contains non-finite constant {value!r}")


def _esmc_decode_json(encoded: str, *, context: str) -> dict[str, object]:
    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=_esmc_json_object,
            parse_constant=_esmc_reject_json_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise EsmcReportError(f"{context} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise EsmcReportError(f"{context} must contain one JSON object")
    return payload


def _esmc_read_json(path: Path) -> dict[str, object]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise EsmcReportError(f"Unable to stat ESMC evidence file {path}: {error}") from error
    if size <= 0 or size > ESMC_REPORT_MAX_BYTES:
        raise EsmcReportError(
            f"ESMC evidence file {path.name!r} has invalid size {size}; "
            f"maximum is {ESMC_REPORT_MAX_BYTES} bytes"
        )
    try:
        encoded = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EsmcReportError(f"Unable to read ESMC evidence file {path}: {error}") from error
    return _esmc_decode_json(encoded, context=f"ESMC evidence file {path.name!r}")


def _esmc_require_mapping(
    value: object,
    fields: set[str],
    *,
    context: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise EsmcReportError(f"{context} fields differ from schema v3")
    return value


def _esmc_require_object(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EsmcReportError(f"{context} must be a JSON object")
    return value


def _esmc_require_list(value: object, *, context: str) -> list[object]:
    if not isinstance(value, list):
        raise EsmcReportError(f"{context} must be a JSON array")
    return value


def _esmc_require_text(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EsmcReportError(f"{context} must be a nonempty string")
    return value


def _esmc_require_sha256(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EsmcReportError(f"{context} must be a canonical lowercase SHA-256 digest")
    return value


def _esmc_require_finite(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EsmcReportError(f"{context} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise EsmcReportError(f"{context} must be a finite number")
    return numeric


def _esmc_require_gpu_capability(
    value: object,
    *,
    context: str,
) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise EsmcReportError(f"{context} must contain exactly two integers")
    major, minor = value
    if (
        isinstance(major, bool)
        or not isinstance(major, int)
        or major < 0
        or isinstance(minor, bool)
        or not isinstance(minor, int)
        or minor < 0
    ):
        raise EsmcReportError(f"{context} must contain exactly two non-negative integers")
    return major, minor


def _esmc_report_sha256(payload: Mapping[str, object]) -> str:
    digest_payload = dict(payload)
    digest_payload.pop("report_sha256", None)
    encoded = json.dumps(
        digest_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _esmc_public_case(case: Mapping[str, object]) -> dict[str, object]:
    return {
        "case_id": case["case_id"],
        "sequence_length": case["sequence_length"],
        "sequence_sha256": case["sequence_sha256"],
        "source": case.get("source"),
        "source_sha256": case.get("source_sha256"),
    }


def _esmc_panel_identity(kind: str, cases: list[dict[str, object]]) -> dict[str, object]:
    definition = {
        "schema_version": 1,
        "kind": kind,
        "seed": 42,
        "cases": cases,
    }
    definition_sha256 = hashlib.sha256(
        json.dumps(definition, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "kind": kind,
        "seed": 42,
        "definition_sha256": definition_sha256,
        "cases": [_esmc_public_case(case) for case in cases],
    }


def _expected_esmc_panels(source_root: Path) -> dict[str, dict[str, object]]:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    generator = random.Random(42)
    generated_cases: list[dict[str, object]] = []
    for length in (13, 15, 16, 17, 29, 31, 32, 33, 61, 127, 128, 129):
        sequence = "M" + "".join(generator.choices(alphabet, k=length - 1))
        generated_cases.append(
            {
                "case_id": f"generated-boundary-{length}",
                "sequence": sequence,
                "sequence_length": length,
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            }
        )

    fixture_path = source_root / "tests" / "parity" / "fixtures" / "esmc_biological_holdout.json"
    try:
        fixture_text = fixture_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EsmcReportError(f"Unable to read immutable ESMC panel fixture: {error}") from error
    fixture = _esmc_decode_json(fixture_text, context="ESMC biological holdout fixture")
    if set(fixture) != {"schema_version", "cases"} or fixture["schema_version"] != 1:
        raise EsmcReportError("ESMC biological holdout fixture fields differ from schema v1")
    raw_cases = fixture["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EsmcReportError("ESMC biological holdout fixture has no ordered cases")
    biological_cases: list[dict[str, object]] = []
    for index, raw_case in enumerate(raw_cases):
        case = _esmc_require_mapping(
            raw_case,
            {"case_id", "sequence", "sequence_sha256", "source", "source_sha256"},
            context=f"ESMC biological holdout case {index}",
        )
        sequence = _esmc_require_text(
            case["sequence"], context=f"ESMC biological holdout case {index} sequence"
        )
        if not sequence.isupper() or not set(sequence).issubset(set(alphabet)):
            raise EsmcReportError(
                f"ESMC biological holdout case {index} is not canonical uppercase protein"
            )
        sequence_sha256 = _esmc_require_sha256(
            case["sequence_sha256"], context=f"ESMC biological holdout case {index} sequence"
        )
        if sequence_sha256 != hashlib.sha256(sequence.encode("ascii")).hexdigest():
            raise EsmcReportError(f"ESMC biological holdout case {index} sequence digest drifted")
        _esmc_require_sha256(
            case["source_sha256"], context=f"ESMC biological holdout case {index} source"
        )
        biological_cases.append({**case, "sequence_length": len(sequence)})

    return {
        "generated_kernel_boundary": _esmc_panel_identity(
            "generated_kernel_boundary", generated_cases
        ),
        "real_biological_holdout": _esmc_panel_identity(
            "real_biological_holdout", biological_cases
        ),
    }


def _expected_biohub_source_contracts(
    source_root: Path,
) -> dict[str, dict[str, object]]:
    expected_fields = {
        "import_name",
        "import_root",
        "package_version",
        "schema_version",
        "source_revision",
        "tree_sha256",
    }
    file_names = {
        "biohub-esm": "biohub-esm-source.json",
        "biohub-transformers": "biohub-transformers-source.json",
    }
    contracts: dict[str, dict[str, object]] = {}
    for source_name in ESMC_REFERENCE_SOURCE_NAMES:
        path = source_root / "docker" / "constraints" / file_names[source_name]
        try:
            encoded = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise EsmcReportError(
                f"Unable to read pinned {source_name} source contract: {error}"
            ) from error
        contract = _esmc_decode_json(encoded, context=f"{source_name} source contract")
        if set(contract) != expected_fields or contract["schema_version"] != 1:
            raise EsmcReportError(f"{source_name} source contract differs from schema v1")
        _esmc_require_text(contract["import_name"], context=f"{source_name} import name")
        _esmc_require_text(contract["import_root"], context=f"{source_name} import root")
        _esmc_require_text(contract["package_version"], context=f"{source_name} package version")
        revision = contract["source_revision"]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or revision != revision.lower()
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise EsmcReportError(
                f"{source_name} source revision is not canonical lowercase 40-hex"
            )
        _esmc_require_sha256(contract["tree_sha256"], context=f"{source_name} tree")
        contracts[source_name] = contract
    return contracts


def _validate_esmc_reference_sources(
    value: object,
    expected_contracts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    sources = _esmc_require_mapping(
        value,
        set(ESMC_REFERENCE_SOURCE_NAMES),
        context="ESMC reference sources",
    )
    source_fields = {
        "schema_version",
        "source_revision",
        "tree_sha256",
        "attestation_sha256",
        "file_count",
        "import_name",
        "import_root",
        "import_file",
        "package_version",
    }
    for source_name in ESMC_REFERENCE_SOURCE_NAMES:
        source = _esmc_require_mapping(
            sources[source_name],
            source_fields,
            context=f"ESMC reference source {source_name}",
        )
        expected_contract = expected_contracts[source_name]
        if source["schema_version"] != 1:
            raise EsmcReportError(f"ESMC reference source {source_name} schema is unsupported")
        for name in (
            "source_revision",
            "tree_sha256",
            "import_name",
            "import_root",
            "package_version",
        ):
            if source[name] != expected_contract[name]:
                raise EsmcReportError(
                    f"ESMC reference source {source_name} {name} differs from the pin"
                )
        _esmc_require_sha256(
            source["attestation_sha256"],
            context=f"ESMC reference source {source_name} attestation",
        )
        file_count = source["file_count"]
        if isinstance(file_count, bool) or not isinstance(file_count, int) or file_count <= 0:
            raise EsmcReportError(f"ESMC reference source {source_name} file count is invalid")
        expected_import_file = f"{expected_contract['import_root']}/__init__.py"
        if source["import_file"] != expected_import_file:
            raise EsmcReportError(
                f"ESMC reference source {source_name} import file differs from its root"
            )
    return sources


def _esmc_runtime_identity_from_source(
    source_root: Path,
    registry: ModelRegistry,
) -> EsmcRuntimeIdentity:
    try:
        from tools.artifacts.build import (
            _render_runtime_bundle,
            _validated_runtime_snapshot,
            _write_runtime_snapshot,
        )

        identities: set[tuple[str, str, str]] = set()
        with tempfile.TemporaryDirectory(prefix="fastplms-esmc-runtime-") as directory:
            temporary_root = Path(directory)
            for spec in (registry[model_id] for model_id in ESMC_MODEL_IDS):
                runtime_revision, payloads, source_tree_sha256 = _validated_runtime_snapshot(
                    source_root,
                    registry,
                    spec,
                )
                package_root = temporary_root / spec.id / "fastplms"
                _write_runtime_snapshot(package_root, payloads)
                runtime_bundle_sha256, _ = _render_runtime_bundle(package_root)
                identities.add((runtime_revision, source_tree_sha256, runtime_bundle_sha256))
    except Exception as error:
        raise EsmcReportError(
            "Unable to derive the clean tracked ESMC runtime identity required for release "
            f"evidence: {error}"
        ) from error
    if len(identities) != 1:
        raise EsmcReportError(
            "ESMC checkpoints do not resolve to one shared runtime/source/bundle identity"
        )
    runtime_revision, source_tree_sha256, runtime_bundle_sha256 = identities.pop()
    return EsmcRuntimeIdentity(
        runtime_revision=runtime_revision,
        source_tree_sha256=source_tree_sha256,
        runtime_bundle_sha256=runtime_bundle_sha256,
    )


def _validate_esmc_runtime_identity(identity: EsmcRuntimeIdentity) -> None:
    source_digest = _esmc_require_sha256(
        identity.source_tree_sha256, context="ESMC expected source tree"
    )
    _esmc_require_sha256(identity.runtime_bundle_sha256, context="ESMC expected runtime bundle")
    revision = identity.runtime_revision
    is_git_revision = (
        isinstance(revision, str)
        and len(revision) == 40
        and revision == revision.lower()
        and all(character in "0123456789abcdef" for character in revision)
    )
    if not is_git_revision and revision != f"source-tree-sha256:{source_digest}":
        raise EsmcReportError(
            "ESMC runtime revision must be a clean 40-hex Git revision or the exact "
            "source-tree-sha256 fallback"
        )


def _esmc_runtime_platform_identity(
    reference_environment: Mapping[str, object],
) -> tuple[str, str]:
    runtime = _esmc_require_object(
        reference_environment.get("runtime"),
        context="ESMC locked reference runtime",
    )
    operating_system = _esmc_require_text(
        runtime.get("operating_system"), context="ESMC locked operating system"
    )
    architecture = _esmc_require_text(
        runtime.get("architecture"), context="ESMC locked architecture"
    )
    gpu = _esmc_require_object(runtime.get("gpu"), context="ESMC locked GPU identity")
    gpu_name = _esmc_require_text(gpu.get("name"), context="ESMC locked GPU name")
    capability = gpu.get("capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in capability
        )
    ):
        raise EsmcReportError("ESMC locked GPU capability is malformed")
    return (
        f"{operating_system.lower()}/{architecture.lower()}",
        f"{gpu_name}/SM{capability[0]}{capability[1]}",
    )


def _esmc_unavailability_identity(
    backend: str,
    reference_environment: Mapping[str, object],
) -> dict[str, str]:
    platform_identity, accelerator_identity = _esmc_runtime_platform_identity(reference_environment)
    if backend == "flash_attention_2":
        historical_evidence = "separate_historical_focused_evidence_only"
        reason = (
            f"The locked {platform_identity} {accelerator_identity} release environment "
            "has no validated FlashAttention 2 "
            "kernel. Prior focused execution evidence is historical and is not part of "
            "the current ESMC release distribution."
        )
    elif backend == "flash_attention_3":
        historical_evidence = "none"
        reason = (
            "The manifest-pinned FlashAttention 3 kernel has no validated artifact for "
            f"the locked {platform_identity} {accelerator_identity} release environment."
        )
    else:
        raise EsmcReportError(f"ESMC backend {backend!r} is not a structured unavailable backend")
    return {
        "code": "locked_platform_kernel_unavailable",
        "platform": platform_identity,
        "accelerator": accelerator_identity,
        "dispatch_contract": "fail_closed_without_dispatch",
        "historical_evidence": historical_evidence,
        "reason": reason,
    }


def _validate_esmc_candidate_environment(value: object) -> dict[str, object]:
    environment = _esmc_require_mapping(
        value,
        {
            "python",
            "torch",
            "transformers",
            "cuda_runtime",
            "cuda_driver",
            "gpu",
            "packages",
        },
        context="ESMC candidate environment",
    )
    for name in ("python", "torch", "transformers", "cuda_runtime", "cuda_driver"):
        _esmc_require_text(environment[name], context=f"ESMC candidate environment {name}")
    packages = _esmc_require_mapping(
        environment["packages"],
        {
            "fastplms",
            "huggingface-hub",
            "kernels",
            "tokenizers",
            "transformer-engine",
            "transformer-engine-torch",
        },
        context="ESMC candidate package inventory",
    )
    for name, version in packages.items():
        if version is not None:
            _esmc_require_text(version, context=f"ESMC candidate package {name}")
    for required in ("fastplms", "huggingface-hub", "kernels", "tokenizers"):
        if packages[required] is None:
            raise EsmcReportError(f"ESMC candidate package {required!r} is unavailable")
    gpu = _esmc_require_mapping(
        environment["gpu"],
        {"name", "capability", "total_memory_bytes"},
        context="ESMC candidate GPU identity",
    )
    _esmc_require_text(gpu["name"], context="ESMC candidate GPU name")
    capability = gpu["capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in capability
        )
    ):
        raise EsmcReportError("ESMC candidate GPU capability is invalid")
    memory = gpu["total_memory_bytes"]
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
        raise EsmcReportError("ESMC candidate GPU memory identity is invalid")
    return environment


def _validate_esmc_reference_environment(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EsmcReportError("ESMC reference environment is missing")
    required = {
        "cuda_device",
        "cuda_device_capability",
        "cuda_total_memory",
        "cuda_runtime",
        "packages",
        "python",
        "torch",
    }
    if not required.issubset(value):
        raise EsmcReportError("ESMC reference environment fields are incomplete")
    for name in ("cuda_device", "cuda_runtime", "python", "torch"):
        _esmc_require_text(value[name], context=f"ESMC reference environment {name}")
    _esmc_require_text(value["cuda_device"], context="ESMC reference environment CUDA device")
    capability = value["cuda_device_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in capability
        )
    ):
        raise EsmcReportError("ESMC reference CUDA capability is invalid")
    memory = value["cuda_total_memory"]
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
        raise EsmcReportError("ESMC reference GPU memory identity is invalid")
    packages_text = _esmc_require_text(
        value["packages"], context="ESMC reference package inventory"
    )
    packages = _esmc_decode_json(packages_text, context="ESMC reference package inventory")
    if not isinstance(packages, dict):
        raise EsmcReportError("ESMC reference package inventory is not an object")
    return value


def _validate_locked_esmc_reference_environment(
    value: object,
    *,
    source_root: Path,
) -> dict[str, object]:
    try:
        return validate_biohub_reference_environment_evidence(
            value,
            repository_root=source_root,
            contract_path=source_root / "docker/constraints/biohub-reference-lock.json",
        )
    except BiohubReferenceEnvironmentError as error:
        raise EsmcReportError(f"ESMC locked reference environment is invalid: {error}") from error


def _validate_esmc_environment_binding(
    candidate_environment: Mapping[str, object],
    dynamic_reference_environment: Mapping[str, object],
    locked_reference_environment: Mapping[str, object],
) -> None:
    candidate_gpu = _esmc_require_object(
        candidate_environment.get("gpu"), context="ESMC candidate GPU binding"
    )
    locked_runtime = _esmc_require_object(
        locked_reference_environment.get("runtime"),
        context="ESMC locked reference runtime binding",
    )
    locked_gpu = _esmc_require_object(
        locked_runtime.get("gpu"), context="ESMC locked reference GPU binding"
    )
    candidate_identity = {
        "python": candidate_environment.get("python"),
        "torch": candidate_environment.get("torch"),
        "cuda_runtime": candidate_environment.get("cuda_runtime"),
        "cuda_driver": candidate_environment.get("cuda_driver"),
        "gpu": dict(candidate_gpu),
    }
    dynamic_reference_identity = {
        "python": dynamic_reference_environment.get("python"),
        "torch": dynamic_reference_environment.get("torch"),
        "cuda_runtime": dynamic_reference_environment.get("cuda_runtime"),
        "cuda_driver": candidate_environment.get("cuda_driver"),
        "gpu": {
            "name": dynamic_reference_environment.get("cuda_device"),
            "capability": dynamic_reference_environment.get("cuda_device_capability"),
            "total_memory_bytes": dynamic_reference_environment.get("cuda_total_memory"),
        },
    }
    locked_identity = {
        "python": locked_runtime.get("python_version"),
        "torch": locked_runtime.get("torch"),
        "cuda_runtime": locked_runtime.get("cuda_runtime"),
        "cuda_driver": locked_runtime.get("cuda_driver"),
        "gpu": dict(locked_gpu),
    }
    if candidate_identity != dynamic_reference_identity:
        raise EsmcReportError("ESMC candidate and native reference environments differ")
    if candidate_identity != locked_identity:
        raise EsmcReportError(
            "ESMC candidate environment differs from the locked reference runtime"
        )


def _validate_esmc_kernel(
    value: object,
    backend: str,
    environment: Mapping[str, object],
    registry: ModelRegistry,
) -> None:
    kernel_spec = registry.attention_kernels.get(backend)
    if kernel_spec is None:
        expected = {
            "implementation": backend,
            "provider": "torch",
            "torch_version": environment["torch"],
        }
    else:
        packages = _esmc_require_object(
            environment["packages"], context="ESMC candidate package inventory"
        )
        expected = {
            "implementation": backend,
            "provider": "huggingface_kernels",
            "repository": kernel_spec.repository,
            "revision": kernel_spec.revision,
            "version": kernel_spec.version,
            "expected_variant": kernel_spec.expected_variant,
            "supported_dtypes": list(kernel_spec.dtypes),
            "kernels_package_version": packages["kernels"],
        }
    if value != expected:
        raise EsmcReportError(
            f"ESMC {backend} kernel identity differs from the manifest/runtime contract"
        )


def _validate_esmc_logits_metrics(value: object, *, context: str) -> None:
    metrics = _esmc_require_mapping(
        value,
        {"confident_top1_agreement", "mean_jsd"},
        context=f"{context} logits metrics",
    )
    agreement = _esmc_require_finite(
        metrics["confident_top1_agreement"], context=f"{context} top-1 agreement"
    )
    mean_jsd = _esmc_require_finite(metrics["mean_jsd"], context=f"{context} mean JSD")
    if not 0.80 <= agreement <= 1.000001:
        raise EsmcReportError(f"{context} top-1 agreement fails the catastrophe gate")
    if not -1e-7 <= mean_jsd <= 0.05:
        raise EsmcReportError(f"{context} mean JSD fails the catastrophe gate")


def _validate_esmc_tensor_metrics(
    value: object,
    *,
    context: str,
    expected_metric_context: str,
) -> tuple[tuple[str, int | None], ...]:
    if not isinstance(value, list) or not value:
        raise EsmcReportError(f"{context} tensor metrics are missing")
    layout: list[tuple[str, int | None]] = []
    hidden_layers: list[int] = []
    output_counts = {"last_hidden_state": 0, "logits": 0}
    fields = {
        "context",
        "output",
        "layer_index",
        "relative_l2",
        "relative_q999",
        "residue_cosine_p01",
        "pooled_cosine_min",
    }
    for index, raw_metric in enumerate(value):
        metric = _esmc_require_mapping(
            raw_metric, fields, context=f"{context} tensor metric {index}"
        )
        if metric["context"] != expected_metric_context:
            raise EsmcReportError(f"{context} tensor metric context is stale or misaligned")
        output = metric["output"]
        layer_index = metric["layer_index"]
        if output == "hidden_state":
            if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
                raise EsmcReportError(f"{context} hidden-state layer index is invalid")
            hidden_layers.append(layer_index)
        elif output in output_counts:
            if layer_index is not None:
                raise EsmcReportError(f"{context} {output} layer index must be null")
            output_counts[output] += 1
        else:
            raise EsmcReportError(f"{context} tensor output {output!r} is unsupported")
        for metric_name, upper in ESMC_CATASTROPHE_UPPER.items():
            numeric = _esmc_require_finite(metric[metric_name], context=f"{context} {metric_name}")
            if not 0 <= numeric <= upper:
                raise EsmcReportError(f"{context} {metric_name} fails the catastrophe gate")
        for metric_name, lower in ESMC_CATASTROPHE_LOWER.items():
            numeric = _esmc_require_finite(metric[metric_name], context=f"{context} {metric_name}")
            if not lower <= numeric <= 1.000001:
                raise EsmcReportError(f"{context} {metric_name} fails the catastrophe gate")
        if not isinstance(output, str) or not (layer_index is None or isinstance(layer_index, int)):
            raise EsmcReportError(f"{context} tensor metric layout is invalid")
        layout.append((output, layer_index))
    if hidden_layers != list(range(len(hidden_layers))):
        raise EsmcReportError(f"{context} hidden-state layers are incomplete or unordered")
    if output_counts != {"last_hidden_state": 1, "logits": 1}:
        raise EsmcReportError(
            f"{context} must contain exactly one last-hidden-state and one logits metric"
        )
    return tuple(layout)


def _validate_esmc_report(
    payload: dict[str, object],
    *,
    spec: ModelSpec,
    backend: str,
    panel: str,
    expected_panel: Mapping[str, object],
    expected_reference_sources: Mapping[str, Mapping[str, object]],
    runtime_identity: EsmcRuntimeIdentity,
    registry: ModelRegistry,
    source_root: Path,
) -> None:
    if set(payload) != ESMC_TOP_LEVEL_FIELDS or (
        payload.get("schema_version") != ESMC_DIAGNOSTIC_SCHEMA_VERSION
    ):
        raise EsmcReportError("ESMC diagnostic fields differ from schema v3")
    report_sha256 = _esmc_require_sha256(
        payload["report_sha256"], context="ESMC report self-digest"
    )
    if report_sha256 != _esmc_report_sha256(payload):
        raise EsmcReportError("ESMC report self-digest does not match its canonical payload")
    if payload["model_id"] != spec.id or payload["dtype"] != "bfloat16":
        raise EsmcReportError("ESMC report model or dtype identity is stale")
    if payload["configured_backend"] != backend:
        raise EsmcReportError("ESMC configured backend identity is invalid")
    record_status = payload["record_status"]
    if record_status not in {"measured", "unavailable"}:
        raise EsmcReportError("ESMC record status is invalid")

    candidate = _esmc_require_mapping(
        payload["candidate"],
        {
            "repo_id",
            "manifest_revision",
            "resolved_commit",
            "checkpoint_repo_id",
            "checkpoint_revision",
            "weights_revision",
            "runtime_revision",
            "source_tree_sha256",
            "runtime_bundle_sha256",
        },
        context="ESMC candidate identity",
    )
    expected_candidate = {
        "repo_id": spec.fast.repo_id,
        "manifest_revision": spec.fast.revision,
        "checkpoint_repo_id": spec.artifact_checkpoint.repo_id,
        "checkpoint_revision": spec.artifact_checkpoint.revision,
        "weights_revision": spec.artifact_checkpoint.revision,
        "runtime_revision": runtime_identity.runtime_revision,
        "source_tree_sha256": runtime_identity.source_tree_sha256,
        "runtime_bundle_sha256": runtime_identity.runtime_bundle_sha256,
    }
    for name, expected in expected_candidate.items():
        if candidate[name] != expected:
            raise EsmcReportError(f"ESMC candidate {name} differs from frozen release identity")
    if candidate["resolved_commit"] not in {None, spec.fast.revision}:
        raise EsmcReportError("ESMC candidate resolved Hub commit is stale")

    reference = _esmc_require_mapping(
        payload["reference"],
        {
            "repo_id",
            "revision",
            "state_transform",
            "environment",
            "reference_environment",
            "reference_sources",
        },
        context="ESMC reference identity",
    )
    if (
        reference["repo_id"] != spec.official.repo_id
        or reference["revision"] != spec.official.revision
        or reference["state_transform"] != spec.family.state_transform
    ):
        raise EsmcReportError("ESMC reference identity differs from the pinned manifest")
    _validate_esmc_reference_sources(
        reference["reference_sources"],
        expected_reference_sources,
    )
    dynamic_reference_environment = _validate_esmc_reference_environment(reference["environment"])
    locked_reference_environment = _validate_locked_esmc_reference_environment(
        reference["reference_environment"], source_root=source_root
    )
    candidate_environment = _validate_esmc_candidate_environment(payload["environment"])
    _validate_esmc_environment_binding(
        candidate_environment,
        dynamic_reference_environment,
        locked_reference_environment,
    )
    _validate_esmc_kernel(payload["kernel"], backend, candidate_environment, registry)

    report_panel = payload["panel"]
    if report_panel != expected_panel:
        raise EsmcReportError(f"ESMC panel {panel!r} differs from its immutable definition")
    report_panel = _esmc_require_object(report_panel, context="ESMC panel identity")
    panel_cases = report_panel["cases"]
    cases = payload["cases"]
    if (
        not isinstance(panel_cases, list)
        or not isinstance(cases, list)
        or (len(cases) != len(panel_cases))
    ):
        raise EsmcReportError("ESMC panel and per-case metrics are not aligned")
    identity_fields = {
        "case_id",
        "sequence_length",
        "sequence_sha256",
        "source",
        "source_sha256",
    }
    violations = payload["published_band_violations"]
    if not isinstance(violations, list) or any(
        not isinstance(item, str) or not item.strip() for item in violations
    ):
        raise EsmcReportError("ESMC published-band violations must be a string list")
    release_gate = _esmc_require_mapping(
        payload["release_gate"], {"mode", "status"}, context="ESMC release gate"
    )
    if record_status == "unavailable":
        if backend not in ESMC_UNAVAILABLE_BACKENDS:
            raise EsmcReportError("Only locked Flash backends may be unavailable")
        if payload["effective_backend"] is not None:
            raise EsmcReportError("Unavailable ESMC records must not claim effective dispatch")
        if payload["unavailability"] != _esmc_unavailability_identity(
            backend, locked_reference_environment
        ):
            raise EsmcReportError("ESMC structured unavailability identity is invalid")
        if payload["catastrophic_gate"] != "not_run":
            raise EsmcReportError("Unavailable ESMC catastrophe gate must be not run")
        if release_gate != {"mode": "availability", "status": "unavailable"}:
            raise EsmcReportError("Unavailable ESMC release-gate identity is invalid")
        if (
            payload["panel_tensor_metrics"] is not None
            or payload["panel_logits_metrics"] is not None
            or violations
        ):
            raise EsmcReportError("Unavailable ESMC records must not contain measurements")
        if cases != panel_cases:
            raise EsmcReportError("Unavailable ESMC cases must be immutable panel identities only")
        return

    if backend not in ESMC_MEASURED_BACKENDS:
        raise EsmcReportError("Current frozen measurements are limited to eager, SDPA, and Flex")
    if payload["effective_backend"] != backend:
        raise EsmcReportError("ESMC effective backend identity is invalid or fell back")
    if payload["unavailability"] is not None:
        raise EsmcReportError("Measured ESMC records carry unavailability metadata")
    if payload["catastrophic_gate"] != "passed":
        raise EsmcReportError("Measured ESMC report catastrophe gate did not pass")
    if release_gate != {"mode": ESMC_RELEASE_GATE_MODES[backend], "status": "passed"}:
        raise EsmcReportError("Measured ESMC release-gate identity is invalid")

    metric_context = f"{spec.id}:bf16:{backend}:{panel}"
    panel_layout = _validate_esmc_tensor_metrics(
        payload["panel_tensor_metrics"],
        context="ESMC panel",
        expected_metric_context=metric_context,
    )
    _validate_esmc_logits_metrics(payload["panel_logits_metrics"], context="ESMC panel")
    for index, (panel_case, raw_case) in enumerate(zip(panel_cases, cases, strict=True)):
        case = _esmc_require_mapping(
            raw_case,
            identity_fields | {"tensor_metrics", "logits_metrics"},
            context=f"ESMC case {index}",
        )
        if not isinstance(panel_case, Mapping) or any(
            case[name] != panel_case[name] for name in identity_fields
        ):
            raise EsmcReportError(f"ESMC case {index} identity is misaligned with its panel")
        _esmc_require_sha256(case["sequence_sha256"], context=f"ESMC case {index} sequence")
        if case["source_sha256"] is not None:
            _esmc_require_sha256(case["source_sha256"], context=f"ESMC case {index} source")
        case_id = _esmc_require_text(case["case_id"], context=f"ESMC case {index} ID")
        case_layout = _validate_esmc_tensor_metrics(
            case["tensor_metrics"],
            context=f"ESMC case {case_id}",
            expected_metric_context=f"{metric_context}:case={case_id}",
        )
        if case_layout != panel_layout:
            raise EsmcReportError(f"ESMC case {case_id} metric layout differs from its panel")
        _validate_esmc_logits_metrics(case["logits_metrics"], context=f"ESMC case {case_id}")
    if backend in {"sdpa", "eager"} and violations:
        raise EsmcReportError(f"ESMC strict backend {backend} has published-band violations")


def load_esmc_report_set(
    report_root: Path,
    registry: ModelRegistry,
    *,
    source_root: Path | None = None,
    expected_runtime_identity: EsmcRuntimeIdentity | None = None,
) -> EsmcReportSet:
    """Load exactly 30 immutable schema-v3 records and fail closed on any drift."""

    source_root = (source_root or Path(__file__).resolve().parents[2]).resolve()
    expected_specs = tuple(spec.id for spec in registry.by_family("esm_plusplus"))
    if expected_specs != ESMC_MODEL_IDS:
        raise EsmcReportError(
            f"ESMC manifest inventory {expected_specs!r} differs from {ESMC_MODEL_IDS!r}"
        )
    family = registry.families["esm_plusplus"]
    supported_backends = tuple(
        backend
        for backend in family.attention
        if "bfloat16" in registry.supported_attention_dtypes(family.id, backend)
    )
    if supported_backends != ESMC_BACKENDS:
        raise EsmcReportError(
            f"ESMC BF16 backend inventory {supported_backends!r} differs from {ESMC_BACKENDS!r}"
        )
    runtime_identity = expected_runtime_identity or _esmc_runtime_identity_from_source(
        source_root, registry
    )
    _validate_esmc_runtime_identity(runtime_identity)
    panels = _expected_esmc_panels(source_root)
    expected_reference_sources = _expected_biohub_source_contracts(source_root)
    expected_names = {
        f"{model_id}-{backend}-{panel}.json"
        for model_id in ESMC_MODEL_IDS
        for backend in ESMC_BACKENDS
        for panel in ESMC_PANEL_KINDS
    }
    if report_root.is_symlink():
        raise EsmcReportError(f"ESMC report root must not be a symlink: {report_root}")
    report_root = report_root.resolve()
    if not report_root.exists() or not report_root.is_dir():
        raise EsmcReportError(f"ESMC report root is not a real directory: {report_root}")
    entries = tuple(report_root.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise EsmcReportError("ESMC report root contains a symlink or non-file entry")
    observed_names = {entry.name for entry in entries}
    if len(observed_names) != len(entries):
        raise EsmcReportError("ESMC report root contains duplicate path identities")
    if observed_names != expected_names:
        missing = sorted(expected_names.difference(observed_names))
        unexpected = sorted(observed_names.difference(expected_names))
        raise EsmcReportError(
            "ESMC release evidence must contain exactly 30 records; "
            f"missing={missing}, unexpected={unexpected}"
        )

    reports: list[dict[str, object]] = []
    for model_id in ESMC_MODEL_IDS:
        spec = registry[model_id]
        for backend in ESMC_BACKENDS:
            for panel in ESMC_PANEL_KINDS:
                path = report_root / f"{model_id}-{backend}-{panel}.json"
                payload = _esmc_read_json(path)
                _validate_esmc_report(
                    payload,
                    spec=spec,
                    backend=backend,
                    panel=panel,
                    expected_panel=panels[panel],
                    expected_reference_sources=expected_reference_sources,
                    runtime_identity=runtime_identity,
                    registry=registry,
                    source_root=source_root,
                )
                reports.append(payload)
    if len(reports) != ESMC_REPORT_COUNT:
        raise EsmcReportError(
            f"ESMC release evidence contains {len(reports)} validated records, expected 30"
        )
    measured_count = sum(report["record_status"] == "measured" for report in reports)
    unavailable_count = sum(report["record_status"] == "unavailable" for report in reports)
    if measured_count != 18 or unavailable_count != 12:
        raise EsmcReportError(
            "ESMC release evidence must contain exactly 18 measured and 12 structured "
            "unavailable records"
        )

    candidate_environments = {
        json.dumps(report["environment"], sort_keys=True) for report in reports
    }
    reference_environments = {
        json.dumps(report["reference"]["environment"], sort_keys=True)
        for report in reports
        if isinstance(report["reference"], Mapping)
    }
    locked_reference_environments = {
        json.dumps(report["reference"]["reference_environment"], sort_keys=True)
        for report in reports
        if isinstance(report["reference"], Mapping)
    }
    reference_sources = {
        json.dumps(report["reference"]["reference_sources"], sort_keys=True)
        for report in reports
        if isinstance(report["reference"], Mapping)
    }
    if (
        len(candidate_environments) != 1
        or len(reference_environments) != 1
        or len(locked_reference_environments) != 1
        or len(reference_sources) != 1
    ):
        raise EsmcReportError(
            "ESMC release evidence crosses candidate/reference devices, software "
            "environments, or source attestations"
        )
    candidate_environment = reports[0]["environment"]
    reference = reports[0]["reference"]
    if not isinstance(candidate_environment, dict):
        raise EsmcReportError("Validated ESMC candidate environment is not an object")
    if not isinstance(reference, dict):
        raise EsmcReportError("Validated ESMC reference identity is not an object")
    reference_environment = reference["reference_environment"]
    if not isinstance(reference_environment, dict):
        raise EsmcReportError("Validated ESMC reference environment is not an object")
    return EsmcReportSet(
        reports=tuple(reports),
        runtime_identity=runtime_identity,
        candidate_environment=candidate_environment,
        reference_environment=reference_environment,
    )


def _esmc_kernel_label(kernel: object) -> str:
    kernel = _esmc_require_object(kernel, context="Rendered ESMC kernel identity")
    if kernel["provider"] == "torch":
        return f"Torch {kernel['torch_version']}"
    return (
        f"{kernel['repository']} "
        f"v{kernel['version']} ({kernel['expected_variant']})"
    )


def _esmc_reference_source_table(value: object) -> list[str]:
    sources = _esmc_require_mapping(
        value,
        set(ESMC_REFERENCE_SOURCE_NAMES),
        context="Rendered ESMC reference sources",
    )
    lines = [
        "Every report carries both official reference source attestations:",
        "",
        _table_row(
            "Source",
            "Schema",
            "Package",
            "Revision",
            "Import file",
            "Tree SHA-256",
            "Attestation SHA-256",
            "Files",
        ),
        _table_row("---", "---", "---", "---", "---", "---", "---", "---"),
    ]
    for source_name in ESMC_REFERENCE_SOURCE_NAMES:
        source = _esmc_require_object(
            sources[source_name],
            context=f"Rendered ESMC reference source {source_name}",
        )
        lines.append(
            _table_row(
                f"`{source_name}`",
                f"`{source['schema_version']}`",
                f"`{source['import_name']} {source['package_version']}`",
                f"`{source['source_revision']}`",
                f"`{source['import_file']}` under `{source['import_root']}`",
                f"`{source['tree_sha256']}`",
                f"`{source['attestation_sha256']}`",
                f"`{source['file_count']}`",
            )
        )
    return lines


def _esmc_number(value: object) -> str:
    numeric = _esmc_require_finite(value, context="Rendered ESMC metric")
    if numeric == 0:
        return "0"
    return f"{numeric:.6g}"


def _esmc_range(values: Iterable[object]) -> str:
    numbers = [
        _esmc_require_finite(value, context="Rendered ESMC range metric")
        for value in values
    ]
    return f"{_esmc_number(min(numbers))} to {_esmc_number(max(numbers))}"


def _esmc_distribution(values: Iterable[object]) -> str:
    numbers = [
        _esmc_require_finite(value, context="Rendered ESMC distribution metric")
        for value in values
    ]
    return (
        f"{_esmc_number(min(numbers))} / "
        f"{_esmc_number(statistics.median(numbers))} / {_esmc_number(max(numbers))}"
    )


def _esmc_pip_check_disclosure(
    evidence: EsmcReportSet | None,
    *,
    heading: str,
) -> str:
    if evidence is None:
        status = "The frozen oracle lock permits"
        exception: Mapping[str, object] = {
            "accepted_diagnostic": (
                "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform"
            ),
            "distribution": "nvidia-cusparselt-cu13",
            "version": "0.8.1",
            "wheel_filename": ("nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl"),
            "wheel_sha256": ("4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f"),
            "filename_platform_tag": "py3-none-manylinux2014_aarch64",
            "wheel_metadata_platform_tag": "py3-none-manylinux2014_sbsa",
            "target_hardware": "NVIDIA GH200 480GB",
            "target_operating_system": "linux",
            "target_architecture": "aarch64",
            "resolution": "validated-vendor-metadata-exception-no-wheel-rewrite",
        }
    else:
        status = "The validated oracle environment recorded"
        pip_check = _esmc_require_mapping(
            evidence.reference_environment.get("pip_check"),
            {
                "status",
                "returncode",
                "diagnostics",
                "accepted_platform_exceptions",
            },
            context="Rendered ESMC pip-check evidence",
        )
        diagnostics = _esmc_require_list(
            pip_check["diagnostics"], context="Rendered ESMC pip-check diagnostics"
        )
        exceptions = _esmc_require_list(
            pip_check["accepted_platform_exceptions"],
            context="Rendered ESMC pip-check platform exceptions",
        )
        if (
            pip_check["status"] != "accepted-platform-exception"
            or pip_check["returncode"] != 1
            or len(diagnostics) != 1
            or len(exceptions) != 1
        ):
            raise EsmcReportError("Rendered ESMC pip-check exception identity is invalid")
        exception = _esmc_require_object(
            exceptions[0], context="Rendered ESMC pip-check platform exception"
        )
        if diagnostics[0] != exception.get("accepted_diagnostic"):
            raise EsmcReportError("Rendered ESMC pip-check diagnostic is not attested")
    return f"""\
{heading} Locked oracle package compatibility exception

{status} exactly one nonzero `pip check` diagnostic:
`{exception["accepted_diagnostic"]}`. It applies only to
`{exception["distribution"]}=={exception["version"]}` on
`{exception["target_hardware"]}` / `{exception["target_operating_system"]}` /
`{exception["target_architecture"]}`. The vendor filename tag is
`{exception["filename_platform_tag"]}`, while the wheel metadata declares
`{exception["wheel_metadata_platform_tag"]}`. The exact wheel is
`{exception["wheel_filename"]}` with SHA-256 `{exception["wheel_sha256"]}`.
FastPLMs accepts this vendor metadata mismatch only after the lock, installed
inventory, wheel bytes, metadata tag, and target identity all match. The wheel
is not rewritten (`{exception["resolution"]}`). Any additional diagnostic or
identity drift fails closed.
"""


def _esmc_diagnostic_table(
    backends: Iterable[tuple[str, str]],
    *,
    model_id: str,
    evidence: EsmcReportSet | None,
) -> str:
    backend_rows = tuple(backends)
    if evidence is None:
        lines = [
            _table_row("Backend", "Support", "Measurement status"),
            _table_row("---", "---", "---"),
            _table_row(
                "`sdpa`",
                "Recommended fidelity path",
                "Pending release measurement",
            ),
        ]
        for backend, support in backend_rows:
            if backend in ESMC_UNAVAILABLE_BACKENDS:
                status = "Unavailable on current GH200/aarch64 lock"
            else:
                status = "Pending release measurement"
            lines.append(
                _table_row(
                    f"`{backend}`",
                    support,
                    status,
                )
            )
        return "\n".join(lines)

    display_backends = ("sdpa", *(backend for backend, _ in backend_rows))
    model_reports = tuple(
        evidence.get(model_id, backend, panel)
        for backend in display_backends
        for panel in ESMC_PANEL_KINDS
    )
    if len(model_reports) != len(display_backends) * len(ESMC_PANEL_KINDS):
        raise EsmcReportError(f"ESMC evidence for {model_id!r} is incomplete")
    measured_reports = tuple(
        report for report in model_reports if report["record_status"] == "measured"
    )
    unavailable_reports = tuple(
        report for report in model_reports if report["record_status"] == "unavailable"
    )
    expected_measured = len(set(display_backends).intersection(ESMC_MEASURED_BACKENDS)) * len(
        ESMC_PANEL_KINDS
    )
    expected_unavailable = len(set(display_backends).intersection(ESMC_UNAVAILABLE_BACKENDS)) * len(
        ESMC_PANEL_KINDS
    )
    if (
        len(measured_reports) != expected_measured
        or len(unavailable_reports) != expected_unavailable
    ):
        raise EsmcReportError(
            f"ESMC evidence for {model_id!r} must contain {expected_measured} measurements "
            f"and {expected_unavailable} structured unavailable records"
        )
    gpu = _esmc_require_object(
        evidence.candidate_environment["gpu"],
        context="Rendered ESMC candidate GPU identity",
    )
    capability = _esmc_require_gpu_capability(
        gpu["capability"],
        context="Rendered ESMC candidate GPU capability",
    )
    reference = _esmc_require_object(
        model_reports[0]["reference"], context="Rendered ESMC reference identity"
    )
    lines = [
        "The following values come from the complete validated schema-v3 release set.",
        f"All reports used `{gpu['name']}` (SM{capability[0]}{capability[1]}, "
        f"{gpu['total_memory_bytes']} bytes), BF16, runtime "
        f"`{evidence.runtime_identity.runtime_revision}`, source tree "
        f"`{evidence.runtime_identity.source_tree_sha256}`, and runtime bundle "
        f"`{evidence.runtime_identity.runtime_bundle_sha256}`. Results are evidence for this",
        "exact accelerator identity and are not cross-device equivalence claims.",
        "",
    ]
    lines.extend(_esmc_reference_source_table(reference["reference_sources"]))
    lines.extend(
        [
            "",
            "### Measurement identity",
            "",
            _table_row(
                "Panel",
                "Configured/effective",
                "dtype",
                "Kernel",
                "Release gate",
                "Catastrophe gate",
                "Band warnings",
                "Report SHA-256",
            ),
            _table_row("---", "---", "---", "---", "---", "---", "---", "---"),
        ]
    )
    for report in model_reports:
        panel = _esmc_require_object(report["panel"], context="Rendered ESMC panel identity")
        release_gate = _esmc_require_object(
            report["release_gate"], context="Rendered ESMC release gate"
        )
        violations = _esmc_require_list(
            report["published_band_violations"],
            context="Rendered ESMC band warnings",
        )
        lines.append(
            _table_row(
                f"`{panel['kind']}` (`{str(panel['definition_sha256'])[:12]}`)",
                (
                    f"`{report['configured_backend']}` / `{report['effective_backend']}`"
                    if report["effective_backend"] is not None
                    else f"`{report['configured_backend']}` / not dispatched"
                ),
                f"`{report['dtype']}`",
                _esmc_kernel_label(report["kernel"]),
                f"`{release_gate['mode']}` / `{release_gate['status']}`",
                f"`{report['catastrophic_gate']}`",
                str(len(violations)),
                f"`{report['report_sha256']}`",
            )
        )
    if unavailable_reports:
        lines.extend(
            (
                "",
                "### Locked-platform unavailable backends",
                "",
                "These records are availability evidence, not numerical measurements. The",
                "backend remains supported, but dispatch fails closed when its locked kernel",
                "is unavailable on the exact report-bound release environment named below.",
                "",
                _table_row(
                    "Backend",
                    "Panel",
                    "Platform",
                    "Dispatch contract",
                    "Historical evidence",
                    "Reason",
                    "Report SHA-256",
                ),
                _table_row("---", "---", "---", "---", "---", "---", "---"),
            )
        )
        for report in unavailable_reports:
            panel = _esmc_require_object(
                report["panel"], context="Rendered unavailable ESMC panel identity"
            )
            unavailable = _esmc_require_object(
                report["unavailability"], context="Rendered ESMC unavailability identity"
            )
            lines.append(
                _table_row(
                    f"`{report['configured_backend']}`",
                    f"`{panel['kind']}`",
                    f"`{unavailable['platform']}` / `{unavailable['accelerator']}`",
                    f"`{unavailable['dispatch_contract']}`",
                    f"`{unavailable['historical_evidence']}`",
                    str(unavailable["reason"]),
                    f"`{report['report_sha256']}`",
                )
            )
    lines.extend(
        (
            "",
            "### Panel aggregates",
            "",
            "Tensor cells are the minimum-to-maximum range across every hidden-state layer,",
            "last hidden state, and logits entry in `panel_tensor_metrics`. Top-1 and JSD are",
            "the panel-level `panel_logits_metrics` aggregates. These are measured values,",
            "not release thresholds.",
            "",
            _table_row(
                "Backend",
                "Panel",
                "Relative L2",
                "Q99.9",
                "Residue cosine P01",
                "Pooled cosine min",
                "Top-1",
                "JSD",
            ),
            _table_row("---", "---", "---", "---", "---", "---", "---", "---"),
        )
    )
    for report in measured_reports:
        panel = _esmc_require_object(report["panel"], context="Rendered ESMC panel identity")
        raw_metrics = _esmc_require_list(
            report["panel_tensor_metrics"], context="Rendered ESMC panel metrics"
        )
        metrics = [
            _esmc_require_object(metric, context="Rendered ESMC panel tensor metric")
            for metric in raw_metrics
        ]
        logits_metrics = _esmc_require_object(
            report["panel_logits_metrics"], context="Rendered ESMC logits metrics"
        )
        lines.append(
            _table_row(
                f"`{report['configured_backend']}`",
                f"`{panel['kind']}`",
                _esmc_range(metric["relative_l2"] for metric in metrics),
                _esmc_range(metric["relative_q999"] for metric in metrics),
                _esmc_range(metric["residue_cosine_p01"] for metric in metrics),
                _esmc_range(metric["pooled_cosine_min"] for metric in metrics),
                _esmc_number(logits_metrics["confident_top1_agreement"]),
                _esmc_number(logits_metrics["mean_jsd"]),
            )
        )
    lines.extend(
        (
            "",
            "### Per-case distributions",
            "",
            "Tensor cells are minimum / median / maximum across every case, output, and",
            "hidden-state layer in `cases[].tensor_metrics`. Top-1 and JSD use the same",
            "minimum / median / maximum summary over `cases[].logits_metrics`.",
            "",
            _table_row(
                "Backend",
                "Panel",
                "Relative L2",
                "Q99.9",
                "Residue cosine P01",
                "Pooled cosine min",
                "Top-1",
                "JSD",
            ),
            _table_row("---", "---", "---", "---", "---", "---", "---", "---"),
        )
    )
    for report in measured_reports:
        panel = _esmc_require_object(report["panel"], context="Rendered ESMC panel identity")
        raw_cases = _esmc_require_list(report["cases"], context="Rendered ESMC cases")
        cases = [_esmc_require_object(case, context="Rendered ESMC case") for case in raw_cases]
        tensor_metrics: list[Mapping[str, object]] = []
        case_logits: list[Mapping[str, object]] = []
        for case in cases:
            raw_case_metrics = _esmc_require_list(
                case["tensor_metrics"], context="Rendered ESMC case tensor metrics"
            )
            tensor_metrics.extend(
                _esmc_require_object(metric, context="Rendered ESMC case tensor metric")
                for metric in raw_case_metrics
            )
            case_logits.append(
                _esmc_require_object(
                    case["logits_metrics"], context="Rendered ESMC case logits metrics"
                )
            )
        lines.append(
            _table_row(
                f"`{report['configured_backend']}`",
                f"`{panel['kind']}`",
                _esmc_distribution(metric["relative_l2"] for metric in tensor_metrics),
                _esmc_distribution(metric["relative_q999"] for metric in tensor_metrics),
                _esmc_distribution(metric["residue_cosine_p01"] for metric in tensor_metrics),
                _esmc_distribution(metric["pooled_cosine_min"] for metric in tensor_metrics),
                _esmc_distribution(metric["confident_top1_agreement"] for metric in case_logits),
                _esmc_distribution(metric["mean_jsd"] for metric in case_logits),
            )
        )
    return "\n".join(lines)


def _render_esmc_capability_evidence(evidence: EsmcReportSet | None) -> list[str]:
    lines = [
        "## Frozen ESMC release evidence",
        "",
    ]
    if evidence is None:
        lines.extend(
            (
                "**Status: pending.** Default documentation generation never discovers or",
                "trusts reports implicitly. Release rendering requires an explicitly selected,",
                "complete schema-v3 set of exactly 30 records on one exact GH200/aarch64",
                "target: 18 measured eager, SDPA, and Flex records plus 12 structured",
                "FlashAttention 2/3 unavailable records across three checkpoints and two",
                "immutable sequence panels.",
                "The set must also carry the final candidate/reference image identities,",
                "dependency lock, installed inventory, and official source attestations.",
                "A partial, stale, malformed, self-digest-invalid, or cross-device set fails",
                "closed and cannot replace this status.",
                "",
            )
        )
        lines.extend(_esmc_pip_check_disclosure(evidence, heading="###").rstrip().splitlines())
        lines.append("")
        return lines

    gpu = _esmc_require_object(
        evidence.candidate_environment["gpu"],
        context="Rendered ESMC candidate GPU identity",
    )
    capability = _esmc_require_gpu_capability(
        gpu["capability"],
        context="Rendered ESMC candidate GPU capability",
    )
    reference = _esmc_require_object(
        evidence.reports[0]["reference"], context="Rendered ESMC reference identity"
    )
    lines.extend(
        (
            f"**Status: validated complete set ({len(evidence.reports)}/30 records).**",
            "The set contains 18 measured eager, SDPA, and Flex records and 12",
            "structured FlashAttention 2/3 locked-platform unavailable records.",
            "",
            f"Exact device: `{gpu['name']}`; capability: `SM{capability[0]}"
            f"{capability[1]}`; memory: `{gpu['total_memory_bytes']}` bytes; "
            "dtype: `bfloat16`.",
            f"Runtime revision: `{evidence.runtime_identity.runtime_revision}`; source-tree "
            f"SHA-256: `{evidence.runtime_identity.source_tree_sha256}`; runtime-bundle "
            f"SHA-256: `{evidence.runtime_identity.runtime_bundle_sha256}`.",
            "",
        )
    )
    lines.extend(_esmc_reference_source_table(reference["reference_sources"]))
    lines.extend(("",))
    lines.extend(_esmc_pip_check_disclosure(evidence, heading="###").rstrip().splitlines())
    lines.extend(
        (
            "",
            "Results are not transferred to another accelerator identity. Each model card",
            "defines and publishes the corresponding per-case minimum/median/maximum",
            "distributions.",
            "",
            _table_row(
                "Checkpoint",
                "Backend",
                "Panel",
                "Relative L2 range",
                "Q99.9 range",
                "Residue cosine range",
                "Pooled cosine range",
                "Top-1",
                "JSD",
                "Band warnings",
            ),
            _table_row("---", "---", "---", "---", "---", "---", "---", "---", "---", "---"),
        )
    )
    measured_reports = tuple(
        report for report in evidence.reports if report["record_status"] == "measured"
    )
    unavailable_reports = tuple(
        report for report in evidence.reports if report["record_status"] == "unavailable"
    )
    if len(measured_reports) != 18 or len(unavailable_reports) != 12:
        raise EsmcReportError(
            "Rendered ESMC set must contain 18 measurements and 12 unavailable records"
        )
    for report in measured_reports:
        panel = _esmc_require_object(report["panel"], context="Rendered ESMC panel identity")
        raw_metrics = _esmc_require_list(
            report["panel_tensor_metrics"], context="Rendered ESMC panel metrics"
        )
        metrics = [
            _esmc_require_object(metric, context="Rendered ESMC panel tensor metric")
            for metric in raw_metrics
        ]
        logits = _esmc_require_object(
            report["panel_logits_metrics"], context="Rendered ESMC logits metrics"
        )
        violations = _esmc_require_list(
            report["published_band_violations"],
            context="Rendered ESMC band warnings",
        )
        lines.append(
            _table_row(
                f"`{report['model_id']}`",
                f"`{report['configured_backend']}`",
                f"`{panel['kind']}` (`{str(panel['definition_sha256'])[:12]}`)",
                _esmc_range(metric["relative_l2"] for metric in metrics),
                _esmc_range(metric["relative_q999"] for metric in metrics),
                _esmc_range(metric["residue_cosine_p01"] for metric in metrics),
                _esmc_range(metric["pooled_cosine_min"] for metric in metrics),
                _esmc_number(logits["confident_top1_agreement"]),
                _esmc_number(logits["mean_jsd"]),
                str(len(violations)),
            )
        )
    lines.extend(
        (
            "",
            "### Current locked-platform Flash availability",
            "",
            _table_row(
                "Checkpoint",
                "Backend",
                "Panel",
                "Status",
                "Dispatch contract",
                "Historical evidence",
                "Reason",
            ),
            _table_row("---", "---", "---", "---", "---", "---", "---"),
        )
    )
    for report in unavailable_reports:
        panel = _esmc_require_object(
            report["panel"], context="Rendered unavailable ESMC panel identity"
        )
        unavailable = _esmc_require_object(
            report["unavailability"], context="Rendered ESMC unavailability identity"
        )
        lines.append(
            _table_row(
                f"`{report['model_id']}`",
                f"`{report['configured_backend']}`",
                f"`{panel['kind']}`",
                (f"`unavailable` on `{unavailable['platform']}` / `{unavailable['accelerator']}`"),
                f"`{unavailable['dispatch_contract']}`",
                f"`{unavailable['historical_evidence']}`",
                str(unavailable["reason"]),
            )
        )
    lines.append("")
    return lines


def render_support(registry: ModelRegistry) -> str:
    """Render the complete support matrix without importing model code."""

    lines = [
        GENERATED_MARKER,
        "",
        "# Model support",
        "",
        "This file is generated from `src/fastplms/models.toml`. A listed capability is",
        "selectable. Strict-parity exceptions are documented in the checkpoint cards.",
        "",
        "## Family interfaces",
        "",
        "| Family | Architecture | Checkpoints | Public input | AutoClasses | Tokenizer class |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for family in registry.families.values():
        count = len(registry.by_family(family.id))
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{family.id}`",
                    family.architecture,
                    str(count),
                    family.public_input.replace("|", "\\|"),
                    _code(sorted(family.auto_map)),
                    _tokenizer_class_label(family),
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## AutoClass weight status",
            "",
            "`pretrained` means the advertised head is present in the checkpoint. "
            "`base weights + untrained task head` means the task head must be "
            "trained before use. `FastPLMs extension` is an integration or head "
            "that is not an official pretrained ANKH capability.",
            "",
            "| Family | AutoClass | Weight status |",
            "| --- | --- | --- |",
        )
    )
    for family in registry.families.values():
        for auto_class in sorted(family.auto_map):
            lines.append(
                f"| `{family.id}` | `{auto_class}` | `{_auto_class_status(family, auto_class)}` |"
            )

    lines.extend(
        (
            "",
            "## Family execution",
            "",
            "| Family | Attention | Precision | BF16 execution | Extra | Reference |",
            "| --- | --- | --- | --- | --- | --- |",
        )
    )
    for family in registry.families.values():
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{family.id}`",
                    _code(family.attention),
                    _precision_contract(family),
                    f"`{family.bf16_execution}`",
                    f"`{family.extra}`",
                    f"`{family.reference_container}`",
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## Family release contracts",
            "",
            "| Family | Checkpoint terms | Hub license | Weight publication | Tiers |",
            "| --- | --- | --- | --- | --- |",
        )
    )
    for family in registry.families.values():
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{family.id}`",
                    family.checkpoint_license.replace("|", "\\|"),
                    _hub_license_label(family),
                    (
                        "blocked"
                        if not family.weights_publication_allowed
                        else "complete checkpoint required"
                        if family.requires_complete_weight_publication
                        else "manifest policy"
                    ),
                    _code(family.test_tiers),
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## Runtime assets",
            "",
            _table_row(
                "ID",
                "Family",
                "Repository",
                "Path",
                "SHA-256",
                "Size",
                "License",
                "Trust boundary",
                "Offline behavior",
            ),
            _table_row("---", "---", "---", "---", "---", "---:", "---", "---", "---"),
        )
    )
    for asset in registry.runtime_assets.values():
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{asset.id}`",
                    f"`{asset.consumer_family}`",
                    f"`{asset.repository}`",
                    f"`{asset.path}`",
                    f"`{asset.sha256}`",
                    str(asset.size),
                    f"`{asset.license_expression}`",
                    f"`{asset.trust_kind}`",
                    f"`{asset.offline_behavior}`",
                )
            )
            + " |"
        )

    lines.extend(
        (
            "",
            "## Checkpoints",
            "",
            "| ID | Family | Size | FastPLMs checkpoint | Official checkpoint | "
            "Artifact source | State transform | Generation contract | MSA conditioning | "
            "Unresolved files |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |",
        )
    )
    for spec in registry.values():
        fast_url = f"https://huggingface.co/{spec.fast.repo_id}"
        official_url = f"https://huggingface.co/{spec.official.repo_id}"
        unresolved = len(spec.fast.unresolved_files) + len(spec.official.unresolved_files)
        if spec.family.id == "esmfold2":
            if spec.msa_conditioning is None:
                raise ValueError(f"{spec.id}: ESMFold2 MSA conditioning is undeclared")
            msa_conditioning = (
                "`optional` (full checkpoint)"
                if spec.msa_conditioning
                else "`none` (Fast; MSA inputs rejected)"
            )
        else:
            msa_conditioning = "not applicable"
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{spec.id}`",
                    f"`{spec.family.id}`",
                    f"`{spec.size_category}`",
                    f"[{spec.fast.repo_id}]({fast_url})",
                    f"[{spec.official.repo_id}]({official_url})",
                    f"`{spec.artifact_source}`",
                    f"`{spec.family.state_transform}`",
                    f"`{spec.generation_contract}`",
                    msa_conditioning,
                    str(unresolved),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "A nonzero unresolved-file count blocks release. It is not permission to",
            "omit that file from checkpoint, tokenizer, artifact, or compliance checks.",
            "",
        )
    )
    return "\n".join(lines)


def render_capability_evidence(
    registry: ModelRegistry,
    *,
    esmc_evidence: EsmcReportSet | None = None,
) -> str:
    """Render the release evidence required for every advertised capability."""

    missing_families = sorted(set(registry.families).difference(FAMILY_DOCUMENTATION))
    if missing_families:
        raise ValueError(
            "Capability evidence has no documentation mapping for: " + ", ".join(missing_families)
        )

    lines = [
        GENERATED_MARKER,
        "",
        "# Capability-to-evidence manifest",
        "",
        "This manifest maps every advertised FastPLMs 1.0 capability to its user",
        "documentation, runnable example, and required validation tier. It is a",
        "coverage contract, not a statement that an unreported run passed. The exact",
        "checkpoint list and family declarations come from `src/fastplms/models.toml`.",
        "",
        "The Example column links a curated CLI when that interface exposes the whole",
        "capability. Programmatic-only forms instead link their runnable CPU contract so",
        "the manifest does not imply broader CLI coverage than the example provides.",
        "",
    ]
    lines.extend(_render_esmc_capability_evidence(esmc_evidence))
    lines.extend(_render_evidence_selector_catalog())
    lines.extend(_render_curated_example_cpu_evidence())
    lines.extend(
        (
            "## Families and AutoClasses",
            "",
            _table_row(
                "Family",
                "Tokenizer mode",
                "AutoClass",
                "Weight status",
                "Guide",
                "Family workflow and runnable entry-point contract",
                "Required evidence",
            ),
            _table_row("---", "---", "---", "---", "---", "---", "---"),
        )
    )
    for family in registry.families.values():
        guide, _ = FAMILY_DOCUMENTATION[family.id]
        for auto_class in sorted(family.auto_map):
            example = _autoclass_workflow_example(family, auto_class)
            evidence = _render_evidence_keys(
                autoclass_evidence_keys(registry, family.id, auto_class)
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        f"`{family.id}`",
                        f"`{family.tokenizer_mode}`",
                        f"`{auto_class}`",
                        f"`{_auto_class_status(family, auto_class)}`",
                        f"[guide]({guide})",
                        (
                            f"[family workflow]({example}); "
                            "[runnable AutoClass contract](../../tests/cpu/"
                            "test_autoclass_evidence_matrix.py)"
                        ),
                        evidence,
                    )
                )
                + " |"
            )

    lines.extend(
        (
            "",
            "## Attention backends",
            "",
            "| Backend | Advertising families | Guide | Example | Required evidence |",
            "| --- | --- | --- | --- | --- |",
        )
    )
    advertised_backends = sorted(
        {backend for family in registry.families.values() for backend in family.attention}
    )
    for backend in advertised_backends:
        families = sorted(
            family.id for family in registry.families.values() if backend in family.attention
        )
        evidence = _render_evidence_keys(attention_backend_evidence_keys(registry, backend))
        lines.append(
            f"| `{backend}` | {_code(families)} | "
            "[guide](../attention_backends.md) | "
            "[example](../../examples/attention_switching.py) | "
            f"{evidence} |"
        )

    lines.extend(
        (
            "",
            "## Input, embedding, and storage contracts",
            "",
            _table_row("Capability", "Guide", "Example", "Required evidence"),
            _table_row("---", "---", "---", "---"),
        )
    )
    _append_capability_rows(lines, EMBEDDING_CAPABILITY_ROWS)

    lines.extend(
        (
            "",
            "## Generation and adaptation contracts",
            "",
            _table_row("Capability", "Guide", "Example", "Required evidence"),
            _table_row("---", "---", "---", "---"),
        )
    )
    _append_capability_rows(lines, GENERATION_CAPABILITY_ROWS)

    lines.extend(
        (
            "",
            "## Structure contracts",
            "",
            _table_row("Capability", "Guide", "Example", "Required evidence"),
            _table_row("---", "---", "---", "---"),
        )
    )
    _append_capability_rows(lines, STRUCTURE_CAPABILITY_ROWS)
    _append_capability_rows(lines, _esmfold2_structure_capability_rows(registry))
    lines.extend(
        (
            "",
            "Release evidence must name the exact head, checkpoint and runtime revisions,",
            "tokenizer identity, backend, dtype, hardware, sequence or structure panel,",
            "seed, environment, and input hash. Missing evidence remains visibly pending;",
            "it must not be replaced by a synthetic benchmark number or an inferred claim.",
            "",
        )
    )
    return "\n".join(lines)


def _preferred_auto_class(spec: ModelSpec) -> str:
    preference = (
        "AutoModel",
        "AutoModelForMaskedLM",
        "AutoModelForSeq2SeqLM",
        "AutoModelForProteinFolding",
    )
    for name in preference:
        if name in spec.auto_map:
            return name
    return sorted(spec.auto_map)[0]


def _feature_statuses(spec: ModelSpec) -> tuple[tuple[str, str], ...]:
    """Return concise, checkpoint-specific public capability statuses."""

    family_id = spec.family.id
    sequence_head = "AutoModelForSequenceClassification" in spec.auto_map
    token_head = "AutoModelForTokenClassification" in spec.auto_map

    if family_id == "esmfold2":
        embedding = "Special: ESMC state mixture to 256-wide residue embeddings"
    elif family_id == "ankh":
        embedding = "Special: encoder or explicitly prepared decoder states"
    elif family_id == "e1":
        embedding = "Special: tokenizer-free raw-sequence preparation"
    elif family_id in EMBEDDING_FAMILIES:
        embedding = "Supported: shared ordered embedding API"
    else:
        embedding = "Unavailable for this structure-only checkpoint"

    if family_id in SEQUENCE_TTT_AUTO_CLASSES:
        ttt = "Supported: low-rank masked-residue adaptation"
    elif family_id == "esmfold2" and "experimental" not in spec.id:
        ttt = "Special: opt-in folding TTT on the ESMC backbone"
    elif family_id == "esmfold":
        ttt = "Unavailable: the checkpoint has no trained MLM head"
    elif family_id == "esmfold2":
        ttt = "Unavailable for this experimental checkpoint"
    else:
        ttt = "Unavailable for this inference-only checkpoint"

    if spec.family.id == "esm_plusplus":
        attention = "Special: SDPA fidelity path; alternate backends have explicit bands"
    else:
        attention = f"Supported: {_code(spec.family.attention)}"

    compliance = (
        "Declared: exact release evidence is required"
        if "compliance" in spec.family.test_tiers
        else "Unavailable: this provisional family has no compliance tier"
    )

    return (
        (
            "Sequence classification",
            "Supported: base weights with an untrained task head"
            if sequence_head
            else "Unavailable: no advertised AutoClass",
        ),
        (
            "Token classification",
            "Supported: base weights with an untrained task head"
            if token_head
            else "Unavailable: no advertised AutoClass",
        ),
        (
            "PEFT fine-tuning",
            "Supported pattern: preserve the separately trained `classifier`"
            if sequence_head
            else "Supported pattern: attach LoRA to the pretrained model",
        ),
        ("Embeddings", embedding),
        ("Test-time training", ttt),
        ("Attention variants", attention),
        ("Compliance", compliance),
    )


def _capability_summary(spec: ModelSpec) -> str:
    lines = [
        "## Capabilities",
        "",
        "| Feature | Status |",
        "| --- | --- |",
    ]
    lines.extend(f"| {feature} | {status} |" for feature, status in _feature_statuses(spec))
    lines.extend(
        (
            "",
            "A supported interface is not a pretrained downstream predictor. Classification",
            "heads start untrained, and declared compliance metadata is not a claim that an",
            "arbitrary local build passed its release gate.",
            "",
        )
    )
    return "\n".join(lines)


def _task_head_usage(spec: ModelSpec) -> str:
    if not {
        "AutoModelForSequenceClassification",
        "AutoModelForTokenClassification",
    }.issubset(spec.auto_map):
        return ""

    model_id = spec.fast.repo_id
    if spec.family.id == "e1":
        preparation = """\
sequences = ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"]
batch = sequence_model.prep_tokens.get_batch_kwargs(
    sequences,
    device=sequence_model.device,
)
biological = batch["sequence_ids"].ne(-1)
"""
    else:
        preparation = f"""\
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
sequences = ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"]
batch = tokenizer(sequences, padding=True, return_tensors="pt")
biological = batch["attention_mask"].bool()
for special_id in tokenizer.all_special_ids:
    biological &= batch["input_ids"].ne(special_id)
"""

    tokenizer_import = (
        "" if spec.family.id == "e1" else "from transformers import AutoTokenizer\n"
    )
    return f"""\
## Downstream classification

Both downstream AutoClasses reuse the checkpoint backbone and initialize a new,
untrained `classifier`. Sequence labels have shape `(b,)`; residue labels have
shape `(b, l)` and use `-100` outside biological positions:

```python
import torch
{tokenizer_import}\
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

model_id = "{model_id}"
sequence_model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=2, trust_remote_code=True
).eval()
token_model = AutoModelForTokenClassification.from_pretrained(
    model_id, num_labels=3, trust_remote_code=True
).eval()
{preparation}
sequence_labels = torch.zeros(len(sequences), dtype=torch.long)
token_labels = torch.full_like(batch["input_ids"], -100)
token_labels[biological] = 0

with torch.inference_mode():
    sequence_output = sequence_model(**batch, labels=sequence_labels)
    token_output = token_model(**batch, labels=token_labels)
print(sequence_output.logits.shape)  # (b, 2)
print(token_output.logits.shape)     # (b, l, 3)
```

"""


def _peft_usage(spec: ModelSpec) -> str:
    has_classifier = "AutoModelForSequenceClassification" in spec.auto_map
    if has_classifier:
        model_name = "sequence_model"
        task_import = ", TaskType"
        task_type = "        task_type=TaskType.SEQ_CLS,\n"
        modules_to_save = '        modules_to_save=["classifier"],\n'
        persistence = (
            "This checkpoint advertises a classification head, so the separately "
            "trained `classifier` is saved with the adapter."
        )
    else:
        model_name = "model"
        task_import = ""
        task_type = ""
        modules_to_save = ""
        persistence = (
            "This checkpoint has no advertised classifier. Supply the task-specific "
            "objective and preserve any new head through `modules_to_save`."
        )
    return f"""\
## PEFT fine-tuning

Install the direct training dependencies, then attach LoRA to the loaded checkpoint:

```bash
python -m pip install "datasets>=4.8,<5" "peft>=0.19,<0.20"
```

```python
from peft import LoraConfig{task_import}, get_peft_model

peft_model = get_peft_model(
    {model_name},
    LoraConfig(
{task_type}\
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
{modules_to_save}\
    ),
)
```

{textwrap.fill(persistence, width=79)}
All FastPLMs checkpoints follow the Transformers `PreTrainedModel` contract and
can be adapted with PEFT. The ESM2-specific shipped CLI is an example, not a
support boundary. Record the target modules, base revision, data identity, and
trainable parameter scope.

"""


def _sequence_ttt_usage(spec: ModelSpec) -> str:
    auto_class = SEQUENCE_TTT_AUTO_CLASSES.get(spec.family.id)
    if auto_class is None:
        return ""
    return f"""\
## Test-time training

TTT samples masked views of one protein and updates only injected low-rank
adapters. Base checkpoint weights remain frozen:

```python
from transformers import {auto_class}

ttt_model = {auto_class}.from_pretrained(
    "{spec.fast.repo_id}",
    trust_remote_code=True,
)
metrics = ttt_model.ttt(
    seq="MSTNPKPQRKTKRNT",
    ttt_config={{"steps": 3, "batch_size": 1, "seed": 7}},
)
ttt_model.save_pretrained("adapted", safe_serialization=True)
ttt_model.ttt_reset()
print(metrics)
```

Persisted adapters retain their deterministic reset state. TTT adds latency
and memory, can worsen an output, and does not establish biological function.

"""


def _attention_usage(spec: ModelSpec) -> str:
    recommended = "sdpa" if "sdpa" in spec.family.attention else spec.family.attention[0]
    declared = textwrap.fill(
        f"Declared variants are {_code(spec.family.attention)}. An unavailable "
        "requested backend raises instead of silently switching implementations.",
        width=79,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if "compliance" in spec.family.test_tiers:
        compliance = (
            "This family declares the `compliance` tier. Release evidence binds the exact "
            "checkpoint, backend, dtype, hardware, inputs, and reference revision."
        )
    else:
        compliance = (
            "This family does not declare the `compliance` tier. Boltz2 remains "
            "provisional and its structure checks must not be broadened into parity claims."
        )
    return f"""\
## Attention and compliance

The quick start selects `{recommended}` explicitly. {declared}
`output_attentions=True` may use the documented, one-call eager fallback solely
to materialize attention tensors; the configured backend remains unchanged.

{textwrap.fill(compliance, width=79)}

"""


def _sequence_forward_usage(spec: ModelSpec) -> str:
    if spec.family.id not in {"esm2", "esm_plusplus", "dplm", "ankh"}:
        return ""
    if spec.family.id == "ankh":
        return f"""\
## Tokenization and forward inference

`{spec.fast.repo_id}` contains the complete encoder-decoder checkpoint.
`AutoModel` loads the encoder view without allocating the decoder, while
`AutoModelForSeq2SeqLM` loads the encoder, decoder, cross-attention, and
language-model head.

Use the tokenizer owned by the loaded model so tokenizer files, revision,
offline/cache policy, and ANKH's residue-aware pre-tokenizer stay aligned.
Pass raw protein strings without inserted residue spaces:

```python
import torch

tokenizer = model.tokenizer
batch = tokenizer(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    padding=True,
    return_tensors="pt",
)

with torch.inference_mode():
    output = model(**batch)

print(output.last_hidden_state.shape)
```

"""
    return f"""\
## Tokenization and forward inference

Load the tokenizer from the same artifact as the model. Padding is represented
explicitly by the attention mask:

```python
import torch
from transformers import AutoTokenizer

model_id = "{spec.fast.repo_id}"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)
batch = tokenizer(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    padding=True,
    return_tensors="pt",
)

with torch.inference_mode():
    output = model(**batch)

print(output.last_hidden_state.shape)
```

"""


def _embedding_usage(spec: ModelSpec) -> str:
    if spec.family.id not in EMBEDDING_FAMILIES or spec.family.id == "esmfold2":
        return ""
    if spec.family.id == "ankh":
        return f"""\
## Dataset embeddings

Dataset embeddings default to the encoder final state. Select a native encoder
layer directly:

```python
encoder_result = model.embed_dataset(
    ["MSTNPKPQRKTKRNT"],
    hidden_state_source="encoder",
    hidden_state_index=-1,
    full_embeddings=True,
)
print(encoder_result[0].tensor.shape)  # (l, d)
```

Decoder representations require `AutoModelForSeq2SeqLM` and exactly one
aligned decoder input. ANKH does not invent a shifted target:

```python
from transformers import AutoModelForSeq2SeqLM

seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
    "{spec.fast.repo_id}",
    trust_remote_code=True,
).eval()
decoder_result = seq2seq.embed_dataset(
    ["MSTNPKPQRKTKRNT"],
    hidden_state_source="decoder",
    hidden_state_index=-1,
    decoder_inputs=["M<extra_id_0>"],
    full_embeddings=True,
)
print(decoder_result[0].tensor.shape)  # (decoder_length, d)
```

Pooling excludes boundary, padding, sentinel, and other non-biological
positions. Persisted results record the selected stack, layer, inputs, masks,
and alignment policy.

"""
    return """\
## Dataset embeddings

The shared embedding mixin preserves input order and biological-position
masking. It accepts sequences, identified records, mappings, or a FASTA path:

```python
pooled = model.embed_dataset(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    batch_size=2,
    pooling=("mean", "std"),
)
residues = model.embed_dataset(
    ["MSTNPKPQRKTKRNT"],
    full_embeddings=True,
)
print(pooled[0].tensor.shape)   # (2 * d,)
print(residues[0].tensor.shape) # (l, d)
```

Set `output` and `format="safetensors"` or `"sqlite"` for transactional,
bounded-memory persistence. Resume verifies input order, model state, tokenizer
policy, backend, dtype, and pooling configuration before appending.

"""


def _family_usage_notes(
    spec: ModelSpec,
    *,
    allow_generic: bool = False,
    esmc_evidence: EsmcReportSet | None = None,
) -> str:
    family_id = spec.family.id
    model_id = spec.fast.repo_id
    if family_id == "esm2":
        return f"""\
## Masked language modeling and contacts

Use the masked-language-model AutoClass when logits are required:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
masked_model = AutoModelForMaskedLM.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
batch = tokenizer("MSTNPKPQRKTKRNT", return_tensors="pt")

with torch.inference_mode():
    logits = masked_model(**batch).logits
    contacts = masked_model.predict_contacts(
        batch["input_ids"],
        batch["attention_mask"],
    )

print(logits.shape, contacts.shape)
```

Contact prediction materializes attention maps and should not be enabled in a
high-throughput embedding path unless those maps are required.

Plain `AutoModel` omits the optional ESM pooler because this masked-language-
model checkpoint contains no trained pooler weights. Pass
`add_pooling_layer=True` only when intentionally initializing and training that
head.

"""
    if family_id == "esm_plusplus":
        sae_id, sae_layer = ESMC_SAE_EXAMPLES[spec.id]
        esmc_table = _esmc_diagnostic_table(
            (
                ("eager", "Supported"),
                ("flash_attention_2", "Supported"),
                ("flex_attention", "Supported, numerically divergent"),
                ("flash_attention_3", "Supported, numerically divergent"),
            ),
            model_id=spec.id,
            evidence=esmc_evidence,
        )
        return f"""\
## ESMC behavior

This artifact exposes the Biohub ESMC sequence encoder and masked-language-model
head through Transformers. It is also the language-model family used by
ESMFold2. SDPA is the default and the recommended choice for highest numerical
fidelity. Flex Attention and FlashAttention 3 are supported, non-experimental
backends, but their BF16 arithmetic may be numerically divergent from SDPA.
Those deviations produce diagnostic warnings rather than strict parity
failures; dispatch integrity, masks, finite outputs, shapes, and catastrophic
biological disagreement remain hard gates.

The current GH200/aarch64 release environment validates eager, SDPA, and Flex.
Flash requests fail closed because compatible locked kernels are unavailable
on this platform.

When `sequence_id` is supplied, it is authoritative for ESMC attention grouping
and padding, and `attention_mask` is ignored. Values greater than or equal to
zero are valid sequence-group IDs; `-1` denotes padding. Omit `sequence_id` to
use `attention_mask` as the padding contract.

### Hidden-state sparse autoencoders

ESM++ accepts hidden-state SAEs from the official
[Biohub ESMC SAE collection](https://huggingface.co/collections/biohub/esmc-saes-for-hidden-states-all-layers).
Choose an SAE trained for this checkpoint's ESMC scale, then load and attach
only the required layer modules:

```python
import torch
from transformers import AutoModel

sae = AutoModel.from_pretrained("{sae_id}", device=model.device)
sae.initialize_layers([{sae_layer}])
model.add_sae_models([sae.layers["{sae_layer}"]])

with torch.inference_mode():
    output = model(**batch, normalize_sae=True)

features = output.sae_outputs["layer{sae_layer}"]
print(features.shape, features.layout)  # (valid_token_count, codebook_dim), sparse COO
```

SAEs run by default after attachment; `compute_sae=False` bypasses their
execution. Outputs are detached sparse tensors keyed by `layer{{N}}`, exclude
padding under the `sequence_id`-before-`attention_mask` rule, and optionally use
Biohub's `(features / max) * idf` normalization. SAE computation requires
`input_ids` and rejects mask tokens because the SAEs were trained on unmasked
sequences. This interface supports hidden-state SAEs only, not MLP-output SAEs.
FastPLMs does not mirror SAE weights or add them to its model manifest.

### Experimental FP8 inference

The default remains the checkpoint's declared BF16 behavior. FP8 is an
explicit experimental inference opt-in on every ESM++ scale:

```python
import torch
from transformers import AutoModel

fp8_model = AutoModel.from_pretrained(
    "{model_id}",
    trust_remote_code=True,
    dtype=torch.bfloat16,
).cuda().eval()
fp8_model.enable_fp8()
print(fp8_model.esmc_precision_status)

with torch.inference_mode():
    fp8_output = fp8_model(**{{name: value.cuda() for name, value in batch.items()}})
```

FP8 forward calls require `torch.inference_mode()`, and the sequence dimension
is padded to a multiple of 16 internally. Conversion uses Transformer Engine
for the supported linear set and fails closed when the dependency, compatible
CUDA hardware, or complete conversion contract is unavailable. It never
silently falls back to BF16 and is not a numerical-parity claim.

{esmc_table}

{ESMC_RELEASE_DOCUMENTATION}

"""
    if family_id == "esm3":
        return """\
## Sequence inference and masked-sequence generation

ESM3 owns its sequence preparation. This example exercises the sequence track;
the public input contract also supports structure and function tracks through
the multimodal helpers:

```python
import torch

batch = model.tokenize_sequences(
    ["MKTAYIAKQ", "GGGG"],
    device=model.device,
)
with torch.inference_mode():
    output = model(**batch)

print(output.last_hidden_state.shape)
print(output.logits.shape)
print(output.structure_logits.shape)
print(output.function_logits.shape)
```

When `return_dict=False`, ESM3 follows the standard base-model tuple prefix:
`last_hidden_state`, then requested `hidden_states` and `attentions`. Multimodal
logits and extensions follow that prefix. Prefer named fields for individual
tracks.

Generate masked sequence positions with an explicit seed:

```python
from fastplms.models.esm3.modeling_esm3 import FastESM3GenerationConfig

config = FastESM3GenerationConfig(
    num_steps=8,
    temperature=1.0,
    seed=7,
)
generated = model.generate("MK____A", config)
print(generated)
```

Underscores mark positions to generate. Model outputs are predictions over
tracks, not experimental measurements of structure or function.

"""
    if family_id == "e1":
        return """\
## Tokenizer-free E1 input

E1 has no tokenizer. The model retains native raw-sequence preparation,
boundary tokens, sequence positions, and retrieval-augmented context behavior.
The ordinary representation path accepts sequences directly:

```python
result = model.embed_dataset(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    batch_size=2,
    pooling=("mean",),
)
print(result[0].tensor.shape)
```

Lower-level masked-language-model calls must use the E1 batch preparer rather
than an `AutoTokenizer`. E1 launch messages and distributed legal files retain
the attribution required by the upstream agreement.

"""
    if family_id == "dplm":
        license_url = "https://github.com/bytedance/dplm/blob/main/LICENSE"
        readme_url = "https://github.com/bytedance/dplm/blob/main/README.md#overview"
        return f"""\
## Diffusion sequence generation

DPLM defines the requested length from biological positions in a tokenized
input, masks those positions, and iteratively retains confident predictions:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
generator = AutoModelForMaskedLM.from_pretrained(
    model_id,
    trust_remote_code=True,
).cuda().eval()
input_ids = tokenizer("A" * 64, return_tensors="pt")["input_ids"].cuda()

with torch.inference_mode():
    generated_ids = generator.generate(input_ids, max_iter=100)

sequence = tokenizer.decode(
    generated_ids[0],
    skip_special_tokens=True,
).replace(" ", "")
print(sequence)
```

Omitting `max_iter` uses the official 500-step schedule. A shorter schedule
changes the sampling process rather than providing an equivalent faster mode.

Plain `AutoModel` omits the optional ESM pooler because this diffusion
checkpoint contains no trained pooler weights. Pass `add_pooling_layer=True`
only when intentionally initializing and training that head.

DPLM1 and DPLM2 checkpoint weights are Apache-2.0. The maintained ByteDance
[LICENSE]({license_url}) is Apache-2.0 and the
[README]({readme_url})
explicitly scopes the repository release to the pretrained DPLM1 and DPLM2
weights. FastPLMs artifacts record `weights_license_status="resolved"` and
`redistributable=true`; complete publication is permitted only after all
artifact, legal, parity, and atomic-publication preflights pass.

"""
    if family_id == "dplm2":
        license_url = "https://github.com/bytedance/dplm/blob/main/LICENSE"
        readme_url = "https://github.com/bytedance/dplm/blob/main/README.md#overview"
        return f"""\
## Amino-acid and structure co-generation

DPLM2 uses separate structure and amino-acid tracks with modality-specific
boundary and mask tokens:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
generator = AutoModelForMaskedLM.from_pretrained(
    model_id,
    trust_remote_code=True,
).cuda().eval()
vocab = tokenizer.get_vocab()
l = 64
structure = [
    vocab["<cls_struct>"],
    *([vocab["<mask_struct>"]] * l),
    vocab["<eos_struct>"],
]
amino_acids = [
    vocab["<cls_aa>"],
    *([vocab["<mask_aa>"]] * l),
    vocab["<eos_aa>"],
]
input_ids = torch.tensor([structure + amino_acids], device="cuda")

with torch.inference_mode():
    generated = generator.generate(input_ids, max_iter=100)["output_tokens"]
print(generated.shape)
```

Generic `cls_token`, `eos_token`, `mask_token`, and `unk_token` aliases are
intentionally unset. Callers constructing multimodal tensors must choose the
amino-acid or structure token explicitly. Raw amino-acid sequences remain
supported by `model.embed_dataset(...)`.

Plain `AutoModel` omits the optional ESM pooler because this co-generation
checkpoint contains no trained pooler weights. Pass `add_pooling_layer=True`
only when intentionally initializing and training that head.

The checkpoint weights are Apache-2.0. The maintained ByteDance
[LICENSE]({license_url}) and [README]({readme_url}) document the license
basis for the pretrained DPLM1 and DPLM2 weights. Complete publication remains
subject to all artifact, legal, parity, and atomic-publication preflights.

"""
    if family_id == "ankh":
        return f"""\
## Encoder and sequence-to-sequence use

`{spec.fast.repo_id}` contains the complete ANKH encoder-decoder checkpoint.
Use `AutoModel` for encoder embeddings and `AutoModelForSeq2SeqLM` for
task-specific decoding:

```python
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

repo_id = "{spec.fast.repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
encoder = AutoModel.from_pretrained(repo_id, trust_remote_code=True).eval()
seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
    repo_id,
    trust_remote_code=True,
).eval()
batch = tokenizer("MSTNPKPQRKTKRNT", return_tensors="pt")

with torch.inference_mode():
    encoder_hidden = encoder(**batch).last_hidden_state
    generated_ids = seq2seq.generate(**batch, max_new_tokens=16)
print(encoder_hidden.shape)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
```

ANKH artifacts retain CC BY-NC-SA 4.0 terms. The notes below distinguish the
official heads from FastPLMs extensions. The complete checkpoint is larger than
the former encoder-only mirror while preserving encoder-output parity.

"""
    if family_id == "boltz2":
        return """\
## Protein structure prediction

The high-level helper prepares a protein-only input, runs the declared Boltz2
inference core, and returns coordinates and confidence fields:

```python
import torch

model = model.cuda().eval()
output = model.predict_structure(
    amino_acid_sequence="MSTNPKPQRKTKRNTNRRPQDVKFPGG",
    recycling_steps=3,
    num_sampling_steps=50,
    diffusion_samples=1,
    seed=7,
)
model.save_as_cif(output, "prediction.cif")

print(output.sample_atom_coords.shape)
print(output.plddt, output.ptm, output.iptm)
```

The validation boundary below describes the currently supported inference
subset and its provisional status. The helper scopes and restores Python,
NumPy, CPU Torch, and CUDA RNG state. Parameters and prepared features remain
FP32; supported CUDA inference executes inside BF16 autocast.

"""
    if family_id == "esmfold":
        return """\
## Protein structure prediction

ESMFold accepts a raw sequence and returns structure tensors and confidence:

```python
import torch

model = model.cuda().eval()
with torch.inference_mode():
    output = model.infer(
        "MKTLLILAVVAAALA",
        num_recycles=4,
    )

print(output["mean_plddt"])

summary = model.fold_protein(
    "MKTLLILAVVAAALA",
    return_pdb_string=True,
)
with open("prediction.pdb", "w", encoding="utf-8") as handle:
    handle.write(summary["pdb_string"])
print(summary["plddt"], summary["ptm"])
```

FastPLMs does not expose ProteinTTT for ESMFold. The pinned folding checkpoint
does not contain a trained masked-language-model head for that objective, so
`ttt()` and TTT folding requests raise explicitly.

"""
    if family_id == "esmfold2":
        if spec.msa_conditioning is None:
            raise ValueError(f"{spec.id}: ESMFold2 MSA conditioning is undeclared")
        ttt_note = ""
        binder_note = ""
        esmc_table = _esmc_diagnostic_table(
            (
                ("eager", "Supported"),
                ("flex_attention", "Supported, numerically divergent"),
            ),
            model_id="esmc_6b",
            evidence=esmc_evidence,
        )
        if spec.msa_conditioning:
            msa_contract = """\
## Alignment-conditioning contract

This is a full 48-block ESMFold2 checkpoint. It supports both
single-sequence inference and optional MSA-conditioned inference. Typed
multichain and multimolecule inputs may attach an MSA to each applicable
protein chain.

"""
            typed_input_contract = """\
The typed interface also supports RNA, protein MSAs, modifications, covalent
bonds, and distogram conditioning."""
        else:
            msa_contract = """\
## Alignment-conditioning contract

This 24-block Fast checkpoint is inference-optimized for single-sequence
conditioning and was trained without MSA conditioning. It is not
MSA-conditioned and rejects `ProteinInput.msa` and low-level MSA-derived
features. Typed multichain and multimolecule inputs remain supported when
every protein chain uses `msa=None`. Use the corresponding full ESMFold2
checkpoint for MSA-conditioned inference. This follows the official Biohub
architecture description in [Appendix A.2.1](https://biohub.ai/papers/esm_protein.pdf).

"""
            typed_input_contract = """\
The typed interface also supports RNA, modifications, covalent bonds, and
distogram conditioning. Protein MSA inputs are not supported by this Fast
checkpoint; every protein chain must use `msa=None`."""
        if "experimental" not in spec.id:
            ttt_note = """\
## Optional folding TTT

The standard and Fast checkpoints expose opt-in folding TTT on their ESMC
backbone:

```python
adapted = model.fold_protein_ttt(
    "MSTNPKPQRKTKRNT",
    num_loops=1,
    num_sampling_steps=50,
    seed=7,
    ttt_config={"steps": 3, "batch_size": 1, "seed": 7},
)
print(adapted.ttt_metrics)
```

Entering a gradient-enabled path reloads canonical BF16 ESMC weights. TTT adds
latency and memory, can worsen a prediction, and does not calibrate confidence
or establish biological validity. Folding TTT is result-scoped: its transient
ESMC adapter modules are excluded from checkpoint state, so it is not a generic
`save_pretrained` adapter-persistence path.

"""
        else:
            ttt_note = """\
## Test-time training

This experimental checkpoint does not expose folding TTT. Use the corresponding
standard or Fast checkpoint when opt-in ESMC-backbone adaptation is required.

"""
            binder_note = f"""\
## Binder-design research example

The FastPLMs binder-design workflow uses the experimental Fast Cutoff2025
checkpoint for differentiable inversion, both experimental Cutoff2025
checkpoints as critics, and ESM++ as the sequence prior:

![FastPLMs EGFR minibinder design]({BINDER_IMAGE_URL})

```bash
python examples/binder_design_fastplms.py \\
  --target-name pd-l1 \\
  --binder-name minibinder \\
  --batch-size 4 \\
  --steps 150 \\
  --output-dir artifacts/binder-design
```

The workflow ranks candidates by mean iPTM across the approved critics after
the minibinder isoelectric-point filter. These are model-based prioritization
signals, not experimental evidence of affinity or specificity. See the
[complete workflow](https://github.com/Synthyra/FastPLMs/blob/main/docs/binder_design.md).

"""
        return f"""\
{msa_contract}
## Protein folding

The single-protein helper returns typed structure and confidence outputs:

```python
result = model.fold_protein(
    "MSTNPKPQRKTKRNT",
    num_loops=1,
    num_sampling_steps=200,
    num_diffusion_samples=1,
    seed=7,
)
pdb_text = model.result_to_pdb(result)
cif_text = model.result_to_cif(result)
print(result.ptm, result.plddt.mean().item())
```

No target structure is required. For complexes, construct the input from the
types exposed by the loaded artifact:

```python
types = model.input_types
complex_input = types.StructurePredictionInput(
    sequences=[
        types.ProteinInput(id="A", sequence="MSTNPKPQRKTKRNT"),
        types.ProteinInput(id="B", sequence="MKTIIALSYIFCLVFA"),
        types.DNAInput(id="C", sequence="ATGC"),
        types.LigandInput(id="L", smiles="O"),
    ]
)
complex_result = model.fold(
    complex_input,
    num_loops=1,
    num_sampling_steps=200,
    seed=7,
)
print(complex_result.ptm, complex_result.plddt.mean().item())
```

{typed_input_contract} The public schema recognizes
`PocketConditioning`, but the pinned official runtime discards it and hard-codes
a zero pocket feature. FastPLMs therefore rejects non-null pocket conditioning
instead of silently ignoring it. Prepared `ref_pos` values are component
reference geometries created during featurization, not target coordinates.
Predicted coordinates and confidence scores are outputs and do not establish
biochemical activity.

## Learned representation and ESMC precision

ESMFold2 applies its learned state mixture and projection as
`H: (b, l, 81, 2560) -> Z: (b, l, 256)`. Retrieve `Z` through the public
embedding API:

```python
representations = model.embed_dataset(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    batch_size=2,
    full_embeddings=True,
)
print(representations[0].tensor.shape)  # (sequence_length, 256)
```

`model.embed_dataset(..., full_embeddings=True)` returns one `(l, 256)` residue
tensor per single-chain input. It rejects complexes, ligands, MSAs,
chain-separated inputs, `cls`, and `parti` in the embedding path.

Set `esmc_precision` to `auto`, `bf16`, `fp32`, or `fp8` when loading.
`auto` always resolves to BF16. Explicit FP8 is experimental, inference-only,
and strict:

```python
model.reload_esmc(precision="fp8", device="cuda:0")
print(model.esmc_precision_status)
```

FP8 raises when the validated CUDA and Transformer Engine path is unavailable.
Canonical BF16 weights are retained, and transient quantization state is never
serialized.

The ESMC backbone uses SDPA as the recommended highest-fidelity path. Flex
Attention is supported and non-experimental but can be numerically divergent;
ESMFold2 does not advertise FlashAttention for the folding interface.

{esmc_table}

{ESMC_RELEASE_DOCUMENTATION}

## Hash-pinned CCD runtime asset

Structure preparation requires `ccd.pkl` from
`biohub/ESMFold2`. The manifest pins
its 417,306,584-byte size and SHA-256
`9ff44b1927c6b9198e38ffe0928706827a09a350c15530beeeabebfa88038fc5`
under MIT terms. This is a trusted-deserialization boundary: FastPLMs only
allows the exact manifest repository/revision snapshot link to resolve within
that repository's contained blob directory; user-supplied asset and `cache_dir`
symlinks are rejected. The loader creates a private temporary snapshot, verifies
its size and SHA-256, and unpickles only that loader-owned snapshot, closing
path-replacement and in-place source-write races. Offline execution requires the
exact cache object and never downloads a replacement.

{ttt_note}{binder_note}"""
    if allow_generic:
        return ""
    raise ValueError(f"Unsupported model-card family: {family_id!r}")


def render_model_card(
    spec: ModelSpec,
    *,
    allow_generic_family: bool = False,
    esmc_evidence: EsmcReportSet | None = None,
) -> str:
    """Render one checkpoint card whose claims are limited to manifest evidence."""

    auto_class = _preferred_auto_class(spec)
    unresolved = len(spec.fast.unresolved_files) + len(spec.official.unresolved_files)
    license_yaml = render_hub_license_yaml(spec.family)
    checkpoint_terms = render_checkpoint_terms(spec.family)
    canonical_state_provenance = ""
    tokenizer_provenance = ""
    notes = ""
    capability_summary = _capability_summary(spec)
    attention_usage = _attention_usage(spec)
    sequence_forward = _sequence_forward_usage(spec)
    embedding_usage = _embedding_usage(spec)
    task_head_usage = _task_head_usage(spec)
    peft_usage = _peft_usage(spec)
    sequence_ttt_usage = _sequence_ttt_usage(spec)
    family_usage = _family_usage_notes(
        spec,
        allow_generic=allow_generic_family,
        esmc_evidence=esmc_evidence,
    )
    local_artifact = spec.fast.repo_id.rsplit("/", maxsplit=1)[-1]
    public_input_intro = textwrap.fill(
        "Accepted inputs are "
        f"{spec.family.public_input[0].lower() + spec.family.public_input[1:]}.",
        width=79,
    )
    auto_class_intro = textwrap.fill(
        f"Supported Transformers entry points are {_code(sorted(spec.auto_map))}.",
        width=79,
    )
    recommended_attention = (
        "sdpa" if "sdpa" in spec.family.attention else spec.family.attention[0]
    )
    if spec.family.tokenizer_class is not None:
        tokenizer_provenance = f"- Tokenizer class: `{spec.family.tokenizer_class}`\n"
    if spec.canonical_state_sha256 is not None:
        canonical_state_provenance = (
            "- Canonical transformed state SHA-256: "
            f"`{spec.canonical_state_sha256}`\n"
            "- Conversion equality attestation: recorded in `provenance.json`\n"
        )
    if spec.notes and spec.family.id != "esm_plusplus":
        wrapped_notes = textwrap.fill(
            spec.notes,
            width=79,
            break_long_words=False,
            break_on_hyphens=False,
        )
        notes = f"""\
## Notes and limitations

{wrapped_notes}

"""
    auto_status = ", ".join(
        f"`{name}` = `{_auto_class_status(spec.family, name)}`" for name in sorted(spec.auto_map)
    )
    weights_allowed = str(spec.family.weights_publication_allowed).lower()
    weights_license_status = "resolved" if spec.family.weights_publication_allowed else "unresolved"
    complete_weights = str(spec.family.requires_complete_weight_publication).lower()
    return f"""---
library_name: transformers
{license_yaml}
tags:
  - protein-language-model
  - fastplms
---

{GENERATED_MARKER}

# {spec.fast.repo_id}

This checkpoint packages the FastPLMs `{spec.family.architecture}` implementation.

{public_input_intro}
{auto_class_intro}

{capability_summary}
{_installation_section(spec)}## Quick start

```python
from transformers import {auto_class}

model_id = "{spec.fast.repo_id}"
model = {auto_class}.from_pretrained(
    model_id,
    trust_remote_code=True,
    attn_implementation="{recommended_attention}",
).eval()
```

For offline validation, replace `model_id` with the manifest-built
`dist/hub/{local_artifact}` path and pass `local_files_only=True`.

{attention_usage}{sequence_forward}{embedding_usage}{task_head_usage}{peft_usage}\
{sequence_ttt_usage}{family_usage}{notes}## Runtime contract

- Public input: {spec.family.public_input}
- Advertised AutoClasses: {_code(sorted(spec.auto_map))}
- AutoClass weight status: {auto_status}
- Attention implementations: {_code(spec.family.attention)}
- Precision policies: {_precision_contract(spec.family)}
- BF16 execution: `{spec.family.bf16_execution}`
- Generation contract: `{spec.generation_contract}`
- Artifact dependency set: `{"core + structure" if spec.family.extra == "structure" else "core"}`
- Weight publication allowed: `{weights_allowed}`
- Weight license status: `{weights_license_status}`
- Redistributable: `{weights_allowed}`
- Complete weight publication required: `{complete_weights}`

## Release record

- FastPLMs weights: `{spec.fast.repo_id}`
- Runtime revision: recorded separately in the built artifact and published commit
- Source-tree and runtime-bundle SHA-256: recorded in `provenance.json`
{canonical_state_provenance}\
- Official checkpoint: `{spec.official.repo_id}`
- Artifact source: `{spec.artifact_source}`
- State transform: `{spec.family.state_transform}`
{tokenizer_provenance}- Pinned upstreams: {_code(spec.family.upstreams)}
- Release tiers: {_code(spec.family.test_tiers)}
- Unresolved required file identities: `{unresolved}`

`provenance.json` records exact file identities, conversion, source revisions,
legal texts, schema, and attestations. A nonzero unresolved count blocks release.

## Validation boundary

Declared tiers compare applicable configuration, tokenizer behavior, state,
and representative inference with the pinned reference. Metadata alone does
not claim a build passed, a backend is faster, or an output is biologically
valid.

## License

Checkpoint terms: {checkpoint_terms}. The Hub model-card identifier is
`{spec.family.hub_license}`. Applicable source licenses, notices, attribution,
and conversion records are distributed with the local artifact. Review them
before use.
"""


def expected_outputs(
    root: Path,
    registry: ModelRegistry,
    *,
    esmc_evidence: EsmcReportSet | None = None,
) -> dict[Path, str]:
    """Return every generated path and its deterministic UTF-8 content."""

    output = {
        root / "docs" / "generated" / "support.md": render_support(registry),
        root / "docs" / "generated" / "capability_evidence.md": render_capability_evidence(
            registry,
            esmc_evidence=esmc_evidence,
        ),
    }
    for spec in registry.values():
        output[root / "model_cards" / f"{spec.id}.md"] = render_model_card(
            spec,
            esmc_evidence=esmc_evidence,
        )
    return output


def synchronize(
    root: Path,
    *,
    check: bool,
    esmc_report_root: Path | None = None,
    require_esmc_release_evidence: bool = False,
) -> list[str]:
    """Write generated files or return descriptions of stale files."""

    registry = get_model_registry()
    esmc_evidence = None
    if esmc_report_root is not None or require_esmc_release_evidence:
        selected_root = esmc_report_root
        if selected_root is None:
            selected_root = Path(
                os.environ.get(
                    "FASTPLMS_DIAGNOSTIC_REPORTS",
                    "artifacts/diagnostics/esmc",
                )
            )
        if not selected_root.is_absolute():
            selected_root = root / selected_root
        esmc_evidence = load_esmc_report_set(
            selected_root,
            registry,
            source_root=root,
        )
    outputs = expected_outputs(root, registry, esmc_evidence=esmc_evidence)
    failures: list[str] = []
    for path, content in outputs.items():
        rendered = content.rstrip() + "\n"
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == rendered:
            continue
        if check:
            failures.append(f"stale or missing generated file: {path.relative_to(root)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8", newline="\n")

    expected_cards = {path.resolve() for path in outputs if path.parent.name == "model_cards"}
    for path in sorted((root / "model_cards").glob("*.md")):
        if path.name == "README.md" or path.resolve() in expected_cards:
            continue
        try:
            generated = GENERATED_MARKER in path.read_text(encoding="utf-8")
        except OSError:
            generated = False
        if generated and check:
            failures.append(f"stale generated model card: {path.relative_to(root)}")
        elif generated:
            path.unlink()
    return failures


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--esmc-report-root",
        type=Path,
        help=(
            "strictly validate and render one explicit complete 30-record schema-v3 "
            "ESMC release-evidence set"
        ),
    )
    parser.add_argument(
        "--require-esmc-release-evidence",
        action="store_true",
        help=(
            "require release evidence from --esmc-report-root, "
            "FASTPLMS_DIAGNOSTIC_REPORTS, or artifacts/diagnostics/esmc"
        ),
    )
    arguments = parser.parse_args(argv)
    try:
        failures = synchronize(
            arguments.source_root.resolve(),
            check=arguments.check,
            esmc_report_root=arguments.esmc_report_root,
            require_esmc_release_evidence=arguments.require_esmc_release_evidence,
        )
    except EsmcReportError as error:
        print(f"invalid ESMC release evidence: {error}")
        return 1
    if failures:
        for failure in failures:
            print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
