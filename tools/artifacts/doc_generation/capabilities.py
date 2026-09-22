"""Derive documented capabilities and executable evidence selectors from the manifest."""

from __future__ import annotations

from dataclasses import dataclass

from fastplms.registry import ModelFamily, ModelRegistry
from tools.artifacts.doc_generation.esmc_evidence import (
    ESMC_MEASURED_BACKENDS,
)

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


CAPABILITY_EVIDENCE_SELECTORS: dict[str, EvidenceSelector] = {
    "cpu:esmfold2-small": EvidenceSelector(
        tier="check",
        targets=("tests/unit/test_esmfold2_small.py", "tests/unit/test_esmc_native_conversion.py"),
        scope=(
            "300M and 600M configuration, dependency conversion, confidence absence, "
            "and FP8 rejection. Inference evidence is separately scoped in "
            "docs/validation/esmfold2_small.md."
        ),
    ),
    "cpu:autoclass-runtime": EvidenceSelector(
        tier="cpu_contract",
        targets=(
            "tests/cpu/test_autoclass_evidence_matrix.py::"
            "test_autoclass_runtime_evidence_matrix_exactly_matches_all_45_entries",
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
                "modifications, and bonds; pocket and distogram requests fail closed"
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
                ("cpu:structure-contracts", "cpu:esmfold2-small")
                if spec.backbone_model is not None
                else (
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
