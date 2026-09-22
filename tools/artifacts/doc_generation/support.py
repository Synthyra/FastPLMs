"""Render model support tables and the executable capability evidence catalog."""

from __future__ import annotations

from collections.abc import Iterable

from fastplms.registry import ModelFamily, ModelRegistry
from tools.artifacts.doc_generation.capabilities import (
    CAPABILITY_EVIDENCE_SELECTORS,
    CURATED_EXAMPLE_CPU_CASES,
    EMBEDDING_CAPABILITY_ROWS,
    FAMILY_DOCUMENTATION,
    GENERATION_CAPABILITY_ROWS,
    STRUCTURE_CAPABILITY_ROWS,
    CapabilityEvidenceRow,
    _esmfold2_structure_capability_rows,
    attention_backend_evidence_keys,
    autoclass_evidence_keys,
)
from tools.artifacts.doc_generation.card_metadata import (
    GENERATED_MARKER,
    _auto_class_status,
    _code,
    _hub_license_label,
    _precision_contract,
    _table_row,
    _tokenizer_class_label,
)
from tools.artifacts.doc_generation.esmc_evidence import (
    EsmcReportSet,
)
from tools.artifacts.doc_generation.esmc_rendering import (
    _render_esmc_capability_evidence,
)


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


def _autoclass_workflow_example(family: ModelFamily, auto_class: str) -> str:
    if family.id == "esm2" and auto_class in {
        "AutoModelForMaskedLM",
        "AutoModelForSequenceClassification",
        "AutoModelForTokenClassification",
    }:
        return "../../examples/task_heads.py"
    return FAMILY_DOCUMENTATION[family.id][1]


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
            "| Family | Attention | Auto order | Precision | BF16 execution | Extra | Reference |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        )
    )
    for family in registry.families.values():
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{family.id}`",
                    _code(family.attention),
                    # An empty order means the family rejects attn_implementation="auto".
                    _code(family.attention_auto_order) if family.attention_auto_order else "none",
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
