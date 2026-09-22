"""Compose checkpoint cards from manifest metadata and evidence-backed sections."""

from __future__ import annotations

import textwrap
from pathlib import Path

from fastplms.registry import ModelSpec
from tools.artifacts.doc_generation.card_metadata import (
    GENERATED_MARKER,
    _auto_class_status,
    _code,
    _installation_section,
    _precision_contract,
)
from tools.artifacts.doc_generation.confidence_rendering import (
    _confidence_adaptation_section,
    _confidence_research_section,
    _live_confidence_section,
)
from tools.artifacts.doc_generation.esmc_evidence import (
    EsmcReportSet,
)
from tools.artifacts.doc_generation.folding_rendering import (
    _esmfold2_folding_speed_section,
)
from tools.artifacts.doc_generation.usage_examples import (
    _attention_usage,
    _embedding_usage,
    _esmfold2_quick_start,
    _family_usage_notes,
    _peft_usage,
    _sequence_forward_usage,
    _sequence_ttt_usage,
    _task_head_usage,
)
from tools.artifacts.license_metadata import render_checkpoint_terms, render_hub_license_yaml


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


def _model_title(spec: ModelSpec) -> str:
    """Return a readable checkpoint name for the card heading."""

    repository_name = spec.fast.repo_id.rsplit("/", maxsplit=1)[-1]
    if spec.family.id == "esm_plusplus":
        scale = repository_name.removeprefix("ESMplusplus_")
        return f"ESM++ {scale if scale == '6B' else scale.title()}"
    if spec.family.id == "ankh":
        name, scale = repository_name.rsplit("_", maxsplit=1)
        return f"{name}-{scale.upper() if scale == 'xl' else scale.title()}"
    scale_names = {
        "base": "Base",
        "large": "Large",
        "small": "Small",
        "xl": "XL",
    }
    return " ".join(scale_names.get(part.lower(), part) for part in repository_name.split("_"))


def _model_overview(spec: ModelSpec) -> str:
    """Introduce the checkpoint in task-oriented prose."""

    public_input = spec.family.public_input[0].lower() + spec.family.public_input[1:]
    if spec.confidence_adaptation is None:
        package_description = (
            f"packages the `{spec.official.repo_id}` checkpoint with the FastPLMs runtime"
        )
    else:
        package_description = (
            f"packages the `{spec.official.repo_id}` checkpoint with the FastPLMs runtime "
            "and a Synthyra-adapted native confidence head"
        )
    overview = (
        f"`{spec.fast.repo_id}` {package_description} for Hugging Face Transformers. "
        f"It accepts {public_input}."
    )
    entry_points = (
        "The repository uses the standard Transformers loading interface with "
        "`trust_remote_code=True`. See Technical details for each registered class and "
        "whether its weights come from the checkpoint."
    )
    paragraphs = [
        textwrap.fill(overview, width=79, break_long_words=False, break_on_hyphens=False),
        textwrap.fill(
            entry_points,
            width=79,
            break_long_words=False,
            break_on_hyphens=False,
        ),
    ]
    if {
        "AutoModelForSequenceClassification",
        "AutoModelForTokenClassification",
    }.issubset(spec.auto_map):
        paragraphs.append(
            textwrap.fill(
                "The sequence- and token-classification classes reuse the pretrained "
                "backbone, but their task heads are newly initialized. Fine-tune those "
                "heads before interpreting their logits as predictions.",
                width=79,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "## Model overview\n\n" + "\n\n".join(paragraphs) + "\n\n"


def render_model_card(
    spec: ModelSpec,
    *,
    allow_generic_family: bool = False,
    esmc_evidence: EsmcReportSet | None = None,
    evidence_root: Path | None = None,
) -> str:
    """Render one checkpoint card whose claims are limited to manifest evidence."""

    auto_class = _preferred_auto_class(spec)
    unresolved = len(spec.fast.unresolved_files) + len(spec.official.unresolved_files)
    license_yaml = render_hub_license_yaml(spec.family)
    checkpoint_terms = render_checkpoint_terms(spec.family)
    canonical_state_record = ""
    tokenizer_details = ""
    notes = ""
    model_overview = _model_overview(spec)
    esmfold2_quick_start = _esmfold2_quick_start(spec)
    confidence_adaptation = _confidence_adaptation_section(spec, evidence_root)
    confidence_research = _live_confidence_section(spec) + _confidence_research_section(spec, evidence_root)
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
        folding_speed=_esmfold2_folding_speed_section(spec, evidence_root),
    )
    local_artifact = spec.fast.repo_id.rsplit("/", maxsplit=1)[-1]
    recommended_attention = "sdpa" if "sdpa" in spec.family.attention else spec.family.attention[0]
    generic_quick_start = (
        ""
        if spec.family.id == "esmfold2"
        else f"""## Quick start

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
`dist/hub/{local_artifact}` path. Pass `local_files_only=True`.

"""
    )
    if spec.family.tokenizer_class is not None:
        tokenizer_details = f"- Tokenizer class: `{spec.family.tokenizer_class}`\n"
    if spec.canonical_state_sha256 is not None:
        canonical_state_record = (
            "- Canonical transformed state identity: recorded in `source-record.json`\n"
            "- Conversion equality attestation: recorded in `source-record.json`\n"
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
    if spec.confidence_adaptation is not None:
        validation_scope = (
            "The adapted confidence head passed held-out quality checks on short "
            "monomers and dimers. Focused Transformers reload, confidence ranges, "
            "seeded coordinate equality, and two-chain CIF checks also passed. "
            "These results are not a full structure benchmark."
        )
    elif spec.backbone_model is not None:
        validation_scope = (
            "ESMFold2-300 passed a Docker BF16 reference comparison on one compact "
            "Protein G sequence. ESMFold2-600 is not inference-validated. Both have "
            "configuration, weight identity, and artifact loading checks. This is "
            "bounded checkpoint evidence, not a full structure benchmark result."
        )
    elif "compliance" in spec.family.test_tiers:
        validation_scope = (
            "Release validation includes the `compliance` tier. Its evidence identifies "
            "the checkpoint, backend, dtype, hardware, inputs, and reference revision."
        )
    else:
        validation_scope = (
            "Boltz2 remains provisional and does not declare the `compliance` tier. "
            "Its structure checks are not parity claims."
        )
    return f"""---
library_name: transformers
{license_yaml}
tags:
  - protein-language-model
  - fastplms
---

{GENERATED_MARKER}

# {_model_title(spec)}

{esmfold2_quick_start}{confidence_adaptation}{model_overview}{_installation_section(spec)}{generic_quick_start}\
{attention_usage}{sequence_forward}{embedding_usage}{task_head_usage}{peft_usage}\
{sequence_ttt_usage}{family_usage}{confidence_research}{notes}## Technical details

- Inputs: {spec.family.public_input}
- Transformers classes: {_code(sorted(spec.auto_map))}
- Checkpoint weights: {auto_status}
- Attention backends: {_code(spec.family.attention)}
- Precision: {_precision_contract(spec.family, spec)}
- BF16 execution: `{spec.family.bf16_execution}`
- Generation contract: `{spec.generation_contract}`
- Dependencies: `{"core + structure" if spec.family.extra == "structure" else "core"}`
- Weight publication allowed: `{weights_allowed}`
- Weight license status: `{weights_license_status}`
- Redistributable: `{weights_allowed}`
- Complete weight publication required: `{complete_weights}`

## Validation and sources

FastPLMs pins the checkpoint, upstream source revisions, state transformation,
and required files in `models.toml`. Built artifacts record exact source
identities and conversion details in `source-record.json`.

- FastPLMs checkpoint: `{spec.fast.repo_id}`
- Runtime revision: recorded separately in the built artifact and published commit
- Runtime source identities: recorded in `source-record.json`
{canonical_state_record}\
- Official checkpoint: `{spec.official.repo_id}`
- Artifact source: `{spec.artifact_source}`
- State transform: `{spec.family.state_transform}`
{tokenizer_details}- Pinned upstreams: {_code(spec.family.upstreams)}
- Release tiers: {_code(spec.family.test_tiers)}
- Unresolved required file identities: `{unresolved}`

{textwrap.fill(validation_scope, width=79, break_long_words=False, break_on_hyphens=False)}

Declared tiers compare configuration, tokenizer behavior, state, and
representative inference with the pinned reference. A nonzero unresolved count
blocks release. Metadata alone does not show that a build passed, that a backend
is faster, or that an output is biologically valid.

## License

Checkpoint terms: {checkpoint_terms}. The Hub model-card identifier is
`{spec.family.hub_license}`. The local artifact contains applicable source
licenses, notices, attribution, and conversion records. Review them before use.
"""
