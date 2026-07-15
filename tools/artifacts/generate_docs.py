"""Generate model support data and model cards from the typed manifest."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from fastplms.registry import ModelFamily, ModelRegistry, ModelSpec, get_model_registry
from tools.artifacts.license_metadata import (
    render_checkpoint_terms,
    render_hub_license_yaml,
)

GENERATED_MARKER = "<!-- Generated from src/fastplms/models.toml. Do not edit. -->"


def _code(values: Iterable[str]) -> str:
    return ", ".join(f"`{value}`" for value in values)


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


def _bf16_execution_description(family: ModelFamily) -> str:
    if family.bf16_execution == "static_parameters":
        return "parameters loaded directly in BF16"
    if family.bf16_execution == "fp32_parameters_autocast":
        return "FP32 parameters with CUDA BF16 autocast"
    raise ValueError(f"Unsupported BF16 execution policy: {family.bf16_execution!r}")


def _tokenizer_class_label(family: ModelFamily) -> str:
    if family.tokenizer_class is None:
        return "`n/a`"
    return f"`{family.tokenizer_class}`"


def render_support(registry: ModelRegistry) -> str:
    """Render the complete support matrix without importing model code."""

    lines = [
        GENERATED_MARKER,
        "",
        "# Model support",
        "",
        "This file is generated from `src/fastplms/models.toml`. A listed capability",
        "is a release contract and must pass its declared live compliance tier.",
        "",
        "## Families",
        "",
        "| Family | Architecture | Checkpoints | Input | Tokenizer class | AutoClasses | "
        "Attention | Precision | BF16 execution | Extra | Reference | Checkpoint terms | "
        "Hub license | Tiers |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
                    f"`{family.tokenizer_mode}`",
                    _tokenizer_class_label(family),
                    _code(sorted(family.auto_map)),
                    _code(family.attention),
                    _precision_contract(family),
                    f"`{family.bf16_execution}`",
                    f"`{family.extra}`",
                    f"`{family.reference_container}`",
                    family.checkpoint_license.replace("|", "\\|"),
                    _hub_license_label(family),
                    _code(family.test_tiers),
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
            "Artifact source | State transform | Generation contract | Unresolved files |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: |",
        )
    )
    for spec in registry.values():
        fast_url = f"https://huggingface.co/{spec.fast.repo_id}/tree/{spec.fast.revision}"
        official_url = (
            f"https://huggingface.co/{spec.official.repo_id}/tree/{spec.official.revision}"
        )
        unresolved = len(spec.fast.unresolved_files) + len(spec.official.unresolved_files)
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


def _family_usage_notes(spec: ModelSpec) -> str:
    if spec.family.id != "esmfold2":
        return ""
    return """\
## Learned representation and FP8

ESMFold2 combines the ordered 81 ESMC-6B states `H: (b, l, 81, 2560)`
with the checkpoint's learned projection:

```python
Z = model.project_esmc_hidden_states(H)  # Z: (b, l, 256)
```

`model.embed_dataset(..., full_embeddings=True)` returns one residue tensor
with shape `(l, 256)` per single-chain input. Residue-statistic poolers are
supported; `cls`, `parti`, complexes, ligands, MSAs, and chain-separated
embedding inputs are rejected.

Set `esmc_precision` to `auto`, `bf16`, `fp32`, or `fp8` when loading. The
runtime can be rebuilt explicitly with
`model.reload_esmc(precision=..., device=...)`; `model.esmc_precision_status`
records the requested and resolved precision, reason, device, and Transformer
Engine version. `auto` always resolves to BF16. Explicit `fp8` is an
experimental, inference-only opt-in and raises when the path is unavailable.
Canonical BF16 weights are retained, and transient Transformer Engine
quantization state is never serialized.

"""


def render_model_card(spec: ModelSpec) -> str:
    """Render one checkpoint card whose claims are limited to manifest evidence."""

    auto_class = _preferred_auto_class(spec)
    unresolved = len(spec.fast.unresolved_files) + len(spec.official.unresolved_files)
    artifact_directory = spec.fast.repo_id.split("/", maxsplit=1)[1]
    license_yaml = render_hub_license_yaml(spec.family)
    checkpoint_terms = render_checkpoint_terms(spec.family)
    tokenizer_load = ""
    tokenizer_provenance = ""
    notes = ""
    family_usage = _family_usage_notes(spec)
    if spec.family.tokenizer_class is not None:
        tokenizer_load = f"""
The paired custom tokenizer is loaded through the same pinned artifact:

```python
from transformers import AutoTokenizer

artifact_path = "dist/hub/{artifact_directory}"
tokenizer = AutoTokenizer.from_pretrained(
    artifact_path,
    local_files_only=True,
    trust_remote_code=True,
)
```
"""
        tokenizer_provenance = f"- Tokenizer class: `{spec.family.tokenizer_class}`\n"
    if spec.notes:
        notes = f"""\
## Notes and limitations

{spec.notes}

"""
    return f"""---
library_name: transformers
{license_yaml}
tags:
  - protein-language-model
  - fastplms
---

{GENERATED_MARKER}

# {spec.fast.repo_id}

This checkpoint uses the FastPLMs `{spec.family.architecture}` implementation.
Its input mode is `{spec.family.tokenizer_mode}` and its advertised AutoClasses
are {_code(sorted(spec.auto_map))}.

## Load

```python
from transformers import {auto_class}

artifact_path = "dist/hub/{artifact_directory}"
model = {auto_class}.from_pretrained(
    artifact_path,
    local_files_only=True,
    trust_remote_code=True,
)
```
{tokenizer_load}

After publication, replace `artifact_path` with the Hub repository ID and pass
the immutable revision of the published FastPLMs 1.0 artifact. The checkpoint
revision below identifies the source weights; it is not a claim that the
generated artifact already exists at that Hub revision.

Leave attention unspecified for the Transformers default or request one of
{_code(spec.family.attention)} with `attn_implementation`.
The BF16 execution policy is `{spec.family.bf16_execution}`:
{_bf16_execution_description(spec.family)}.

{family_usage}{notes}## Provenance

- FastPLMs checkpoint: `{spec.fast.repo_id}@{spec.fast.revision}`
- Official checkpoint: `{spec.official.repo_id}@{spec.official.revision}`
- Artifact source: `{spec.artifact_source}`
- State transform: `{spec.family.state_transform}`
- Generation contract: `{spec.generation_contract}`
- BF16 execution: `{spec.family.bf16_execution}`
{tokenizer_provenance}- Pinned upstreams: {_code(spec.family.upstreams)}
- Reference container: `{spec.family.reference_container}`
- Release tiers: {_code(spec.family.test_tiers)}
- Unresolved required file identities: `{unresolved}`

The local artifact records exact file identities, conversion provenance, source
revisions, and legal texts in `provenance.json`. A nonzero unresolved count is a
release blocker.

## Validation boundary

For tiers declared by the manifest, the release contract compares applicable
semantic configuration, tokenizer behavior, state keys, shapes, dtypes,
values, aliases, and representative inference with the pinned official
implementation. This metadata does not by itself claim that a particular build
passed, that one backend is faster, or that an output has biological or
therapeutic validity.

## License

Checkpoint terms: {checkpoint_terms}. The Hub model-card identifier is
`{spec.family.hub_license}`. Applicable source licenses, notices, attribution,
and conversion records are distributed with the local artifact. Review them
before use.
"""


def expected_outputs(root: Path, registry: ModelRegistry) -> dict[Path, str]:
    """Return every generated path and its deterministic UTF-8 content."""

    output = {root / "docs" / "generated" / "support.md": render_support(registry)}
    for spec in registry.values():
        output[root / "model_cards" / f"{spec.id}.md"] = render_model_card(spec)
    return output


def synchronize(root: Path, *, check: bool) -> list[str]:
    """Write generated files or return descriptions of stale files."""

    registry = get_model_registry()
    outputs = expected_outputs(root, registry)
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
    arguments = parser.parse_args(argv)
    failures = synchronize(arguments.source_root.resolve(), check=arguments.check)
    if failures:
        for failure in failures:
            print(failure)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
