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
BINDER_IMAGE_URL = (
    "https://raw.githubusercontent.com/Synthyra/FastPLMs/main/"
    "docs/assets/egfr_fastplms_binder_design.png"
)


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
        "This file is generated from `src/fastplms/models.toml`. A listed capability is",
        "selectable. Strict-parity exceptions are documented in the checkpoint cards.",
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


def _sequence_forward_usage(spec: ModelSpec) -> str:
    if spec.family.id not in {"esm2", "esm_plusplus", "dplm", "ankh"}:
        return ""
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
    if spec.family.id not in {
        "ankh",
        "dplm",
        "dplm2",
        "e1",
        "esm2",
        "esm3",
        "esm_plusplus",
    }:
        return ""
    return """\
## Dataset embeddings

The shared embedding API accepts sequences, `(id, sequence)` pairs,
`EmbeddingInput` records, or a FASTA path. Results preserve order and duplicate
identifiers:

```python
result = model.embed_dataset(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    batch_size=2,
    pooling=("mean", "std"),
)

for record in result:
    print(record.id, record.sequence, record.tensor.shape)
```

Set `full_embeddings=True` for one residue tensor with shape `(l, d)` per
sequence. Set `output` to a directory for transactional safetensors or choose
`format="sqlite"` for batch-level commits and exact resume. Pooling excludes
boundary, padding, and other non-biological positions.

For a long FASTA run, stream completed batches into SQLite:

```python
persisted = model.embed_dataset(
    "proteins.fasta",
    batch_size=64,
    pooling=("mean",),
    output="protein-embeddings.sqlite",
    format="sqlite",
    resume=True,
)
```

Resume verifies the input order, model state, tokenizer policy, backend, dtype,
and pooling configuration. It never appends incompatible records to an
existing run.

"""


def _family_usage_notes(spec: ModelSpec) -> str:
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

"""
    if family_id == "esm_plusplus":
        return """\
## ESMC behavior

This artifact exposes the Biohub ESMC sequence encoder and masked-language-model
head through Transformers. It is also the language-model family used by
ESMFold2. Request SDPA when exact pinned Biohub inference parity is required;
the provenance section records backend-specific validation boundaries for this
checkpoint.

"""
    if family_id == "esm3":
        return f"""\
## Sequence and multimodal inference

ESM3 owns its sequence preparation because its forward pass can combine
sequence, structure, and function tracks:

```python
import torch

batch = model.tokenize_sequences(
    ["MKTAYIAKQ", "GGGG"],
    device=model.device,
)
with torch.inference_mode():
    output = model(**batch)

print(output.logits.shape)
print(output.structure_logits.shape)
print(output.function_logits.shape)
```

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

"""
    if family_id == "dplm2":
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

"""
    if family_id == "ankh":
        return f"""\
## Encoder and sequence-to-sequence use

`AutoModel` loads the optimized ANKH encoder. The official-compatible decoder
and language-model head are available through `AutoModelForSeq2SeqLM`:

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
seq2seq = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
batch = tokenizer("MSTNPKPQRKTKRNT", return_tensors="pt")

with torch.inference_mode():
    generated_ids = seq2seq.generate(**batch, max_new_tokens=16)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
```

The separately named masked-language-model extension is a FastPLMs extension,
not an official ANKH masked-language-model equivalent. ANKH artifacts retain
CC BY-NC-SA 4.0 terms.

"""
    if family_id == "boltz2":
        return f"""\
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
)
model.save_as_cif(output, "prediction.cif")

print(output.sample_atom_coords.shape)
print(output.plddt, output.ptm, output.iptm)
```

Boltz2 is provisional in FastPLMs 1.0. Configuration, declared inference-core
weights, feature preparation, and seeded execution are tested, but
native-environment BF16 end-to-end inference does not yet meet the fixed
numerical-equivalence limits.

"""
    if family_id == "esmfold":
        return f"""\
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
        ttt_note = ""
        binder_note = ""
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
or establish biological validity.

"""
        else:
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

For complexes, construct the model's `StructurePredictionInput` with explicit
protein, DNA, RNA, ligand, and MSA objects. Confidence scores are model outputs
and do not establish biochemical activity.

## Learned representation and ESMC precision

ESMFold2 combines the ordered 81 ESMC-6B states `H: (b, l, 81, 2560)` with the
checkpoint's learned projection:

```python
Z = model.project_esmc_hidden_states(H)  # Z: (b, l, 256)
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

{ttt_note}{binder_note}"""
    raise ValueError(f"Unsupported model-card family: {family_id!r}")


def render_model_card(spec: ModelSpec) -> str:
    """Render one checkpoint card whose claims are limited to manifest evidence."""

    auto_class = _preferred_auto_class(spec)
    unresolved = len(spec.fast.unresolved_files) + len(spec.official.unresolved_files)
    license_yaml = render_hub_license_yaml(spec.family)
    checkpoint_terms = render_checkpoint_terms(spec.family)
    tokenizer_provenance = ""
    notes = ""
    sequence_forward = _sequence_forward_usage(spec)
    embedding_usage = _embedding_usage(spec)
    family_usage = _family_usage_notes(spec)
    if spec.family.tokenizer_class is not None:
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

## Quick start

```python
from transformers import {auto_class}

model_id = "{spec.fast.repo_id}"
model = {auto_class}.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/<model>` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default or request one of
{_code(spec.family.attention)} with `attn_implementation`.
The BF16 execution policy is `{spec.family.bf16_execution}`:
{_bf16_execution_description(spec.family)}.

{sequence_forward}{embedding_usage}{family_usage}{notes}## Runtime contract

- Input mode: `{spec.family.tokenizer_mode}`
- Advertised AutoClasses: {_code(sorted(spec.auto_map))}
- Attention implementations: {_code(spec.family.attention)}
- Precision policies: {_precision_contract(spec.family)}
- BF16 execution: `{spec.family.bf16_execution}`
- Generation contract: `{spec.generation_contract}`
- Optional dependency group: `{spec.family.extra}`

## Provenance

- FastPLMs checkpoint: `{spec.fast.repo_id}@{spec.fast.revision}`
- Official checkpoint: `{spec.official.repo_id}@{spec.official.revision}`
- Artifact source: `{spec.artifact_source}`
- State transform: `{spec.family.state_transform}`
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
