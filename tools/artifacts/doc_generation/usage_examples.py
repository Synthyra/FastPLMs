"""Render model-family usage examples while preserving their native interfaces."""

from __future__ import annotations

import textwrap

from fastplms.registry import ModelSpec
from tools.artifacts.doc_generation.capabilities import (
    EMBEDDING_FAMILIES,
    SEQUENCE_TTT_AUTO_CLASSES,
)
from tools.artifacts.doc_generation.card_metadata import (
    _code,
)
from tools.artifacts.doc_generation.esmc_evidence import (
    EsmcReportSet,
)
from tools.artifacts.doc_generation.esmc_rendering import (
    ESMC_RELEASE_DOCUMENTATION,
    _esmc_diagnostic_table,
)

BINDER_IMAGE_URL = (
    "https://raw.githubusercontent.com/Synthyra/FastPLMs/main/"
    "docs/assets/egfr_fastplms_binder_design.png"
)


ESMC_SAE_EXAMPLES = {
    "esmc_small": ("biohub/ESMC-300M-sae-layer23-k64-codebook65536", 23),
    "esmc_large": ("biohub/ESMC-600M-sae-layer27-k64-codebook65536", 27),
    "esmc_6b": ("biohub/ESMC-6B-sae-layer60-k64-codebook16384", 60),
}


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
    elif spec.family.id in {"esmfold", "esmfold2"}:
        preparation = """\
sequences = ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"]
batch = sequence_model.prepare_classifier_inputs(sequences)
biological = batch["attention_mask"].bool()
"""
    else:
        preparation = """\
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
sequences = ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"]
batch = tokenizer(sequences, padding=True, return_tensors="pt")
biological = batch["attention_mask"].bool()
for special_id in tokenizer.all_special_ids:
    biological &= batch["input_ids"].ne(special_id)
"""

    tokenizer_import = (
        ""
        if spec.family.id in {"e1", "esmfold", "esmfold2"}
        else "from transformers import AutoTokenizer\n"
    )
    task_description = (
        "The sequence and token prediction AutoClasses use the checkpoint backbone "
        "and create a new, untrained `classifier`. Sequence labels have shape `(b,)`. "
        "Residue labels have shape `(b, l)` and use `-100` outside biological positions."
    )
    if spec.family.id in {"esmfold", "esmfold2"}:
        task_description += (
            " The folding trunk is skipped. The classifier uses the checkpoint's learned "
            "pLM state mixture and projection, followed by one trainable transformer probe."
        )
    task_description = textwrap.fill(
        task_description,
        width=79,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return f"""\
## Downstream prediction

{task_description}

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
            "This checkpoint advertises a classification head. Save the separately "
            "trained `classifier` with the adapter."
        )
    else:
        model_name = "model"
        task_import = ""
        task_type = ""
        modules_to_save = ""
        persistence = (
            "This checkpoint has no advertised classifier. Supply the task objective "
            "and preserve any new head through `modules_to_save`."
        )
    return f"""\
## PEFT fine-tuning

Install the training dependencies. Then attach LoRA to the loaded checkpoint:

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
can use PEFT. The ESM2-specific shipped CLI is an example, not a
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
adapters. Base checkpoint weights stay frozen:

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

Saved adapters retain their deterministic reset state. TTT adds latency and
memory, can worsen an output, and does not show biological function.

"""


def _attention_usage(spec: ModelSpec) -> str:
    recommended = "sdpa" if "sdpa" in spec.family.attention else spec.family.attention[0]
    declared = textwrap.fill(
        f"Available backends are {_code(spec.family.attention)}. Requesting an "
        "unavailable backend raises instead of silently changing implementation.",
        width=79,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return f"""\
## Attention backends

The quick start uses `{recommended}`.

{declared}

`output_attentions=True` can use the documented one-call eager fallback to
materialize attention tensors. The configured backend does not change.

"""


def _sequence_forward_usage(spec: ModelSpec) -> str:
    if spec.family.id not in {"esm2", "esm_plusplus", "dplm", "ankh"}:
        return ""
    if spec.family.id == "ankh":
        return f"""\
## Tokenization and forward inference

`{spec.fast.repo_id}` contains the complete encoder-decoder checkpoint.
`AutoModel` loads the encoder without the decoder. `AutoModelForSeq2SeqLM`
loads the encoder, decoder, cross-attention, and language-model head.

Use the tokenizer from the loaded model. This keeps tokenizer files, revision,
offline/cache policy, and ANKH's residue-aware pre-tokenizer aligned. Pass raw
protein strings without residue spaces:

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

Load the tokenizer from the same artifact as the model. The attention mask
shows padding explicitly:

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

Dataset embeddings use the final encoder state by default. Select a native
encoder layer directly:

```python
encoder_result = model.embed_dataset(
    ["MSTNPKPQRKTKRNT"],
    hidden_state_source="encoder",
    hidden_state_index=-1,
    full_embeddings=True,
)
print(encoder_result[0].tensor.shape)  # (l, d)
```

Decoder representations require `AutoModelForSeq2SeqLM` and one aligned decoder
input. ANKH does not create a shifted target:

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

The shared embedding mixin keeps input order and biological-position masking.
It accepts sequences, identified records, mappings, or a FASTA path:

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
bounded-memory storage. Resume checks input order, model state, tokenizer
policy, backend, dtype, and pooling configuration before it appends data.

"""


def _esmfold2_quick_start(spec: ModelSpec) -> str:
    """Return the short two-chain folding example for ESMFold2 cards."""

    if spec.family.id != "esmfold2":
        return ""
    is_small_variant = spec.id in {"esmfold2_300", "esmfold2_600"}
    sampling_steps = "        num_sampling_steps=15,\n" if is_small_variant else ""
    step_note = (
        "This example uses 15 diffusion steps, matching the experimental config."
        if is_small_variant
        else "The example omits `num_sampling_steps` and uses the model default."
    )
    confidence_note = (
        "This variant has a Synthyra-adapted native confidence head and returns pLDDT, "
        "PAE, pTM, and iPTM fields. Confidence calculation is optional."
        if spec.confidence_adaptation is not None
        else (
            "The confidence fields are unavailable because this experimental variant "
            "has a disabled confidence head."
            if is_small_variant
            else "This variant has an enabled confidence head and returns confidence fields."
        )
    )
    confidence_note = textwrap.fill(
        confidence_note,
        width=79,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return f"""\
## Quick start

Load the published model, fold two protein chains together, and write an mmCIF
file. {step_note}

```python
from pathlib import Path

import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "{spec.fast.repo_id}",
    trust_remote_code=True,
    dtype=torch.float32,
    device_map="cuda",
    esmc_precision="bf16",
    attn_implementation="sdpa",
).eval()
model.set_chunk_size(32)

types = model.input_types
complex_input = types.StructurePredictionInput(
    sequences=[
        types.ProteinInput(id="A", sequence="MSTNPKPQRKTKRNT"),
        types.ProteinInput(id="B", sequence="MKTIIALSYIFCLVFA"),
    ]
)
with torch.inference_mode():
    result = model.fold(
        complex_input,
        num_loops=3,
{sampling_steps}        num_diffusion_samples=1,
        seed=17,
        verbose=True,
    )
Path("complex.cif").write_text(model.result_to_cif(result), encoding="utf-8")
```

Set `verbose=False` to silence the folding progress display. {confidence_note}

"""


def _family_usage_notes(
    spec: ModelSpec,
    *,
    allow_generic: bool = False,
    esmc_evidence: EsmcReportSet | None = None,
    folding_speed: str = "",
) -> str:
    family_id = spec.family.id
    model_id = spec.fast.repo_id
    if family_id == "esm2":
        return f"""\
## Masked language modeling and contacts

Use the masked-language-model AutoClass when you need logits:

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

Contact prediction creates attention maps. Do not enable it in a high-throughput
embedding path unless you need these maps.

Plain `AutoModel` omits the optional ESM pooler because this masked-language-
model checkpoint has no trained pooler weights. Pass `add_pooling_layer=True`
only when you intend to initialize and train that head.

"""
    if family_id == "esm_plusplus":
        fp8_usage = "FP8 is restricted to ESMC-6B; smaller ESM++ models use BF16."
        if spec.id == "esmc_6b":
            fp8_usage = f"""### Experimental FP8 inference

The default uses checkpoint BF16 behavior. FP8 is an explicit experimental
inference option for ESMC-6B:

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

FP8 forward calls require `torch.inference_mode()`. The model pads the sequence
dimension to a multiple of 16. Transformer Engine converts supported linear
layers. The call fails if the dependency, compatible CUDA hardware, or complete
conversion set is unavailable. It does not silently use BF16. FP8 does not
claim numerical parity.

"""
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

This artifact provides the Biohub ESMC sequence encoder and masked-language-
model head through Transformers. ESMFold2 also uses this language-model family.
SDPA is the default and gives the highest numerical fidelity. Flex Attention and
FlashAttention 3 are supported non-experimental backends. Their BF16 arithmetic
can differ numerically from SDPA. These differences give diagnostic warnings,
not strict-parity failures. Dispatch, masks, finite outputs, shapes, and large
biological disagreements remain hard gates.

The current GH200/aarch64 release environment validates eager, SDPA, and Flex.
Flash requests raise because compatible locked kernels are unavailable on this
platform.

When `sequence_id` is supplied, it controls ESMC attention groups and padding.
`attention_mask` is ignored. Values greater than or equal to zero are valid
sequence-group IDs. `-1` marks padding. Omit `sequence_id` to use
`attention_mask` for padding.

### Hidden-state sparse autoencoders

ESM++ supports hidden-state SAEs from the official
[Biohub ESMC SAE collection](https://huggingface.co/collections/biohub/esmc-saes-for-hidden-states-all-layers).
This artifact implements the SAE contract, so no Biohub runtime code is needed.
Select an SAE for this ESMC scale, then load only the layers you need:

```python
import torch

model.load_sae_models("{sae_id}", [{sae_layer}])

with torch.inference_mode():
    output = model(**batch, normalize_sae=True)

features = output.sae_outputs["layer{sae_layer}"]
print(features.shape, features.layout)  # (valid_token_count, codebook_dim), sparse COO
```

`load_sae_models` reads the shared `config.json` and one
`layer_{{index}}.safetensors` shard per requested layer, from a Hub repository
or a local directory, and attaches the layers on the model device in the model
dtype. `add_sae_models` still accepts official Biohub `ESMCSAEModel.layers`
entries.

SAEs run after you attach them. Use `compute_sae=False` to skip SAE work.
Outputs are detached sparse tensors with keys such as `layer{{N}}`. They omit
padding. The model uses `sequence_id`, then `attention_mask`, for padding.
`normalize_sae=True` uses Biohub `(features / max) * idf` normalization. SAE
computation requires `input_ids`. It rejects mask tokens because Biohub trained
the SAEs with unmasked sequences. This interface supports hidden-state SAEs
only, not MLP-output SAEs. FastPLMs does not copy SAE weights or add SAE
checkpoints to its model manifest.

{fp8_usage}

{esmc_table}

{ESMC_RELEASE_DOCUMENTATION}

"""
    if family_id == "esm3":
        return """\
## Sequence inference and masked-sequence generation

ESM3 prepares its sequence input. This example uses the sequence track. The
public input contract also supports structure and function tracks through the
multimodal helpers:

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

When `return_dict=False`, ESM3 uses the standard base-model tuple prefix:
`last_hidden_state`, then requested `hidden_states` and `attentions`. Multimodal
logits and extensions follow this prefix. Use named fields for individual tracks.

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

Underscores mark positions to generate. Model outputs are track predictions,
not experimental measurements of structure or function.

"""
    if family_id == "e1":
        return """\
## Tokenizer-free E1 input

E1 has no tokenizer. The model keeps native raw-sequence preparation, boundary
tokens, sequence positions, and retrieval-augmented context behavior. The
ordinary representation path accepts sequences directly:

```python
result = model.embed_dataset(
    ["MSTNPKPQRKTKRNT", "MKTIIALSYIFCLVFA"],
    batch_size=2,
    pooling=("mean",),
)
print(result[0].tensor.shape)
```

Lower-level masked-language-model calls must use the E1 batch preparer, not an
`AutoTokenizer`. E1 launch messages and distributed legal files keep the
attribution required by the upstream agreement.

"""
    if family_id == "dplm":
        license_url = "https://github.com/bytedance/dplm/blob/main/LICENSE"
        readme_url = "https://github.com/bytedance/dplm/blob/main/README.md#overview"
        return f"""\
## Diffusion sequence generation

DPLM gets the requested length from biological positions in a tokenized input.
It masks these positions and retains confident predictions at each iteration:

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

If you omit `max_iter`, DPLM uses the official 500-step schedule. A shorter
schedule changes the sampling process. It is not an equivalent faster mode.

Plain `AutoModel` omits the optional ESM pooler because this diffusion checkpoint
has no trained pooler weights. Pass `add_pooling_layer=True` only when you intend
to initialize and train that head.

DPLM1 and DPLM2 checkpoint weights use Apache-2.0. The ByteDance
[LICENSE]({license_url}) uses Apache-2.0. Its [README]({readme_url}) limits the
repository release to pretrained DPLM1 and DPLM2 weights. FastPLMs artifacts
record `weights_license_status="resolved"` and `redistributable=true`. Complete
publication requires all artifact, legal, parity, and atomic-publication checks.

"""
    if family_id == "dplm2":
        license_url = "https://github.com/bytedance/dplm/blob/main/LICENSE"
        readme_url = "https://github.com/bytedance/dplm/blob/main/README.md#overview"
        return f"""\
## Amino-acid and structure co-generation

DPLM2 uses separate structure and amino-acid tracks. Each track has its own
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

Generic `cls_token`, `eos_token`, `mask_token`, and `unk_token` aliases are not
set. Code that creates multimodal tensors must select the amino-acid or structure
token explicitly. Raw amino-acid sequences remain supported by
`model.embed_dataset(...)`.

Plain `AutoModel` omits the optional ESM pooler because this co-generation
checkpoint has no trained pooler weights. Pass `add_pooling_layer=True` only
when you intend to initialize and train that head.

The checkpoint weights use Apache-2.0. The ByteDance [LICENSE]({license_url})
and [README]({readme_url}) document the license for pretrained DPLM1 and DPLM2
weights. Complete publication requires all artifact, legal, parity, and
atomic-publication checks.

"""
    if family_id == "ankh":
        return f"""\
## Encoder and sequence-to-sequence use

`{spec.fast.repo_id}` contains the complete ANKH encoder-decoder checkpoint.
Use `AutoModel` for encoder embeddings. Use `AutoModelForSeq2SeqLM` for
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

ANKH artifacts retain CC BY-NC-SA 4.0 terms. The notes below distinguish official
heads from FastPLMs extensions. The complete checkpoint is larger than the former
encoder-only mirror and preserves encoder-output parity.

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
    verbose=False,
)
model.save_as_cif(output, "prediction.cif")

print(output.sample_atom_coords.shape)
print(output.plddt, output.ptm, output.iptm)
```

The validation boundary below describes the supported inference subset and its
provisional status. The helper saves and restores Python, NumPy, CPU Torch, and
CUDA RNG state. Parameters and prepared features stay FP32. Supported CUDA
inference runs in BF16 autocast.

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
        verbose=False,
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
has no trained masked-language-model head for this objective. `ttt()` and TTT
folding requests raise.

"""
    if spec.id in {"esmfold2_300", "esmfold2_600"}:
        backbone = (
            "Synthyra/ESMplusplus_small"
            if spec.id == "esmfold2_300"
            else "Synthyra/ESMplusplus_large"
        )
        confidence_note = (
            "The Synthyra-adapted native confidence head returns pLDDT, PAE, pTM, and "
            "iPTM. Confidence calculation is optional."
            if spec.confidence_adaptation is not None
            else "The confidence head is disabled: pLDDT, pTM, iPTM, and PAE are unavailable."
        )
        backbone_layers, backbone_width = (31, 960) if spec.id == "esmfold2_300" else (37, 1152)
        return f"""## Protein folding

This experimental Fast checkpoint has 24 folding blocks and uses the frozen
`{backbone}` backbone. The config-declared step-1500000 backbone and the
pinned ESM++ weights are tensor-exact in BF16 after layout conversion.

```python
import torch

model = model.cuda().eval()
with torch.inference_mode():
    output = model.infer_protein(
        "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
        seed=17,
        num_diffusion_samples=1,
    )
print(output.sample_atom_coords.shape)
```

Folding parameters remain FP32 with CUDA BF16 autocast. The backbone uses
BF16; FP8 requests fail. The 15-step sampler and three folding loops remain
the checkpoint defaults. Protein inputs require `msa=None`.
This checkpoint was trained without MSA conditioning. It rejects
`ProteinInput.msa` and MSA-derived features. Typed multichain and multimolecule
inputs remain supported without MSA conditioning.

{confidence_note}
The 300 and 600 suffixes describe backbone scale, not total model parameters.

{folding_speed}
## Learned representation and ESMC precision

The learned projection maps `H: (b, l, {backbone_layers}, {backbone_width}) -> Z: (b, l, 256)`.
`embed_dataset` returns one `(l, 256)` residue representation per sequence.
The experimental architecture does not expose folding TTT.

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

This is a full 48-block ESMFold2 checkpoint. It supports single-sequence
inference and optional MSA-conditioned inference. Typed multichain and
multimolecule inputs can attach an MSA to each applicable protein chain.

"""
            typed_input_contract = """\
The typed interface also supports RNA, protein MSAs, modifications, and covalent
bonds."""
        else:
            msa_contract = """\
## Alignment-conditioning contract

This 24-block Fast checkpoint is optimized for single-sequence inference. It
was trained without MSA conditioning. It rejects `ProteinInput.msa` and low-level
MSA-derived features. Typed multichain and multimolecule inputs remain supported
when every protein chain uses `msa=None`. Use the full ESMFold2 checkpoint for
MSA-conditioned inference. This follows the official Biohub architecture
description in [Appendix A.2.1](https://biohub.ai/papers/esm_protein.pdf).

"""
            typed_input_contract = """\
The typed interface also supports RNA, modifications, and covalent bonds.
Protein MSA inputs are not supported by this Fast checkpoint; every protein
chain must use `msa=None`."""
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
latency and memory and can worsen a prediction. It does not calibrate confidence
or show biological validity. Folding TTT is result-scoped. Its transient ESMC
adapter modules are excluded from checkpoint state. It is not a generic
`save_pretrained` adapter-persistence path.

"""
        else:
            ttt_note = """\
## Test-time training

This experimental checkpoint does not expose folding TTT. Use the corresponding
standard or Fast checkpoint when you need opt-in ESMC-backbone adaptation.

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

{typed_input_contract} The public schema recognizes `PocketConditioning` and
`DistogramConditioning`, but the pinned official forward consumes neither. Its
feature builder hard-codes a zero pocket feature and constructs distogram tensors
that the released model ignores. FastPLMs therefore rejects non-null pocket and
distogram conditioning instead of silently ignoring scientific inputs. Prepared
`ref_pos` values are component reference geometries created during featurization,
not target coordinates.
Predicted coordinates and confidence scores are outputs and do not establish
biochemical activity.

{folding_speed}
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

## Verified CCD runtime asset

Structure preparation requires `ccd.pkl` from
`biohub/ESMFold2`. The manifest pins its repository, revision, size, content
identity, and MIT terms. This is a trusted-deserialization boundary. FastPLMs
accepts only the pinned snapshot link inside the repository blob directory.
User-supplied asset and `cache_dir` symlinks are rejected. The loader verifies a
private temporary snapshot before deserialization, protecting against
path-replacement and in-place source-write races. Offline execution requires the
exact cached object and never downloads a replacement.

{ttt_note}{binder_note}"""
    if allow_generic:
        return ""
    raise ValueError(f"Unsupported model-card family: {family_id!r}")
