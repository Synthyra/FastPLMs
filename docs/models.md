# Models

The generated [support matrix](generated/support.md) lists the current model
families, checkpoints, AutoClasses, backends, precisions, licenses, and release
tiers. `src/fastplms/models.toml` generates this table. Edit the manifest, not
the table.

## Dependencies and platform requirements

FastPLMs 1.0 sequence models require Python 3.11-3.14, PyTorch 2.13, and
Transformers 5.13. Install these dependencies directly. The pinned Hugging Face
model repository provides the runtime source:

```bash
python -m pip install \
  "torch>=2.13,<2.14" \
  "transformers>=5.13,<5.14"
```

Eager and SDPA run on CPU or CUDA. Optimized backends and structure families
have additional requirements for dependencies, dtype, CUDA, and platform. Check
the generated support matrix and the applicable family guide before you select
one.

## Shared loading contract

Tokenizer-based sequence models load with the standard Transformers auto
classes:

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained(
    "Synthyra/ESM2-150M",
    trust_remote_code=True,
    attn_implementation="sdpa",
)
```

Each manifest entry declares its supported AutoClasses. Unsupported class and
attention combinations fail explicitly. The default attention implementation is
not specified. Transformers then selects its standard SDPA path.

## Sequence model families

### ESM2

FastPLMs preserves ESM2 tokenization, encoder outputs, masked-language-model
head, contact head, tied weights, and checkpoint keys. Eager, SDPA, Flex
Attention, and both revision-pinned Hugging Face FlashAttention kernels are
tested against the pinned checkpoint and Transformers references. Plain
`AutoModel` omits the optional ESM pooler. The published masked-language-model
checkpoints have no trained pooler weights. Set `add_pooling_layer=True` only
when you intend to initialize and train that head.

### ESM++ and ESMC

The ESM++ family provides ESMC sequence encoders with Hugging Face auto
loading, shared embeddings, and residue-aware pooling. ESMFold2 also uses this
language model. The model records the selected attention implementation and
rejects an unavailable requested kernel.

ESMC follows the pinned Biohub mask precedence. If you provide `sequence_id`,
it is authoritative. Non-negative values identify chains, and `-1` identifies
padding. In this case, ESMC ignores `attention_mask`. Without `sequence_id`,
`attention_mask` is the padding mask. Its default uses the tokenizer padding ID.
To specify chain isolation and padding, encode both in `sequence_id`. ESMC does
not intersect the two public masks.

ESM++ supports hidden-state sparse autoencoders (SAEs) from the official
[Biohub ESMC SAE collection](https://huggingface.co/collections/biohub/esmc-saes-for-hidden-states-all-layers).
This includes the five Biohub Platform checkpoints for ESMC-300M layer 23,
ESMC-600M layer 27, and ESMC-6B layer 60. Compatible hidden-state Hub variants
are also supported.

Biohub owns the SAE weights. FastPLMs does not copy SAE weights or add SAE
checkpoints to its manifest. Load the SAE container with `AutoModel`. Load only
the required layers. Then attach the layers to an ESM++ model of the same scale:

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "Synthyra/ESMplusplus_6B"
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
sae = AutoModel.from_pretrained(
    "biohub/ESMC-6B-sae-layer60-k64-codebook16384",
    allow_patterns=["config.json", "layer_60.safetensors"],
    device=model.device,
)
sae.initialize_layers([60])
model.add_sae_models([sae.layers["60"]])

inputs = tokenizer("MSTNPKPQRKTKRNT", return_tensors="pt")
inputs = {name: value.to(model.device) for name, value in inputs.items()}
with torch.inference_mode():
    output = model(**inputs, normalize_sae=True)

features = output.sae_outputs["layer60"]
print(features.shape, features.layout)  # (valid_token_count, codebook_dim), sparse COO
```

SAEs run after you attach them. Set `compute_sae=False` to skip SAE computation.
`sae_outputs` uses keys such as `layer60`. Each detached sparse tensor contains
only tokens that are valid under the `sequence_id`, then `attention_mask`, rule.
`normalize_sae=True` applies Biohub `(features / max) * idf` normalization. SAE
computation requires `input_ids`. It rejects mask tokens because Biohub trained
these SAEs with unmasked sequences. This interface supports hidden-state SAEs.
It does not support MLP-output SAEs.

The default ESM++ path uses the checkpoint BF16 behavior. FP8 is an explicit,
experimental inference option for all ESM++ scales. Load BF16 weights on CUDA.
Call `model.enable_fp8()` on an evaluation model. Check
`model.esmc_precision_status`. FP8 forward calls require
`torch.inference_mode()`. The model pads the sequence dimension to a multiple
of 16. Transformer Engine converts the supported linear layers. The call fails
when Transformer Engine, compatible CUDA hardware, or the complete conversion
set is unavailable. It does not silently use BF16. FP8 does not claim numerical
parity.

Exact semantic configuration, tokenizer, state, alias, and SDPA contracts are
validated against the pinned Biohub implementation. SDPA is the recommended
high-fidelity path. Flex Attention and FlashAttention 3 are supported,
non-experimental backends. They report diagnostic numerical-deviation warnings
instead of strict parity gates. Every checkpoint card gives the required
relative L2, Q99.9, residue-cosine, pooled-cosine, top-1, and Jensen-Shannon
table. A cell remains pending until a frozen-head report is available from the
exact GH200/aarch64 validation target for the backend, dtype, software stack,
and sequence panel. H100 and H200 are supported Hopper-class devices. Their
measurements are not interchangeable with current GH200 release evidence.

### ESM3

ESM3 retains sequence, structure, and function tracks and generation helpers.
It does not import the official checkout at runtime. Feature tests cover
encoding, multimodal inputs, forward outputs, and seeded generation. Dataset
embedding uses the sequence representation and excludes non-protein track
tokens. With `return_dict=False`, the standard base-model prefix is
`(last_hidden_state, hidden_states, attentions)` with disabled optional fields
omitted; multimodal logits and extensions follow. Named output fields are the
recommended interface for individual tracks.

### DPLM

DPLM generation starts with a tokenized sequence. Its biological positions set
the requested length. The sampler replaces these positions with the mask token,
predicts all positions, keeps the most confident predictions, and repeats until
no masks remain. The optional `partial_masks` argument is a Boolean tensor with
the same shape and device as `input_tokens`. `True` marks a position that must
remain fixed.

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "Synthyra/DPLM-150M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForMaskedLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
).cuda().eval()

X = tokenizer("A" * 64, return_tensors="pt")["input_ids"].cuda()
with torch.inference_mode():
    generated_tokens = model.generate(X, max_iter=100)
sequence = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).replace(" ", "")
```

`sampling_strategy` accepts `gumbel_argmax`, `argmax`, or `vanilla`. If you omit
`max_iter`, DPLM uses the official 500-step schedule. A shorter schedule reduces
latency but changes the sampling process.

DPLM supports eager, SDPA, Flex Attention, and the precompiled Hugging Face
kernels implementation of FlashAttention 3. The official BF16 inference path
loads FP32-resident parameters and enters an explicit CUDA BF16 autocast context:

```python
model = AutoModelForMaskedLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    attn_implementation="sdpa",
    dtype=torch.float32,
).cuda().eval()

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(X)
```

Static BF16 parameter storage is not a supported DPLM precision path. The
official autocast contract is exact at every hidden state with SDPA. It meets
the fixed engineering target with eager, Flex Attention, and FlashAttention 3.
Released DPLM checkpoints set attention-probability dropout to zero. For custom
fine-tuning configurations with a nonzero value, FastPLMs applies this dropout
in eager and SDPA training calls. Flex Attention and FlashAttention 3 fail
closed for nonzero training dropout because these paths do not implement the
declared stochastic contract. Evaluation always uses zero attention dropout.
Decoder cross-attention supports eager and SDPA. It fails closed when another
backend is requested. Plain DPLM and DPLM2 `AutoModel` loads also omit the
optional untrained ESM pooler by default. It remains available through explicit
`add_pooling_layer=True`.

DPLM1 and DPLM2 checkpoint weights use Apache-2.0. The pinned ByteDance
[LICENSE](https://github.com/bytedance/dplm/blob/main/LICENSE)
and [README](https://github.com/bytedance/dplm/blob/main/README.md#overview)
provide the immutable license basis. The README explicitly applies the official
repository release to pretrained weights for both model families.

### DPLM2

DPLM2 applies the same confidence-based unmasking separately to amino-acid and
structure tokens. Co-generation input X contains two equal-length tracks,
including boundary tokens. The official order is structure first and amino acids
second. The output mapping contains `output_tokens` with the same shape as X.

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "Synthyra/DPLM2-150M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(
    checkpoint,
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
X = torch.tensor([structure + amino_acids], device="cuda")

with torch.inference_mode():
    generated_tokens = model.generate(X, max_iter=100)["output_tokens"]
```

The DPLM2 tokenizer preserves separate amino-acid and structure boundary,
unknown, and mask tokens. Generic `cls_token`, `eos_token`, `mask_token`, and
`unk_token` aliases are intentionally unset. Therefore, multimodal tracks must
add the matching `<cls_aa>`/`<eos_aa>` or `<cls_struct>`/`<eos_struct>` tokens
and tokenize them with `add_special_tokens=False`. `model.embed_dataset(...)`
and amino-acid TTT still accept raw sequences. The model adapter adds
`<cls_aa>` and `<eos_aa>`, uses this tokenizer with `add_special_tokens=False`,
and excludes these boundaries from residue outputs.

The default sampler uses `annealing@2.0:0.1` token sampling and
`stochastic1.0` confidence selection. `argmax` with `deterministic` unmasking
is a deterministic compliance path. The checkpoint sets
`tie_word_embeddings=False`. Input and output embeddings are intentionally
distinct. The trained contact head is `esm.contact_head`.

DPLM2 supports SDPA only. Its BF16 inference contract keeps checkpoint
parameters in FP32 and evaluates the forward pass with CUDA BF16 autocast.
Loading static BF16 parameters for evaluation raises before inference. Eager,
Flex Attention, FlashAttention 2, and FlashAttention 3 requests fail explicitly
because representative deep hidden-state comparisons do not meet the release
engineering target.

### E1

E1 has no tokenizer dependency. Its adapter accepts raw protein sequences and
preserves official boundary-token, context, and retrieval-augmented generation
preparation. Launches display `Profluent-E1`, as required by the upstream
agreement. E1 legal files and modified-file notices are in the applicable
artifacts and containers. E1 supports SDPA and Flex Attention. It does not
support eager because that path does not meet the pinned output contract.

MSA-aware embedding returns the same ordered, duplicate-preserving
`EmbeddingResult`. It uses the same safetensors or SQLite persistence as
ordinary dataset embedding. Record IDs are zero-based input positions. `max_len`
is measured in biological residues. `matrix_embed=True` selects full residue
output. Local A3M input is deterministic and offline. Homology search and Hub
MSA download are separate network acquisition steps. An offline embedding call
never starts them.

Local MMseqs2 search defaults to the official multi-architecture CPU image
`ghcr.io/soedinglab/mmseqs2:18-8cc5c` pinned to manifest digest
`sha256:41b12b0d5f41432fa1b9976123da6e2e06e7fab49a34964f3b54ec038e5845d9`.
It never pulls implicitly. The container runs with `--network none`. Each phase
has a bounded timeout. Each local image inspection must match the requested
repository digest, Linux platform, host architecture, and a valid image ID.

```python
from fastplms.models.e1.retrieval import HomologueSearcher

searcher = HomologueSearcher(
    target_db="databases/uniref30",
    use_gpu=False,
    allow_pull=False,
    allow_network=False,
    phase_timeout=1800,
    target_db_identity="uniref30-release-2025-02",
)
a3m_path = searcher.search("MSTNPKPQRKTKRNT", "msa-results", seq_id="query-1")
```

The database and output must resolve below the current working directory.
Symlink escapes are rejected before Docker runs. A successful search writes
`search-record.json` beside the A3M. This source record contains the image version, manifest
digest, local image ID, platform, database identity, parameters, and sequence
hash. The sidecar also records the A3M size and SHA-256. Cached output is reused
only when source-record and result-integrity checks match. `allow_pull=True` is
an explicit network-acquisition opt-in. `allow_network=True` separately permits
network access in the search container. A local database search does not need it.

GPU MMseqs2 is not selected automatically. The stable official CUDA image is
AMD64-only, so it is incompatible with GH200/ARM64. `use_gpu=True` requires a
caller-supplied, digest-pinned image that is compatible with the current host.
The CPU default fails closed instead of silently trying GPU execution.

### ANKH

The ANKH 1.0 migration replaced all Synthyra ANKH repositories with complete
official-compatible T5 state. `FastAnkhModel` and `AutoModel` load the encoder
and shared embeddings without decoder allocation. `FastAnkhForConditionalGeneration`
and `AutoModelForSeq2SeqLM` load the encoder, decoder, cross-attention, and LM
head from the same repository. The larger full checkpoint changes the default
Hub contents and preserves encoder output parity.

Encoder embeddings are the default. They select the final state unless
`hidden_state_index` or `store_all_hidden_states` requests another view.
Decoder embeddings require exactly one aligned `decoder_inputs` list or
`decoder_input_ids` tensor. FastPLMs does not shift the source implicitly.
Official ANKH tasks use prompts, sentinels, or generated tokens. The decoder
biological mask excludes start, EOS, padding, sentinel, and other special
tokens. Persistence fingerprints stack, layer, decoder input, mask, and
alignment.

The encoder is the representative throughput architecture. It supports the
manifest-declared eager and SDPA attention implementations. Validate exact
encoder and sequence-to-sequence weights, aliases, seeded inference, and
save/reload from the same artifact and new Hub revision before you advertise the
revision. The ANKH 1.0 migration itself used a complete weights-plus-runtime
update; subsequent files-only updates compile and publish runtime files without
re-uploading those weights. The previous
synthesized masked-language-model head remains available only as
`FastAnkhForMaskedLMExtension`. It is a FastPLMs extension, not an official
equivalent.

ANKH code and mirrored weights retain CC BY-NC-SA 4.0 terms. FastPLMs displays
those terms but does not enforce a runtime usage policy.

## Structure model families

### ESMFold

ESMFold retains the official ESM2 language-model trunk, folding trunk, output
heads, and structure export. Structure compliance hashes prepared features, uses
seeded stochastic inputs, checks geometry and finite values, and compares
coordinates and confidence outputs with the pinned reference.

The structure-only ESM2 backbone omits masked-LM and contact-regression heads
because folding consumes hidden states only. Both independently implemented
checkpoint transforms declare and test these omissions. Reported pLDDT remains
on the conventional `(0, 100)` scale. Compliance normalizes it to `(0, 1)`
before it calculates mean absolute error. For multimer inputs, summary mean
pLDDT excludes synthetic linker residues and includes only biological residues
from the requested chains.

The folding checkpoint keeps FP32 parameter storage. CUDA BF16 inference uses
autocast for the folding operation. Loading the checkpoint itself as static BF16
is not the declared compliance path.

### ESMFold2

Supported variants are restricted to:

| Official checkpoint | Folding blocks | MSA conditioning | Intended path |
| --- | ---: | --- | --- |
| `biohub/ESMFold2` | 48 | Optional | Full sequence or complex inference, including MSA-conditioned requests |
| `biohub/ESMFold2-Fast` | 24 | None; MSA-derived inputs are rejected | Inference-optimized single-sequence use |
| `biohub/ESMFold2-Experimental-Cutoff2025` | 48 | Optional | Experimental-cutoff full inference, including MSA-conditioned requests |
| `biohub/ESMFold2-Experimental-Fast-Cutoff2025` | 24 | None; MSA-derived inputs are rejected | Experimental-cutoff, inference-optimized single-sequence use |

Fast is an architectural distinction, not only a speed label. Biohub
[Appendix A.2.1](https://biohub.ai/papers/esm_protein.pdf) describes Fast as a
model with 24 folding blocks. It is trained without MSA conditioning for
single-sequence inference. The full model has 48 folding blocks. Fast variants
are not necessarily limited to a single chain. Supported multichain and
multimolecule requests remain available, but each protein chain uses
single-sequence mode. Fast variants reject MSA-derived inputs. Use a full
variant when the request includes optional MSA conditioning.

All four expose the learned ESMC projection and the `auto`, `bf16`, `fp32`, and
`fp8` ESMC precision policy. The manifest marks `fp8` as experimental. It is an
explicit inference-only opt-in. It does not claim release numerical parity. See
[ESMFold2](esmfold2.md) for the exact embedding, reload, and folding contracts.

The ESMFold2 folding checkpoint remains FP32. Folding computation uses CUDA
BF16 autocast. Requested ESMC precision controls the ESMC backbone separately.
Therefore, selecting BF16 or FP8 ESMC does not change folding-parameter storage.

### Boltz2

Boltz2 accepts a raw amino-acid sequence through its protein helper or prepared
model features through its lower-level interface. It preserves trunk, diffusion,
confidence, and export behavior. Its scientific dependencies are in
`requirements/features/structure.in` and the structure candidate image.
Chemistry and plotting dependencies are not core runtime requirements.

Boltz2 retains FP32 parameters and runs supported CUDA BF16 structure inference
in autocast. Static BF16 parameter loading is not its declared compliance or
artifact-validation path.

`predict_structure(..., seed=...)` owns a scoped Python, NumPy, CPU Torch, and
CUDA RNG context. It restores caller state on return. Prepared features and
parameters remain FP32. The supported CUDA compute path uses BF16 autocast in
this scope. `seed` accepts a Python `int` or `None`. It rejects Booleans,
floats, strings, NumPy integer scalars, and other coercible values before it
reads or changes RNG state.

Boltz2 is provisional in FastPLMs 1.0. Tests cover exact configuration, the
declared inference-core state, feature preparation, and seeded execution.
However, native-environment BF16 end-to-end inference currently exceeds the
fixed numerical-equivalence limits. FastPLMs does not yet claim official
inference equivalence for Boltz2. Its structure tests remain available and do
not block the ESM++ and ESMFold2 release gates.

## Test-time training

ProteinTTT-derived adaptation is opt-in. It is a feature, not default loading
behavior. It never runs during model construction or ordinary inference.
ESMFold2 reloads ESMC in BF16 before a gradient-enabled path. See
[test-time training](ttt.md).

## Adding a checkpoint

Before you change code, capture the official configuration, tokenizer files and
behavior, state-key and alias schema, outputs, source revision, environment, and
licenses. Add immutable checkpoint identities and the conversion record to the
manifest. Then add official-generated goldens, a live reference case, artifact
loading, and all feature tests declared for the family. A selectable backend
must have an explicit implementation, failure behavior, and documented
numerical boundary. Describe only backends that meet the applicable release
thresholds as parity paths.
