# Models

The generated [support matrix](generated/support.md) is the current list of
families, checkpoints, AutoClasses, backends, precisions, licenses, and release
tiers. It is produced from `src/fastplms/models.toml`; edit the manifest, not the
table.

## Install and platform requirements

FastPLMs 1.0 sequence models require Python 3.11-3.14, PyTorch 2.13, and
Transformers 5.13. Install an immutable runtime revision before loading a Hub
checkpoint:

```bash
python -m pip install \
  "fastplms @ git+https://github.com/Synthyra/FastPLMs.git@<runtime-revision>"
```

Eager and SDPA are portable CPU or CUDA paths. Optimized backends and structure
families have additional dependency, dtype, CUDA, and platform requirements;
check the generated support matrix and the relevant family guide before
selecting one.

## Shared loading contract

Tokenizer-based sequence models load through the normal Transformers auto
classes:

```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained(
    "Synthyra/ESM2-150M",
    trust_remote_code=True,
    attn_implementation="sdpa",
)
```

Each manifest entry declares its valid AutoClasses. Unsupported class or
attention combinations fail explicitly. The default attention implementation
is left unspecified so Transformers can select its standard SDPA path.

## Sequence model families

### ESM2

FastPLMs preserves ESM2 tokenization, encoder outputs, masked-language-model
head, contact head, tied weights, and checkpoint keys. Eager, SDPA, Flex
Attention, and both revision-pinned Hugging Face FlashAttention kernels are
tested against the pinned checkpoint and Transformers references.
Plain `AutoModel` omits the optional ESM pooler because the published
masked-language-model checkpoints contain no trained pooler weights. Pass
`add_pooling_layer=True` only when intentionally initializing and training
that head.

### ESM++ and ESMC

The ESM++ family provides the ESMC sequence encoders with Hugging Face auto
loading, shared embeddings, and residue-aware pooling. It is also the language
model used by ESMFold2. The model records the resolved attention implementation
and rejects an unavailable requested kernel.

ESMC follows the pinned Biohub mask precedence. When `sequence_id` is supplied,
it is authoritative: non-negative values identify chains and `-1` identifies
padding, while `attention_mask` is ignored. Without `sequence_id`,
`attention_mask` is the ordinary padding mask and defaults from the tokenizer
padding ID. Callers that need both chain isolation and padding must encode both
in `sequence_id`; the two public masks are not intersected.

Exact semantic configuration, tokenizer, state, alias, and SDPA contracts are
validated against the pinned Biohub implementation. SDPA is the recommended
highest-fidelity path. Flex Attention and FlashAttention 3 are supported,
non-experimental backends with diagnostic numerical-deviation warnings rather
than strict parity gates. Every checkpoint card exposes the required relative
L2, Q99.9, residue-cosine, pooled-cosine, top-1, and Jensen-Shannon table. Cells
remain explicitly pending until a frozen-head report from the exact
GH200/aarch64 validation target for the backend, dtype, software stack, and
sequence panel is attached. H100 and H200 remain supported Hopper-class
devices, but measurements from them are not interchangeable with or accepted
as the current GH200 release evidence.

### ESM3

ESM3 retains sequence, structure, and function tracks and generation helpers
without importing the official checkout at runtime. Feature tests cover encoding,
multimodal inputs, forward outputs, and seeded generation. Dataset embedding
uses the sequence representation and excludes non-protein track tokens.
With `return_dict=False`, the standard base-model prefix is
`(last_hidden_state, hidden_states, attentions)` with disabled optional fields
omitted; multimodal logits and extensions follow. Named output fields are the
recommended interface for individual tracks.

### DPLM

DPLM generation starts from a tokenized sequence whose biological positions
define the requested length. The sampler replaces those positions with the mask
token, predicts all positions, retains the most confident predictions, and
repeats until no masks remain. The optional `partial_masks` argument is a
boolean tensor with the same shape and device as `input_tokens`; `True` marks
positions that must remain fixed.

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

`sampling_strategy` accepts `gumbel_argmax`, `argmax`, or `vanilla`. The
official 500-step schedule is used when `max_iter` is omitted. A shorter
schedule reduces latency but changes the sampling process.

DPLM advertises eager, SDPA, Flex Attention, and the precompiled Hugging Face
kernels implementation of FlashAttention 3. The official BF16 inference path
loads FP32-resident parameters and enters an explicit CUDA BF16 autocast
context:

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

Static BF16 parameter storage is not an advertised DPLM precision path. The
official autocast contract is exact at every hidden state with SDPA and meets
the fixed engineering target with eager, Flex Attention, and FlashAttention 3.
Released DPLM checkpoints configure attention-probability dropout as zero.
For custom fine-tuning configurations with a nonzero value, FastPLMs applies
that dropout in eager and SDPA training calls. Flex Attention and
FlashAttention 3 fail closed for nonzero training dropout because those paths
do not implement the declared stochastic contract. Evaluation always uses
zero attention dropout. Decoder cross-attention supports eager and SDPA and
fails closed when another backend is requested.
Plain DPLM and DPLM2 `AutoModel` loads likewise omit the optional untrained ESM
pooler by default; it remains available through explicit
`add_pooling_layer=True`.

DPLM1 and DPLM2 checkpoint weights are Apache-2.0. The pinned ByteDance
[LICENSE](https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/LICENSE)
and [README](https://github.com/bytedance/dplm/blob/8a2e15e53416b4536f03f79ad1f6f6a9cbd5e19d/README.md#overview)
provide the immutable license basis; the latter explicitly scopes the official
repository release to the pretrained weights for both model families.

### DPLM2

DPLM2 applies the same confidence-based unmasking separately to amino-acid and
structure tokens. Co-generation input X contains two equal-length tracks,
including their boundary tokens, in the official order: structure first and
amino acids second. The output mapping contains `output_tokens` with the same
shape as X.

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
`unk_token` aliases are intentionally unset, so multimodal tracks must add the
corresponding `<cls_aa>`/`<eos_aa>` or `<cls_struct>`/`<eos_struct>` tokens
explicitly and tokenize them with `add_special_tokens=False`.
`model.embed_dataset(...)` and amino-acid TTT still accept raw sequences: the
model adapter adds `<cls_aa>` and `<eos_aa>`, invokes this exact tokenizer with
`add_special_tokens=False`, and excludes those boundaries from residue outputs.

The default sampler uses `annealing@2.0:0.1` token sampling and
`stochastic1.0` confidence selection. `argmax` with `deterministic` unmasking
provides a deterministic compliance path. The checkpoint sets
`tie_word_embeddings=False`, so input and output embeddings are intentionally
distinct. The trained contact head remains under `esm.contact_head`.

DPLM2 advertises SDPA only. Its BF16 inference contract keeps checkpoint
parameters in FP32 and evaluates the forward pass under CUDA BF16 autocast.
Loading static BF16 parameters for evaluation raises before inference. Eager,
Flex Attention, FlashAttention 2, and FlashAttention 3 requests fail explicitly
because their representative deep hidden-state comparisons miss the release
engineering target.

### E1

E1 has no tokenizer dependency. Its dedicated adapter accepts raw protein
sequences and preserves official boundary-token, context, and retrieval-augmented
generation preparation. Launches display `Profluent-E1` as required by the
upstream agreement. E1 legal files and modified-file notices are distributed
with relevant artifacts and containers. E1 advertises SDPA and Flex Attention;
its eager path is not advertised because it misses the pinned output contract.

MSA-aware embedding returns the same ordered, duplicate-preserving
`EmbeddingResult` and uses the same safetensors or SQLite persistence as
ordinary dataset embedding. Record IDs are zero-based input positions, and
`max_len` is measured in biological residues. `matrix_embed=True` selects full
residue output. Local A3M input is deterministic and offline. Homology search
and Hub MSA download are separate, networked acquisition steps and are never
triggered by an offline embedding call.

Local MMseqs2 search defaults to the official multi-architecture CPU image
`ghcr.io/soedinglab/mmseqs2:18-8cc5c` pinned to manifest digest
`sha256:41b12b0d5f41432fa1b9976123da6e2e06e7fab49a34964f3b54ec038e5845d9`.
It never pulls implicitly. The container runs with `--network none`, each phase
has a bounded timeout, and every local image inspection must match the requested
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

The database and output must resolve beneath the current working directory;
symlink escapes are rejected before Docker runs. Successful searches write
`search-provenance.json` beside the A3M with the image version, manifest digest,
local image ID, platform, database identity, parameters, and sequence hash.
The sidecar also records the A3M size and SHA-256; cached output is reused only
when both provenance and result integrity match. `allow_pull=True` is an explicit
network acquisition opt-in. `allow_network=True` separately permits network
access inside the search container, which a local database search does not
require.

GPU MMseqs2 is not selected automatically. The stable official CUDA image is
AMD64-only, so it is incompatible with GH200/ARM64. `use_gpu=True` requires a
caller-supplied, digest-pinned image that is compatible with the current host;
the CPU default fails closed instead of silently attempting GPU execution.

### ANKH

The ANKH 1.0 migration will replace every existing Synthyra ANKH repository
with full official-compatible T5 state. The currently published immutable
Synthyra revisions are legacy encoder-only checkpoints and are not valid
seq2seq artifacts. Until atomic replacement and validation finish, use a
validated local full artifact for decoder behavior. In the replacement,
`FastAnkhModel` and `AutoModel` load the encoder and shared embeddings cleanly
without decoder allocation, while `FastAnkhForConditionalGeneration` and
`AutoModelForSeq2SeqLM` load the encoder, decoder, cross-attention, and LM head
from the same new immutable commit. The larger full checkpoint changes the
default Hub contents while preserving encoder output parity.

Encoder embeddings are the default and select the final state unless
`hidden_state_index` or `store_all_hidden_states` requests another view.
Decoder embeddings require exactly one explicit aligned `decoder_inputs` list
or `decoder_input_ids` tensor. FastPLMs does not shift the source implicitly,
because official ANKH tasks use prompts, sentinels, or generated tokens. The
decoder biological mask excludes start, EOS, padding, sentinel, and other
special tokens; persistence fingerprints stack, layer, decoder input, mask, and
alignment.

The encoder is the representative throughput architecture and supports the
manifest-declared eager and SDPA attention implementations. Exact encoder and
sequence-to-sequence weights, aliases, seeded inference, and save/reload must be
validated from the same artifact and new Hub revision before that revision is
advertised. Files-only publication is forbidden for the migration.
The previous synthesized masked-language-model head remains available only as
the separately named `FastAnkhForMaskedLMExtension`; it is a FastPLMs extension,
not an official equivalent.

ANKH code and mirrored weights retain CC BY-NC-SA 4.0 terms. FastPLMs displays
those terms but does not enforce a runtime usage policy.

## Structure model families

### ESMFold

ESMFold retains the official ESM2 language-model trunk, folding trunk, output
heads, and structure export. Structure compliance hashes prepared features,
uses seeded stochastic inputs, checks geometry and finite values, and compares
coordinates and confidence outputs with the pinned reference.

The structure-only ESM2 backbone omits the masked-LM and contact-regression
heads because folding consumes hidden states only. Both independently
implemented checkpoint transforms declare and test those omissions. Reported
pLDDT remains on the conventional `(0, 100)` scale; compliance normalizes it to
`(0, 1)` before computing mean absolute error.
For multimer inputs, summary mean pLDDT excludes synthetic linker residues and
includes only biological residues from the requested chains.

The folding checkpoint remains in FP32 parameter storage. CUDA BF16 inference
enters autocast around the folding operation; loading the checkpoint itself as
static BF16 is not the declared compliance path.

### ESMFold2

Supported variants are restricted to:

| Official checkpoint | Folding blocks | MSA conditioning | Intended path |
| --- | ---: | --- | --- |
| `biohub/ESMFold2` | 48 | Optional | Full sequence or complex inference, including MSA-conditioned requests |
| `biohub/ESMFold2-Fast` | 24 | None; MSA-derived inputs are rejected | Inference-optimized single-sequence use |
| `biohub/ESMFold2-Experimental-Cutoff2025` | 48 | Optional | Experimental-cutoff full inference, including MSA-conditioned requests |
| `biohub/ESMFold2-Experimental-Fast-Cutoff2025` | 24 | None; MSA-derived inputs are rejected | Experimental-cutoff, inference-optimized single-sequence use |

The Fast distinction is architectural, not merely a speed label. Biohub's
[Appendix A.2.1](https://biohub.ai/papers/esm_protein.pdf) describes Fast as a
model with 24 folding blocks trained without MSA conditioning for
single-sequence inference, compared with 48 folding blocks in the full model.
The Fast variants are not necessarily single-chain-only: supported multichain
and multimolecule requests remain available, but each protein chain uses
single-sequence mode. Fast variants reject MSA-derived inputs. Use a full variant
whenever optional MSA conditioning is part of the request.

All four expose the learned ESMC projection and the `auto`, `bf16`, `fp32`, and
`fp8` ESMC precision policy. The manifest marks `fp8` as experimental; it is an
explicit inference-only opt-in rather than a release numerical-parity claim.
See [ESMFold2](esmfold2.md) for the exact embedding, reload, and folding contracts.

The ESMFold2 folding checkpoint remains FP32 and folding computation uses CUDA
BF16 autocast. Its ESMC backbone is governed independently by the requested
ESMC precision, so selecting BF16 or FP8 ESMC does not change folding-parameter
storage.

### Boltz2

Boltz2 accepts a raw amino-acid sequence through its protein helper or prepared
model features through its lower-level interface. It preserves trunk,
diffusion, confidence, and export behavior. Its larger scientific dependency
set is isolated in the `structure` extra and structure candidate image. No
chemistry or plotting dependency enters the core package.

Boltz2 retains FP32 parameters and runs supported CUDA BF16 structure inference
inside autocast. Static BF16 parameter loading is not its declared compliance
or artifact-validation path.

`predict_structure(..., seed=...)` owns a scoped Python, NumPy, CPU Torch, and
CUDA RNG context and restores caller state on return. Prepared features and
parameters remain FP32; the supported CUDA compute path enters BF16 autocast
inside that scope. `seed` accepts a Python `int` or `None`; booleans, floats,
strings, NumPy integer scalars, and other coercible values are rejected before
any RNG state is read or changed.

Boltz2 is provisional in FastPLMs 1.0. Exact configuration, the declared
inference-core state, feature preparation, and seeded execution remain covered,
but native-environment BF16 end-to-end inference currently exceeds the fixed
numerical-equivalence limits. FastPLMs does not yet claim official inference
equivalence for Boltz2. Its ongoing structure tests remain available without
blocking the ESM++ and ESMFold2 release gates.

## Test-time training

ProteinTTT-derived adaptation is opt-in and covered as a feature, not a default
loading behavior. It never runs during model construction or ordinary inference.
ESMFold2 reloads ESMC in BF16 before a gradient-enabled path. See
[test-time training](ttt.md).

## Adding a checkpoint

Before code changes, capture the official configuration, tokenizer files and
behavior, state-key and alias schema, outputs, source revision, environment, and
licenses. Add the immutable checkpoint identities and conversion record to the
manifest. Then add official-generated goldens, a live reference case, artifact
loading, and all feature tests declared for the family. A selectable backend
must have an explicit implementation, failure behavior, and documented
numerical boundary. Only backends that meet the applicable release thresholds
may be described as parity paths.
