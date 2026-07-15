# Models

The generated [support matrix](generated/support.md) is the current list of
families, checkpoints, AutoClasses, backends, precisions, licenses, and release
tiers. It is produced from `src/fastplms/models.toml`; edit the manifest, not the
table.

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

### ESM++ and ESMC

The ESM++ family provides the ESMC sequence encoders with Hugging Face auto
loading, shared embeddings, and residue-aware pooling. It is also the language
model used by ESMFold2. The model records the resolved attention implementation
and rejects an unavailable requested kernel.

Exact semantic configuration, tokenizer, state, alias, and SDPA contracts are
validated against the pinned Biohub implementation. Backend-specific H100
reproducibility notes are recorded in the ESM++ checkpoint cards.

### ESM3

ESM3 retains sequence, structure, and function tracks and generation helpers
without importing the official checkout at runtime. Feature tests cover encoding,
multimodal inputs, forward outputs, and seeded generation. Dataset embedding
uses the sequence representation and excludes non-protein track tokens.

### DPLM

DPLM generation starts from a tokenized sequence whose biological positions
define the requested length. The sampler replaces those positions with the mask
token, predicts all positions, retains the most confident predictions, and
repeats until no masks remain. Set `partial_masks=True` at positions that must
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
structure = [vocab["<cls_struct>"], *([50] * l), vocab["<eos_struct>"]]
amino_acids = [vocab["<cls_aa>"], *([vocab["A"]] * l), vocab["<eos_aa>"]]
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
with relevant artifacts and containers.

### ANKH

ANKH compatibility uses the official T5 encoder and sequence-to-sequence heads.
`FastAnkhModel` matches the encoder contract and
`FastAnkhForConditionalGeneration` matches the official decoder and LM head.
The encoder is the representative throughput architecture and supports the
manifest-declared eager and SDPA attention implementations. The optional
sequence-to-sequence head is eager-only; exact weights, aliases, seeded
inference, and save/reload are validated in a separate compliance contract.
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

The folding checkpoint remains in FP32 parameter storage. CUDA BF16 inference
enters autocast around the folding operation; loading the checkpoint itself as
static BF16 is not the declared compliance path.

### ESMFold2

Supported variants are restricted to:

- `biohub/ESMFold2`
- `biohub/ESMFold2-Fast`
- `biohub/ESMFold2-Experimental-Cutoff2025`
- `biohub/ESMFold2-Experimental-Fast-Cutoff2025`

All four expose the learned ESMC projection and the `auto`, `bf16`, `fp32`, and
`fp8` ESMC precision policy. See [ESMFold2](esmfold2.md) for the exact embedding,
reload, and folding contracts.

The ESMFold2 folding checkpoint remains FP32 and folding computation uses CUDA
BF16 autocast. Its ESMC backbone is governed independently by the requested
ESMC precision, so selecting BF16 or FP8 ESMC does not change folding-parameter
storage.

### Boltz2

Boltz2 accepts its native prepared structure inputs and preserves trunk,
diffusion, confidence, and export behavior. Its larger scientific dependency
set is isolated in the `structure` extra and structure candidate image. No
chemistry or plotting dependency enters the core package.

Boltz2 retains FP32 parameters and runs supported CUDA BF16 structure inference
inside autocast. Static BF16 parameter loading is not its declared compliance
or artifact-validation path.

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
loading, and all feature tests declared for the family. A backend is supported
only after it meets the common parity thresholds.
