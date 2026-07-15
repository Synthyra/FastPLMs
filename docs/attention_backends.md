# Attention backends

FastPLMs uses the Transformers attention interface. Callers select a backend at
load time with `attn_implementation` or after loading with
`set_attn_implementation()`.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Synthyra/ESM2-150M",
    trust_remote_code=True,
    attn_implementation="flex_attention",
)
model.set_attn_implementation("sdpa")
```

For prepublication validation, `model_id` is the local manifest-built artifact
under `dist/hub/<model>`. A Hub identifier is used only after that same 1.0
artifact passes the offline artifact tier and is published separately.

If the caller does not choose a backend, FastPLMs leaves the value unspecified
and Transformers normally selects SDPA. FastPLMs does not implement an `auto`
backend and never silently substitutes a different requested kernel.

## Implementations

| Name | Transformation | Mask | Main limitation |
| --- | --- | --- | --- |
| `eager` | Explicit score, softmax, and value products | Additive 4D mask | Materializes attention scores |
| `sdpa` | `scaled_dot_product_attention` | Boolean or additive 4D mask | Kernel dispatch is selected by Torch |
| `flex_attention` | Compiled Flex Attention score function | `BlockMask` | First shape and semantics require compilation |
| `flash_attention_2` | Precompiled `kernels-community/flash-attn2` handler at revision `db6b51744f0c` | Packed 2D mask | ESM2 and ESM++ only |
| `flash_attention_3` | Precompiled `kernels-community/flash-attn3` handler at revision `43f0bd269777` | Packed 2D mask | ESM2, ESM++, and DPLM only |

The manifest lists the subset supported by each family. A requested name that
is not listed for that family raises. A listed optional implementation that
cannot be imported also raises, because dependency absence is a configuration
error rather than evidence that another kernel was tested.

## FlashAttention compatibility policy

The `flash` extra installs Hugging Face `kernels`, not the `flash-attn` Python
package. The adapters follow the
[Transformers kernel-loading contract](https://huggingface.co/docs/transformers/v5.13.0/kernel_doc/loading_kernels)
and resolve only the snapshot-pinned `kernels-community` repositories recorded
in the manifest. The immutable snapshot revisions are
`db6b51744f0cd7061386442c09df890fc6d9f47e` for FlashAttention 2 and
`43f0bd269777115d94ff826e0d113ce9c1c9087b` for FlashAttention 3. The tracked
`kernels.lock` records the exact hash of every published binary variant. The
loader asks `kernels` to download and hash-validate the compatible variant
before importing it. It never falls back to a branch, compiles source, imports
the `flash_attn` package, or substitutes one FlashAttention version for another.

After installing the `flash` extra, `kernels download .` may be used during
image build or cache preparation to fetch both locked binaries. This command
downloads precompiled artifacts only. It is not required when the runtime can
populate its Hugging Face cache on first use.

An explicit kernel-load failure reports the manifest-pinned repository and
revision together with the underlying cause. The exception is not replaced by
a generic dependency error, and no alternate backend is selected.

Both pinned FlashAttention kernels are BF16-only. The Q, K, and V tensors must
share one dtype and one CUDA device. CPU tensors and mixed-device inputs raise
before binary download or import. Direct FP32 and FP16 calls raise before
kernel loading. An
FP32-resident model may use an advertised FlashAttention backend
only inside CUDA BF16 autocast, where the operation resolves to BF16 while the
stored parameters remain FP32. Parity, artifact, embedding, and benchmark
paths derive their backend and dtype combinations from this manifest contract;
they do not probe or fall back to an undeclared precision.

On the locked H100 environment, FlashAttention 2 resolves an exact PyTorch
2.13, CUDA 13, C++11 ABI, x86-64 artifact. FlashAttention 3 resolves a CUDA 13
stable-ABI artifact. Both produce finite dense and mixed-padding outputs and
meet the fixed engineering target for ESM2. For representative ESM++ BF16
inference, FlashAttention 2 has a worst hidden-state relative L2 error of
`0.0103531`, and FlashAttention 3 has an error of `0.0103972`. Both remain below
the `0.03` hard limit but above the required `0.01` engineering target, so the
current ESM++ declaration blocks the 1.0 release. DPLM advertises
FlashAttention 3 only; its FlashAttention 2 result remains outside the
engineering target.

The same ESM++ run measures `0.0101378` for eager attention and `0.0100097` for
Flex Attention. Exact configuration, tokenizer, state, alias, default BF16,
FP32 eager, FP32 and BF16 SDPA, and FP32 Flex contracts pass. The release rule
does not round these BF16 misses down or replace a requested backend with SDPA.

DPLM advertises eager, SDPA, Flex Attention, and FlashAttention 3. Its pinned
official BF16 contract keeps parameter storage in FP32 and uses CUDA BF16
autocast. On the representative H100 case, eager and Flex have worst
hidden-state relative L2 errors of `0.009212` and `0.006768`, respectively.
Static BF16 parameter storage is not the official DPLM precision path and is
not used to justify backend support.

DPLM2 advertises SDPA only. Its pinned BF16 contract also keeps parameters in
FP32 and evaluates them under CUDA BF16 autocast; static BF16 parameter storage
raises before inference. On the representative H100 compliance case, the worst
hidden-state relative L2 errors were `0.011772` for eager, `0.011231` for Flex
Attention, `0.013495` for FlashAttention 2, and `0.012656` for FlashAttention 3.
Each exceeds the fixed `0.01` engineering target. Explicit requests for any of
those backends therefore raise, and their dead kernel paths are not retained in
the DPLM2 implementation.

ANKH advertises eager attention and SDPA only. Its SDPA path forces the math
kernel and temporarily enables reduced-precision reduction so BF16 computation
matches the official encoder, restoring the prior process policy after every
call. Flex Attention is not supported: BF16, FP16, and FP32 probes each exceeded
the fixed relative-L2 engineering target of `0.01` (`0.016673`, `0.015503`, and
`0.016396`, respectively). This family support applies to the optimized ANKH
encoder. The official sequence-to-sequence AutoClass remains eager-only because
its delegated Transformers decoder is outside that encoder implementation.

## Mask semantics

`fastplms.attention` centralizes mask conversion. The same biological validity
mask is normalized into:

- a packed 2D token mask for FlashAttention;
- a 4D mask for eager attention and SDPA;
- a Flex `BlockMask` for padding, causal, block-causal, or declared custom
  semantics.

E1's block-causal pattern is a distinct semantic key. It is never represented
as ordinary padding attention. Mixed-length and skewed-padding parity cases
exercise every required representation.

Flex functions and masks are cached only after explicit execution. The cache key
contains device, dtype, query and key shape, the complete sequence-length tuple,
and mask semantics. This prevents reuse across batches that have the same padded
shape but different valid residues. Importing FastPLMs does not compile Flex or
modify Dynamo or Inductor settings.

## Attention outputs and `parti`

The `parti` embedding pooler constructs an attention graph, so it requires:

```python
attn_implementation="eager"
```

It rejects sequences longer than 2,048 biological residues. Other backends do
not materialize the complete attention graph as a side effect. Models that do
not expose meaningful sequence attention, including ESMFold2, reject `parti`.

## Validation

Backend validation uses the same valid biological positions as official parity.
It measures relative L2 error, relative 99.9th-percentile error, first-percentile
residue cosine, per-sequence pooled cosine, confident-position top-1 agreement,
and Jensen-Shannon divergence for probability tensors. A family-specific relaxed
tolerance is not permitted. A failing advertised backend blocks release until
the implementation is fixed or the backend is removed from the manifest and
documentation.

Performance is measured separately from correctness. See
[benchmarking](benchmarking.md) for compile-time, steady-state, padding, memory,
and regression methodology.
