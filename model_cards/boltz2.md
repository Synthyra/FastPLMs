---
library_name: transformers
license: "mit"
tags:
  - protein-language-model
  - fastplms
---

<!-- Generated from src/fastplms/models.toml. Do not edit. -->

# Synthyra/Boltz2

This checkpoint uses the FastPLMs `Boltz2` implementation.
Its input mode is `structure` and its advertised AutoClasses
are `AutoConfig`, `AutoModel`.

## Quick start

```python
from transformers import AutoModel

model_id = "Synthyra/Boltz2"
model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
).eval()
```

This example uses the published Hub repository. For offline validation, build
the manifest-pinned artifact and replace `model_id` with its local
`dist/hub/<model>` path, then pass `local_files_only=True`.

Leave attention unspecified for the Transformers default or request one of
`eager` with `attn_implementation`.
The BF16 execution policy is `fp32_parameters_autocast`:
FP32 parameters with CUDA BF16 autocast.

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

## Notes and limitations

Boltz2 is provisional in FastPLMs 1.0. Exact configuration, the declared inference-core state, feature preparation, and seeded execution remain tested, but native-environment BF16 end-to-end inference currently exceeds the fixed numerical-equivalence limits. FastPLMs therefore does not claim official inference equivalence for this checkpoint yet. Work on that numerical gap continues independently of the ESM++ and ESMFold2 release gates.

## Runtime contract

- Input mode: `structure`
- Advertised AutoClasses: `AutoConfig`, `AutoModel`
- Attention implementations: `eager`
- Precision policies: `default`
- BF16 execution: `fp32_parameters_autocast`
- Generation contract: `not_applicable`
- Optional dependency group: `structure`

## Provenance

- FastPLMs checkpoint: `Synthyra/Boltz2@3b148fc5efea109c065ec82ba8683d024de7134e`
- Official checkpoint: `boltz-community/boltz-2@6fdef46d763fee7fbb83ca5501ccceff43b85607`
- Artifact source: `fast`
- State transform: `boltz2_inference_core_v1`
- BF16 execution: `fp32_parameters_autocast`
- Pinned upstreams: `boltz`
- Reference container: `reference-boltz2`
- Release tiers: `structure`, `artifact`, `benchmark`
- Unresolved required file identities: `0`

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

Checkpoint terms: MIT. The Hub model-card identifier is
`mit`. Applicable source licenses, notices, attribution,
and conversion records are distributed with the local artifact. Review them
before use.
