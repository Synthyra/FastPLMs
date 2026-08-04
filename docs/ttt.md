# Test-time training

FastPLMs provides an opt-in ProteinTTT adaptation path for supported sequence
models and the ESMC language-model backbone in ESMFold2. Model construction,
inference, embedding, folding, and `state_dict()` do not adapt the model.

## Mechanism

For one protein sequence, TTT samples masked views, computes the
masked-language-model loss, and updates only injected low-rank adapter
parameters. Base checkpoint parameters stay frozen. Returned metrics record
per-step loss and each enabled evaluation value.

Tokenizer-based paths use the tokenizer assets and immutable revision attached
to the loaded checkpoint.

```python
metrics = model.ttt(
    seq="MSTNPKPQRKTKRNT",
    ttt_config={
        "steps": 3,
        "ags": 1,
        "batch_size": 1,
        "seed": 7,
    },
)
model.ttt_reset()
```

`ttt_reset()` restores the initial adapter state. With
`initial_state_reset=True`, each `ttt()` call begins from that initial state.
Seeded tests compare mask sampling, losses, updated parameter scope, reset, and
checkpoint state.

Adapter initialization uses the TTT seed and does not advance the caller Python,
NumPy, CPU Torch, or CUDA RNG streams. Random BERT-style replacements use only
the 20 canonical biological amino acids for the family. They do not use other
vocabulary entries, boundary tokens, or structure modalities. An uneven final
sample batch stays finite and uses its actual valid count.

## Main controls

`TTTConfig` sets the learning rate, optimization steps, gradient accumulation,
sample batch size, mask ratio, crop size, BERT-style leave and replacement
probabilities, optimizer, seed, low-rank width and scale, target modules,
reset, optional step evaluation, and gradient clipping.

`lora_alpha` is a direct multiplier on the low-rank adapter output. It is not
divided by `lora_rank`. This intentionally matches the pinned ProteinTTT call
`inject_trainable_lora(..., scale=lora_alpha)` and differs from the common PEFT
LoRA `alpha / rank` convention. The direct scale is serialized with the TTT
configuration, so changing this interpretation would alter reloaded adapters.

FastPLMs rejects a change to low-rank width, scale, or target modules after
adapter initialization because the change would alter the parameter schema.

## Save and reload

Initialized adapter tensors, their reset baseline, and normalized TTT
configuration are part of `save_pretrained`:

```python
from transformers import AutoModelForMaskedLM

model.ttt(seq="MSTNPKPQRKTKRNT", ttt_config={"steps": 3, "seed": 7})
model.save_pretrained("adapted", safe_serialization=True)
reloaded = AutoModelForMaskedLM.from_pretrained(
    "adapted",
    trust_remote_code=True,
    local_files_only=True,
)
reloaded.ttt_reset()
```

Reload preserves the adapted state and the deterministic reset state. Models
with adapters on transient modules outside checkpoint state fail closed. They
require a model-specific export and do not silently drop adaptation.

## Folding

ESMFold2 exposes a family-specific opt-in folding helper. Adaptation affects
only its language-model backbone. ESMFold2 uses canonical BF16 ESMC weights
before a gradient-enabled path. If serving selected FP8, entering TTT reloads
canonical BF16 weights while preserving the requested serving policy in
configuration and status metadata.

Meta ESMFold does not expose TTT. Its pinned checkpoint contains the folding
language model but no trained masked-language-model head for the ProteinTTT
objective. `ttt()`, `ttt_reset()`, `fold_protein(ttt=True)`, and
`fold_protein_ttt()` therefore raise explicitly. FastPLMs does not construct or
serialize an untrained replacement head.

## Limitations

TTT increases latency and GPU memory. It can worsen a prediction. It is not a
calibration method and does not establish biological function. Compare the
unadapted output, record complete seeds and configuration, and validate on an
independent task-specific set before you make a scientific conclusion.

Boltz2 is inference-only in FastPLMs. The manifest and feature suite define the
model families that advertise TTT.
