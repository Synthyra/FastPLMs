# Test-time training

FastPLMs includes an opt-in ProteinTTT-derived adaptation path for supported
sequence models and the ESMC language-model backbone used by ESMFold2. Ordinary
construction, inference, embedding, folding, and `state_dict()` do not perform
adaptation.

## Mechanism

Given one protein sequence, TTT samples masked views, computes the model's
masked-language-model loss, and updates only injected low-rank adapter
parameters. Base checkpoint parameters remain frozen. The returned metrics
record per-step loss and any explicitly enabled evaluation values.

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

## Main controls

`TTTConfig` controls learning rate, optimization steps, gradient accumulation,
sample batch size, mask ratio, crop size, BERT-style leave and replacement
probabilities, optimizer, seed, low-rank width and scale, target modules, reset,
optional step evaluation, and gradient clipping.

Changing low-rank width, scale, or target modules after adapter initialization
is rejected because it would change the parameter schema.

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

TTT increases latency and GPU memory and can worsen a prediction. It is not a
calibration method and does not establish biological function. Compare the
unadapted output, retain complete seeds and configuration, and validate on an
independent task-specific set before drawing a scientific conclusion.

Boltz2 remains inference-only in FastPLMs. The manifest and feature suite define
the model families that advertise TTT.
