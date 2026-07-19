# Binder design example

`examples/binder_design_fastplms.py` is a research workflow that
optimizes a soft binder sequence against ESMFold2 structural objectives and an
ESM++ sequence prior. It is an example, not a package runtime service and not a
claim that a designed sequence binds experimentally.

## Input, transformation, and output

The input is one target protein chain and either a mutable minibinder prompt or
an antibody framework with mutable CDR positions. A fixed random seed creates
initial sequence logits.

Each optimization step:

1. maps binder logits to residue probabilities;
2. constructs the target and binder folding input;
3. evaluates differentiable intra-chain and inter-chain distogram objectives;
4. adds an ESM++ masked-language-model regularizer;
5. updates only mutable binder logits;
6. retains the lowest-loss discrete candidate.

The two supported Cutoff2025 experimental ESMFold2 variants then act as
critics. Candidates are ranked by mean iPTM across those critics. The workflow
writes sequences, loss trajectories, structures, confidence fields, and a
selection table under the requested output directory.

![FastPLMs EGFR minibinder design](assets/egfr_fastplms_binder_design.png)

## Run

Run inside the project environment with the `structure` extra and the example's
analysis dependencies:

```bash
uv run \
  --extra structure \
  --with abnumber \
  --with pandas \
  --with pyarrow \
  python examples/binder_design_fastplms.py \
  --target-name pd-l1 \
  --binder-name minibinder \
  --batch-size 4 \
  --steps 150 \
  --output-dir artifacts/binder-design
```

Pass `--target-sequence` instead of `--target-name` for a custom target. Pass
`--binder-sequence` with `#` at mutable positions instead of a named binder
prompt.

The example writes `trajectory.jsonl`, `best_sequences.fasta`,
`results.parquet`, `selection.parquet`, and critic-specific structure and
confidence files. Retain the complete command, model revisions, seed, and
output directory when comparing campaigns.

## Validation boundary

Feature tests use short seeded runs to verify prompt construction, mutable masks,
loss finiteness, gradient scope, critic output schema, deterministic ranking,
and structure serialization. They do not validate affinity, specificity,
developability, expression, immunogenicity, toxicity, or therapeutic utility.

Candidates require independent structural review, orthogonal computational
checks, synthesis, and experimental binding and functional validation. Confidence
scores are model outputs, not measurements of biochemical activity.
