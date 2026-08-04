# Binder design example

`examples/binder_design_fastplms.py` is a research workflow. It optimizes a
soft binder sequence against ESMFold2 structural objectives and an ESM++
sequence prior. It is a source-level example, not a published model service or
evidence that a designed sequence binds experimentally.

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

The two Cutoff2025 experimental ESMFold2 variants then act as critics.
Candidates are ranked by mean iPTM across those critics. The workflow writes
sequences, loss trajectories, structures, confidence fields, and a selection
table to the requested output directory.

Prepared atom tensors are padded to the largest observed atom table in the
batch, rounded upward for kernel alignment. They are never sized from the first
sequence or rounded downward, so dense binder batches cannot truncate atoms.

![FastPLMs EGFR minibinder design](assets/egfr_fastplms_binder_design.png)

## Run

Run from a source checkout with the `binder` dependency profile. The published
workflow requires Python 3.11-3.14,
PyTorch 2.13, Transformers 5.13, verified ESMFold2 runtime assets, and CUDA. The
current release evidence target is the exact containerized Linux aarch64
environment on the NVIDIA GH200 workstation. CPU-only, x86-64, Windows, macOS,
H100, and H200 binder runs do not substitute for that evidence.

The script intentionally has no standalone PEP 723 dependency block.
`requirements/profiles/binder.in` composes the core, structure, and bounded
binder-design dependencies. Its binder feature pins AbNumber 0.4.4 and ANARCII
2.0.8, plus pandas and PyArrow:

```bash
uv pip install \
  -r requirements/profiles/binder.in \
  -c requirements/constraints/validation.txt
PYTHONPATH=src python examples/binder_design_fastplms.py \
  --target-name pd-l1 \
  --binder-name minibinder \
  --batch-size 4 \
  --steps 150 \
  --output-dir artifacts/binder-design
```

Pass `--target-sequence` instead of `--target-name` for a custom target. Pass
`--binder-sequence` with `#` at mutable positions instead of a named binder
prompt.

The output directory must not already exist, including as an empty directory.
The CLI checks this before loading models, and the design call creates the path
exclusively before optimization. A concurrent, interrupted, or stale run is
therefore rejected instead of having its files mixed with a new campaign.
`run_manifest.json` is written atomically and last; if it is absent, treat the
directory as an incomplete run and preserve or move it for diagnosis before
choosing a new output path.

The default inversion, critic, and ESM++ repositories are loaded at the
immutable FastPLMs commits declared in `src/fastplms/models.toml`; the example
never follows a mutable Hub branch. For a fully cached, network-free run, add
`--local-files-only`. That option passes `local_files_only=True` to every
top-level model load and sets both `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` before loading nested runtime assets. Missing cached
files fail the run instead of downloading them.

Custom repositories require an explicit immutable commit for every model
(replace the example 40-character values below):

```bash
PYTHONPATH=src python examples/binder_design_fastplms.py \
  --inversion-model lab/esmfold2-inversion \
  --critic-model lab/esmfold2-critic \
  --lm-model lab/esmplusplus \
  --model-revision lab/esmfold2-inversion=1111111111111111111111111111111111111111 \
  --model-revision lab/esmfold2-critic=2222222222222222222222222222222222222222 \
  --model-revision lab/esmplusplus=3333333333333333333333333333333333333333 \
  --local-files-only
```

Repeat `--inversion-model`, `--critic-model`, and `--model-revision` when a
campaign uses multiple checkpoints.

The example writes `trajectory.jsonl`, `best_sequences.fasta`,
`results.parquet`, `selection.parquet`, and critic-specific structure and
confidence files plus `run_manifest.json`. The example records the complete
command and normalized configuration; exact optimizer; ESMFold2 critic and
ESM++ weight and runtime revisions; tokenizer identity; backend; parameter and
compute dtype; Torch, Transformers, CUDA runtime, Python, and package
environment; all random seeds; and target, prompt, and input-file hashes. Each
model record separates the requested Hub commit, resolved Hub commit,
`fastplms_weights_revision`, and `fastplms_runtime_revision`. The tokenizer
record carries the ESM++ snapshot and runtime identity alongside its vocabulary
hash.
Antibody CDR positions are obtained through AbNumber's public
`Chain.multiple_domains` API with ANARCII-backed Chothia numbering; the workflow
does not depend on AbNumber's private modules.
Retain the full output directory when comparing campaigns. CUDA driver identity
and ranked-output-table hashes are useful promotion evidence, but the current
example does not emit them and this manifest must not be cited as if it did.

## Validation boundary

Feature tests use short seeded runs to verify prompt construction, mutable masks,
loss finiteness, gradient scope, critic output schema, deterministic ranking,
and structure serialization. They do not validate affinity, specificity,
developability, expression, immunogenicity, toxicity, or therapeutic utility.

Candidates require independent structural review, orthogonal computational
checks, synthesis, and experimental binding and functional validation.
Confidence scores are model outputs, not measurements of biochemical activity.
