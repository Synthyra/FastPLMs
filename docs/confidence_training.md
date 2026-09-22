# ESMFold2 confidence-head research

Status: both training stages completed through validation-based early stopping.
Both heads passed their held-out quality gates.
Artifact release checks and publication remain pending. The published
Fast 300M and 600M mirrors continue to have disabled confidence outputs.

Both prepared artifacts passed their focused reload, coordinate, and CIF checks.
Final source records and generated cards are prepared, and both refreshed
artifacts passed isolated inference checks. Each upload inventory contains 75
files, including the checkpoint weights. Publication and published reload remain
pending. Reports are `artifacts/confidence/release-300.json` and
`artifacts/confidence/release-600.json`. The monitoring heartbeat is paused
because its two calls completed. Archived pilot evaluation records are
[300M](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/resolve/07cd9e4fee7aeb18ff9d2ce2078f9092fa8ed3f3/docs/evidence/confidence/esmfold2_300.json) and
[600M](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/resolve/07cd9e4fee7aeb18ff9d2ce2078f9092fa8ed3f3/docs/evidence/confidence/esmfold2_600.json); artifact checks are
[300M](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/resolve/07cd9e4fee7aeb18ff9d2ce2078f9092fa8ed3f3/docs/evidence/confidence/esmfold2_300-package.json) and
[600M](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/resolve/07cd9e4fee7aeb18ff9d2ce2078f9092fa8ed3f3/docs/evidence/confidence/esmfold2_600-package.json).

The approved run permits one H100 worker per model for about ten hours in
parallel. Duration is bounded, cost is recorded, and no more than two GPU
workers may run at once. The supplied multimer archive has downloaded and
extracted successfully. Split construction passed, and the two campaigns were
restarted on [Modal](https://modal.com/apps/synthyra/main/ap-EaDYxs3xoQJX4LUulUd5PM).
Both completed all 1,408 model-specific caches and passed the balanced
eight-target overfit check before starting their main training stages.
The initial workers stopped during cache generation with a CPU/GPU tensor
mismatch in equivalent-chain matching, before any optimizer update. The local
dispatch receipt is `artifacts/confidence/result-286b39b5dd5d423b967fb857186c77a7.json`.
Atom matching now receives CPU features alongside CPU coordinates. Valid
completed caches are retained for the next invocation.
The corrected campaign receipt is
`artifacts/confidence/result-2c6cd577525443f78ab72b506e034093.json`.
Both online W&B cache runs were verified: [300 cache](https://wandb.ai/lhallee/fastplms-confidence/runs/f167111492a8)
and [600 cache](https://wandb.ai/lhallee/fastplms-confidence/runs/c0ce0f0e3597).
The first live metric check confirmed 104 of 1,408 caches for 300M and 89 of
1,408 for 600M, with elapsed time, throughput, and peak memory reported by
both runs. These are progress observations, not completed-cache counts.
The task heartbeat follows this replacement campaign every 30 minutes.
The earlier review pause has ended.

At the September 16, 2026, 17:18 UTC check, the main runs had reached 399
updates for 300M and 321 for 600M. These are interim observations; the latest
validation metrics below do not constitute final-test or release results.

| Model | Training run | Validation atom pLDDT MAE | Validation pLDDT/lDDT Spearman | Validation iPTM/DockQ Spearman |
| --- | --- | ---: | ---: | ---: |
| 300M | [W&B training](https://wandb.ai/lhallee/fastplms-confidence/runs/5b1bc71e8778) | 0.0774 | 0.8493 | 0.7750 |
| 600M | [W&B training](https://wandb.ai/lhallee/fastplms-confidence/runs/d86012361165) | 0.0748 | 0.8542 | 0.7641 |

Each validation panel contains 128 targets. The overfit checks completed 100
updates each: [300M](https://wandb.ai/lhallee/fastplms-confidence/runs/029595b2ab3e)
and [600M](https://wandb.ai/lhallee/fastplms-confidence/runs/66d68b5caee4).

At the 18:20 UTC check, 300M had stopped at 1,600 updates after 4,836 seconds
(80.6 minutes), when 12 consecutive validation checks did not satisfy the
checkpoint-selection rule. It did not run for the full ten-hour limit. Its
selected checkpoint has validation atom pLDDT MAE 0.0773, calibration error
0.0420, pLDDT/lDDT Spearman 0.8589, and iPTM/DockQ Spearman 0.7984.
These are validation results. The separate final-test evaluation completed
in [Modal](https://modal.com/apps/synthyra/main/ap-IBxoNRDp2HR9PxgauONgYF).

The 300M final evaluation completed and passed all predeclared quality gates
on 64 held-out monomers and 64 dimers, with two seeds per target.
[W&B evaluation](https://wandb.ai/lhallee/fastplms-confidence/runs/fcbac1a4d26f).

| Metric | 300M final result |
| --- | ---: |
| Atom pLDDT MAE | 0.0682 |
| Cα pLDDT MAE | 0.0682 |
| Calibration error | 0.0334 |
| Target pLDDT/lDDT Spearman | 0.9092 |
| iPTM/DockQ Spearman | 0.8345 |
| pLDDT cross-entropy | 2.4338 |
| PAE cross-entropy | 2.5942 |

Both categorical losses beat the unchanged donor and training-frequency
baseline. Target-bootstrap intervals were 0.0629–0.0745 for atom MAE,
0.8509–0.9447 for pLDDT rank correlation, and 0.7460–0.8904 for interface
rank correlation. These results concern short monomers and dimers only.

Within-target sample ranking remains weak: the head selected the higher-lDDT
sample in 47.6% of comparable pairs and the higher-DockQ sample in 46.0%.
These two-seed results do not support claims of improved sample selection.
The final report is saved locally as `artifacts/confidence/evaluation-300.json`.

The 600M run stopped at 1,900 updates after 6,189 seconds (103.2 minutes).
Its held-out evaluation completed in
[Modal](https://modal.com/apps/synthyra/main/ap-lEsPyOKZ8I0yCycPonQMog),
call `fc-01M2NS2S4FS0K6D1E3ANN09K21`, and passed every quality gate.
[W&B evaluation](https://wandb.ai/lhallee/fastplms-confidence/runs/8a9d898f1696).

| Metric | 600M final result |
| --- | ---: |
| Atom pLDDT MAE | 0.0718 |
| Cα pLDDT MAE | 0.0728 |
| Calibration error | 0.0378 |
| Target pLDDT/lDDT Spearman | 0.8631 |
| iPTM/DockQ Spearman | 0.8782 |
| pLDDT cross-entropy | 2.4533 |
| PAE cross-entropy | 2.5118 |

Both categorical losses beat the donor and frequency baseline on 64 monomers
and 64 dimers. Target-bootstrap intervals were 0.0662–0.0781 for atom MAE,
0.7833–0.9194 for pLDDT correlation, and 0.8009–0.9163 for interface correlation.
Two-seed selection accuracy was 48.8% for lDDT and 54.0% for DockQ, which does
not establish improved sample selection. The report is
`artifacts/confidence/evaluation-600.json`.

The 600M artifact checks passed in
[Modal](https://modal.com/apps/synthyra/main/ap-wMHEklkyH3UxvpOBJqZCl1),
call `fc-01M2NWGRCB6F763KGKJY8EHZFD`.
[W&B artifact check](https://wandb.ai/lhallee/fastplms-confidence/runs/683d881405c4).
All 820 folding tensors remained byte-identical. Transformers reload,
confidence ranges, one- and two-sample coordinate equality, and two-chain CIF
confidence export passed under the same settings as the 300M check below.
The head SHA256 is `413db7dec0730c4928cf351fd77337d70c7943fa4582758b42f2e9055699da4d`;
the combined weights SHA256 is `fd6fd8c75f702a3bfd2ace37a352b8b8a25e6ce09db1ea6ba74975ae1efaf527`.
The report is `artifacts/confidence/package-600.json`.

The 300M artifact passed its isolated Transformers reload and inference checks
on Modal with an H100, PyTorch 2.13, BF16 autocast, FP32 folding parameters,
and SDPA. All 820 original folding tensors remained byte-identical. Seed 17
produced identical coordinates with confidence on and off for one and two
samples. The 112-residue two-chain test exported 870 atoms with known CIF
confidence values. This is a focused integration check, not full benchmark
compliance. [W&B artifact check](https://wandb.ai/lhallee/fastplms-confidence/runs/11ac7998a25c).

The head SHA256 is `4dfc691e529617fc7cc0cd1ee60799ce1b496fdd0d17f0b4bb78b33a96c77733`;
the combined weights SHA256 is `81028c41ba6df9f1f75c1a372b2455e9152b5aa0b68309ebd5faa82e986269c7`.
The report is `artifacts/confidence/package-300.json`. Generated cards and
source records still need the adaptation details before publication.

This document records the planned Modal pilot and its acceptance rules. The
launcher interface is still evolving. Treat the code in `tools/confidence/`
and the reports written under `artifacts/confidence/` as the operational
record when they differ from this document.

## Current audit checks

The latest workflow passed 119 focused tests in 12.48 seconds on Modal,
including the remote stage ordering, time limits, and AtlasFold's one-element
sequence fields, CPU atom matching, W&B cache-run lifecycle, and byte-exact
preservation of folding tensors during head packaging.
[Test run](https://modal.com/apps/synthyra/main/ap-8xH9PQ8HQP5Ty691bc7qLx).
The final source linter passed before this run.

The initial candidate pool left 486 of the required 512 training dimers after
sequence exclusions. Expanding the candidate multiplier from four to eight
produced 16,085 training candidates and 318 official validation candidates.
Split sizes and sequence-exclusion thresholds remain unchanged.
[Preparation run](https://modal.com/apps/synthyra/main/ap-BP6N7FRYFyEH3jCpuBifJ6).

The final [split check](https://modal.com/apps/synthyra/main/ap-JJkjs62PRssVQM0FslyfSN)
passed with 512 monomers and 512 dimers for training, 64 of each for validation,
and 64 of each for the held-out final test. The selected-record SHA256 is
`7cc79fb577fa7121e9f1c1d2f3c73f55a332cbdb1772802a18acf535de63be99`.

The archive-copy update passed 90 focused tests in 20.61 seconds on Modal,
including separate partial files for different Drive objects and rejection of
unapproved archive IDs. [Test run](https://modal.com/apps/synthyra/main/ap-NJZVxAnFUcCc3PeFCloOrp).

The revised workflow passed all 88 focused tests in 17.90 seconds on Modal.
The suite includes native head gradients, label geometry, padded atoms,
data mapping, split exclusions, checkpoint persistence, and distinct-model
dispatch bookkeeping. The repository linter also passed on Modal. These
checks establish implementation behavior; they do not measure trained-head
quality. [Test run](https://modal.com/apps/synthyra/main/ap-i2EarU0uSTgDEwtuwrbhYd),
[lint run](https://modal.com/apps/synthyra/main/ap-lnImLk0hyiYBuz2kO6CmtF).

Cache schema v2 invalidates caches made before the geometry corrections.
Resumed optimizer checkpoints record cumulative elapsed time and hashes of
the selected head and donor validation report. A completed run without an
improved checkpoint is reported as `no_improvement` and cannot proceed to
publication.

Both revised real-model smoke checks also passed in one parallel Modal app.
The 300M worker used an H100; Modal fulfilled the second H100 request with
an H200. Both used BF16 autocast, SDPA, deterministic Torch algorithms, seed
17, three loops, and 15 sampling steps. Each preserved coordinates exactly
with confidence enabled or disabled, matched cached and live logits, and
produced finite gradients in 88 head parameter tensors. Parameters remained
unchanged and no optimizer update occurred.

| Model | Monomer head loss | Peak allocated GPU bytes | W&B record |
| --- | ---: | ---: | --- |
| ESMFold2-300 | 5.855031 | 1,557,954,560 | [300 audit smoke](https://wandb.ai/lhallee/fastplms-confidence/runs/a75699a4cdec) |
| ESMFold2-600 | 5.656716 | 2,048,769,024 | [600 audit smoke](https://wandb.ai/lhallee/fastplms-confidence/runs/86a72914eafb) |

The [parallel app](https://modal.com/apps/synthyra/main/ap-N3Ye6BcHhMfzayAy8uEZue)
completed after both checks. These losses are execution diagnostics on one
monomer, not held-out calibration or training results.

## Historical pre-training checks

The focused 71-test and smoke reports below are historical evidence from
before the current audit. They are retained for traceability and are not a
new test claim for the current run.
All 18 verified app IDs from the recent launches were stopped with zero
active tasks. No other active app with the pilot name remained. The shutdown
inventory is saved in
`artifacts/confidence/modal-stop-report.json`.

Both native models passed a zero-update smoke check on Modal with an NVIDIA
L4, PyTorch 2.13.0, Transformers 5.13.0, BF16 autocast, SDPA, and seed 17.
The panel contained `7duf_A` (67 residues) and the `7ruq` two-chain complex
(73 and 16 residues). It is an execution check, not a confidence-quality
evaluation or a throughput benchmark.

| Model | Head loss on the monomer | Peak allocated GPU bytes | W&B record |
| --- | ---: | ---: | --- |
| ESMFold2-300 | 5.859682 | 1,557,954,560 | [300 smoke check](https://wandb.ai/lhallee/fastplms-confidence/runs/19e5dc88b4c3) |
| ESMFold2-600 | 5.655051 | 2,048,769,024 | [600 smoke check](https://wandb.ai/lhallee/fastplms-confidence/runs/a6f16f8c713f) |

On the monomer, the checks verified cached versus live pLDDT and PAE logits,
exact equality of seeded coordinates and intermediate representations with
confidence on and off, finite gradients for 88 head parameter tensors,
unchanged head and positional-encoder parameter values, and no gradients in
the frozen encoders. Folding caches were generated for both targets.
No optimizer update occurred during these smoke checks. Training and held-out
quality evaluation completed later, as recorded above; publication remains pending.

The historical smoke checks found and fixed a dropped sample axis in cache
writing. Enabling deterministic Torch algorithms resolved the failed
coordinate repeatability check without relaxing its equality threshold. The
latest attempt to download the official `rcsb_multimer` archive failed because
of Google Drive quota exhaustion, and no official mirror was found. The
user-supplied copy resolved download access. The current training status is
recorded above.

## Scope and source records

The pilot trains one native confidence head for each of `esmfold2_300` and
`esmfold2_600`. Every folding and language-model parameter remains frozen.
Initialization uses the confidence subtree from
`biohub/ESMFold2-Experimental-Fast-Cutoff2025` at revision
`74b88548bf19688b8727432db0d698cb2e1d8783`. The expected subtree contains 93
tensors and 31,071,252 bytes. The native head is expected to have about 7.77M
trainable parameters. Keys and shapes must be checked before training.

The design follows the separate confidence-training stage described in the
[Biohub ESMFold paper](https://biohub.ai/papers/esm_protein.pdf). Experimental
structures, atom14 handling, and label conventions come from the pinned
[AtlasFold source](https://github.com/SeonghwanSeo/atlasfold/tree/444f376d85b9954a5f2f5f3f8b3cbcae1201ebb1)
and its [confidence-loss implementation](https://github.com/SeonghwanSeo/atlasfold/blob/444f376d85b9954a5f2f5f3f8b3cbcae1201ebb1/src/atlasfold/train/losses/confidence.py).
AtlasFold's head is not loaded into ESMFold2 because the architectures differ.

## Data and labels

The user supplied a copy of `rcsb_multimer` with Drive ID
`1K-yAbtbFvSYTQ2q8PGhO4d7LHrU3KvrG`. Download receipts record both this ID
and the official ID, plus the downloaded bytes and SHA-256. Its partial file
is separate from the interrupted official download to avoid mixing bytes
from different Drive objects. After the user enabled link sharing, Modal
downloaded and extracted the copy successfully:

- Archive bytes: `29329969577`
- SHA-256: `c29a7cf85dc9a4ca46523fa65b74e2a19dd382c90d08c2f15a209e86686b682a`
- [Download job](https://modal.com/apps/synthyra/main/ap-8Sf4P0KVVCmIkMsjo4KseR)

The archive also contains unused template databases. Preparation excludes
those files; they are not model inputs or training targets. Split construction
and frozen prediction caching must finish before optimization.

The preparation stage downloads only the approved experimental `rcsb`,
`rcsb_multimer`, and validation archives into the Modal Volume
`fastplms-confidence-pilot`. It excludes predicted and distillation data and
template databases. It selects 1,024 adaptation targets, split evenly between
monomers and two-chain biological complexes, with 64 to 384 residues, complete
standard-amino-acid chains, resolution from 0.1 to 3.0 Å, and at least four
resolved Cα atoms per chain. Validation contains 128 targets, also split 64/64.

The final test set contains 128 targets, split 64/64 between monomers and
dimers, held out from the AtlasFold experimental `rcsb` and `rcsb_multimer`
pools used for adaptation and validation. It is also sequence-cluster and
PDB-cluster disjoint from the 1,024 training targets and 128 validation targets
selected from the official pools. The split is held out from checkpoint
selection. Sequence overlap is filtered across splits with MMseqs2 at 40%
identity and 80% coverage, including each chain of a complex.
These exclusions separate the adaptation splits; they do not establish that
the original backbone, folding trunk, or donor head never saw related proteins.

Atom14 coordinates are mapped by residue identity and atom name into native
token order. Chain boundaries and unresolved-atom masks are retained.
Equivalent-chain permutations and crystallographic ambiguous-atom pairs are
resolved before labels are calculated.

The geometry audit corrected the row-oriented Kabsch transform, padding-mask
handling, and AtlasFold's exact local-frame convention. These corrections must
be covered by the remote audit before optimization.

Each model receives its own cache. Caches store FP32 inputs, pair states,
sampled coordinates, masks, mappings, and labels. Relative-position and
token-bond encodings are reconstructed from the frozen model. No cache or
model-derived representation is shared between 300M and 600M.

The pLDDT target is all-heavy-atom lDDT using 15 Å neighborhoods and 0.5, 1,
2, and 4 Å thresholds, with self-pairs and unresolved atoms masked. It uses
50 categorical bins. PAE uses the N-Cα-C local frame, 64 bins over 0 to 32 Å,
and masks invalid frames and unresolved target Cα atoms. Cα-only labels are
reported separately and are never broadcast onto atom-specific predictions.

## Training and evaluation

Folding uses the existing BF16 policy, SDPA, three recycling loops, 15
diffusion steps, and one diffusion sample. The head is trained with FP32
parameters and geometry/loss calculations, BF16 autocast for head computation,
AdamW at `1e-4`, weight decay `0.01`, one target per microbatch, accumulation
of 16, a timed cosine schedule with 100 warmup updates, a safety cap of
1,000,000 updates, and gradient clipping at 1.0. Validation runs every 100
updates with patience of 12 evaluations. Checkpoints include optimizer,
scheduler, RNG, data order state, and cumulative training-clock state so
interrupted runs resume within the approved duration.

The objective is normalized per target:

```text
pLDDT cross-entropy + 0.1 * PAE cross-entropy
```

The PAE weight is a pilot choice. The full-model confidence multiplier is not
used. The unchanged donor head and a label-frequency baseline are evaluated
before optimization. An eight-target overfit check must pass before the main
runs.

Reports include all-atom and Cα pLDDT MAE and calibration, PAE cross-entropy
and overflow fraction, pTM versus TM-score, and iPTM versus DockQ. Metrics are
collapsed by target before calculation; bootstrap intervals resample targets,
not atoms or pairs. Final-test values never select a checkpoint.
Two final-test samples per target also measure whether pLDDT selects the
sample with higher all-atom lDDT and whether iPTM selects the dimer with
higher DockQ. Reports include selection accuracy, quality regret, and ties.

The predeclared release gates are pLDDT MAE at most 0.10 on the 0 to 1 scale,
10-bin calibration error at most 0.05, target-level pLDDT/lDDT Spearman at
least 0.50, and iPTM/DockQ Spearman at least 0.30. The candidate must improve
both categorical cross-entropies, pLDDT and PAE, against the donor and
frequency baselines. Its pLDDT ranking may decline by at most 0.02 from the
donor. Each final-test
stratum must contain at least 64 eligible targets. A failed gate leaves the
checkpoint experimental and blocks publication.

## Modal execution and W&B

All tests and compute run in Modal. The launcher reads the existing secrets
file through the trusted environment loader. Secret values are never printed,
stored in reports, or passed as command-line arguments. Only `HF_TOKEN` and
`WANDB_API_KEY` enter the worker Secret; Modal credentials remain with the
launcher.
GPU workers enable deterministic Torch algorithms and set the cuBLAS
workspace configuration before execution. Seeded coordinate comparisons
retain exact equality checks.

Every training run must initialize online W&B in project `fastplms-confidence`
under the authenticated default entity. Run configuration, source revisions,
dataset and split hashes, exclusions, losses, metrics, throughput, peak
memory, elapsed time, spend estimates, checkpoints, and evaluation reports are
logged or uploaded as W&B artifacts. A run cannot train without successful
W&B initialization. The Modal Volume retains interrupted-run logs and
checkpoints.
Cache generation also initializes an online W&B run before loading the folding
model. It logs completed and remaining targets, throughput, elapsed time, peak
GPU memory, and failure details. Cache progress is separate from optimizer
metrics: the later `overfit` and `train` runs report losses and update counts.

The approved execution uses one H100 worker per model in parallel for about
ten hours each. The controller tracks duration and cost, enforces bounded
worker lifetimes, allows at most two GPU workers, and preserves resumable
checkpoints. There is no dollar allocation or stage-specific spending cap in
the current approval.
The ten-hour training-stage limit includes model setup, baseline evaluation,
optimization, and periodic validation. Resumes retain cumulative elapsed time.

The current entry point is:

```text
.cache/confidence-launcher/Scripts/python.exe -X utf8 -m tools.confidence.launch <stage> --model <model> --gpu H100 --options <JSON>
```

After split verification, the two-model campaign command is:

```text
.cache/confidence-launcher/Scripts/python.exe -X utf8 -m tools.confidence.launch campaign --gpu H100 --parallel --detach
```

`--parallel` dispatches one call for each model. `--detach` leaves those
bounded Modal calls running after the launcher exits. Dispatch records under
`artifacts/confidence/` retain the actual Modal app and function-call IDs.
Do not submit a second training command while those calls remain active.

Supported stages currently include `tests`, `lint`, `format`, `docs`, `prepare`,
`benchmark`, `cache`, `train`, `campaign`, `evaluate`, `package`, and `release`. The pre-training smoke check
uses the `prepare` phase `smoke`, followed by `benchmark` with
`--options '{"smoke":true}'` for each model. The packaging stage requires accepted
evaluation and checks the native artifact on Modal; it does not publish.
The `release` stage refreshes runtime source, generated cards, and measured
records, repeats isolated inference checks, and records the upload inventory
and current remote parent commit. It does not upload. The `docs` stage runs
documentation generators and their freshness checks remotely.
Use the launcher help and current
source for exact options rather than copying unverified command variants.

The `campaign` stage runs cache generation, the balanced eight-target overfit
check, and training in one remote worker. It stops if caching is incomplete or
the overfit check fails. Each worker has a 13-hour outer limit: up to two hours
for caching, 30 minutes for the overfit check, ten hours for training, and
30 minutes for overhead and cleanup. Container startup has a separate
15-minute limit. Checkpoints and stage
reports persist on the Modal Volume. Final evaluation and publication remain
separate stages.

## Publication boundary

The trained head is published only when that model independently passes every
acceptance gate and the reload, confidence-shape, multi-chain, seeded-coordinate,
and CIF-export checks. Publication changes checkpoint weights, so it is a
weight update rather than a files-only update. Until those conditions are met,
existing model cards and generated support reports accurately describe the
300M and 600M mirrors as confidence-disabled.

## Confidence heads v2 on a GH200 workstation

**Metrics review: recomputation required.** The validation, test, agreement,
and gate results below are historical, uncorrected records. The original
Spearman implementation assigned distinct ranks to tied values, including
ties introduced by bootstrap resampling. Corrected correlations, intervals,
and acceptance gates cannot be recovered from aggregates. The GH200
workstation is closed, and neither W&B run contains logged artifacts or raw
per-target predictions. Retain these tables as historical records only; do not cite
them as corrected quality or acceptance results. Model cards withhold the
numerical tables pending raw prediction recovery.

An exhaustive audit of each linked W&B run's 780 update rows found
`train/skipped_targets = 0` throughout. The skipped-target gradient bug
therefore did not affect these recorded training weights. That finding does
not validate the archived correlation estimates.

Training status: complete; metrics and acceptance: pending recomputation.
Both heads trained for all 780 planned updates, so both
schedules finished and both learning rates decayed to the floor. Each head was
evaluated once on the untouched test split, and production `esmfold2` scored
the same targets. No v2 weights are published, and the pilot heads and reports
above stay unchanged as baselines.

| Run | Updates | Hours | Selected checkpoint | Training record |
| --- | ---: | ---: | --- | --- |
| 300M, September 17 to 18, 2026 | 780 | 19.36 | `final-ema.safetensors` | [d5587e06871a](https://wandb.ai/lhallee/fastplms-confidence/runs/d5587e06871a) |
| 600M, September 17 to 18, 2026 | 780 | 19.04 | `final-ema.safetensors` | [01c7e12b6d03](https://wandb.ai/lhallee/fastplms-confidence/runs/01c7e12b6d03) |

Measured GPU hours: 1.74 of 2 for smoke checks, 21.97 of 24 for the 300M
model, 21.65 of 24 for the 600M model, and 2.13 of 5 for the production
reference.

Each run kept its final moving-average weights, which were also its best
validation checkpoint. Validation uses 256 held-out targets with four samples
each; these are validation numbers, not test results.

| Validation on 256 targets | 300M donor | 300M final | 600M donor | 600M final |
| --- | ---: | ---: | ---: | ---: |
| Total cross-entropy | 10.972 | 5.524 | 11.137 | 5.398 |
| pLDDT cross-entropy | 6.891 | 2.739 | 5.916 | 2.677 |
| PAE cross-entropy | 4.081 | 2.785 | 5.221 | 2.721 |
| Target pLDDT/lDDT Spearman | -0.499 | 0.942 | 0.106 | 0.935 |
| Calibration error | 0.350 | 0.0025 | 0.301 | 0.0024 |
| Within-target lDDT accuracy | 0.400 | 0.543 | 0.602 | 0.614 |
| Within-target ipTM accuracy | 0.496 | 0.530 | 0.526 | 0.544 |

The 300M validation compares 105 lDDT pairs and 383 ipTM pairs, the 600M
validation 83 and 327. No training target was skipped, and folding took 76 to
84 percent of each update.

The v2 campaign addresses three limits of the pilot: training stopped before
the learning rate decayed, one cached sample per target gave no signal for
ranking samples of one target, and the data covered only short monomers and
dimers. It runs on one GH200 (aarch64, 97.9 GB GPU memory, CUDA 13.0, PyTorch
2.13) with a budget of 2 GPU hours for smoke checks, 24 GPU hours per model for
training and test evaluation, and 5 GPU hours for the production reference.
`ledger.json` records every GPU stage, and a stage refuses to start once its
budget is spent.

### Data and splits

Targets come from the `rcsb` and `rcsb_multimer` configurations of
[Synthyra/AtlasFold-Data](https://huggingface.co/datasets/Synthyra/AtlasFold-Data).
A structure is eligible when its resolution is at most 4.0 Å, all residues
are standard amino acids, and each chain has at least four resolved C-alpha
atoms. Standard targets have at most 1,024 tokens. Larger assemblies become
whole-chain spatial subsets that grow by nearest C-alpha contact until the
token budget is full, because the folding API folds exactly the chains it
receives. Long variants of 1,025 to 2,048 tokens are built the same way and
are used only for evaluation. The pool holds 545,078 targets.

MMseqs2 `easy-cluster` (40% identity, 80% coverage) groups the 106,860 unique
chain sequences, pilot chains included, into 35,539 clusters. Targets that
share a cluster form one component. Held-out targets come only from
components with at most 50 standard targets and without a pilot record.
Training uses the other standard targets outside held-out components, except
targets that share a cluster with a pilot final-test chain. The split table
SHA-256 is `da801db539deabdc36d6060e5ef8570ca6cff15374a04dca5280bdcc31166bdd`.

| Stratum | Test | Validation | Train |
| --- | ---: | ---: | ---: |
| Monomer, at most 256 tokens | 64 | 32 | 229,635 |
| Monomer, 257 to 512 tokens | 64 | 32 | 141,549 |
| Monomer, 513 to 1,024 tokens | 64 | 32 | 31,333 |
| Homodimer | 96 | 48 | 36,648 |
| Heterodimer | 96 | 48 | 11,304 |
| Three or more chains | 128 | 64 | 25,490 |
| Long, 1,025 to 2,048 tokens | 64 | 0 | 0 |

Training draws a monomer or a multi-chain target with equal probability, then
a target in proportion to the mean inverse training-cluster size of its
chains.

### Training recipe

Each update folds 16 new targets with the frozen model at the defaults of
`model.fold`: three recycling loops, 50 diffusion steps, and four diffusion
samples per target. Short rollouts were rejected after a check on 12
validation targets: 15-step samples scored 0.013 higher all-atom lDDT than
50-step samples on every target (standard deviation 0.007), while head
predictions did not change, so a head trained on them would overestimate
pLDDT at inference. Every sample gets
its own chain permutation for identical chains and its own symmetric-atom
assignment before pLDDT and PAE labels are computed. The head then trains on
each sample in turn with this loss:

```text
pLDDT cross-entropy + PAE cross-entropy + 0.5 * within-target ranking loss
```

The ranking loss is a pairwise logistic loss at temperature 0.05. It orders
the samples of one target by true lDDT through their expected mean pLDDT,
and, for multi-chain targets, by the TM score of true inter-chain aligned
errors through their predicted ipTM. Pairs need a quality difference of at
least 0.01. Partner scores come from a separate no-gradient pass, so the
per-sample gradients sum to the gradient of the joint loss.

Optimization uses AdamW at `1e-4` with weight decay `0.01`, gradient clipping
at 1.0, linear warmup over 10% of the planned updates (at most 300), and
cosine decay to `1e-5` over the planned updates. An exponential moving
average of the weights has a horizon of 10% of the planned updates. There is
no early stopping. Validation scores the moving-average weights on 256 cached
validation targets every 90 minutes. The final moving-average weights are
selected unless their validation cross-entropy exceeds the best validation
checkpoint by more than 2%. Every run logs online to W&B project
`lhallee/fastplms-confidence`, group `esmfold2-confidence-v2`.

### Test evaluation and gates

Each test target is folded once with five samples at the `model.fold`
defaults, and the v2, pilot, and donor heads score the same samples.
Production `esmfold2` folds and scores its own samples of the same targets.
Structure quality uses all-atom lDDT, TM-score from TM-align on C-alpha atoms,
and DockQ averaged over native interfaces. The 512 standard targets give the
headline estimates; the 64 long targets are reported per stratum. Intervals
come from one target bootstrap shared by every head, so differences between
heads are paired.

A v2 head is accepted only if it passes three gates fixed before evaluation:

1. Against the pilot, target pLDDT/lDDT Spearman is not lower, calibration
   error is not higher, and no reported metric is significantly worse.
2. For sample selection, the lower 95% bound of within-target pairwise
   accuracy exceeds 0.5 for pLDDT and for ipTM against DockQ, and top-1 regret
   is below random choice.
3. Against production `esmfold2`, each Spearman correlation is at most 0.03
   lower, calibration error at most 0.01 higher, and each within-target
   accuracy at most 0.03 lower.

Disorder metrics are reported for every head but are not gates. Residues
whose C-alpha atom is missing from the experimental structure stand in for
disordered regions: the evaluation reports the AUROC of low pLDDT for telling
unresolved from resolved residues, the mean pLDDT of each group, and the
fraction of each group below pLDDT 50. No head receives pLDDT labels on
unresolved atoms, so these metrics test whether low confidence generalizes.
Missing density also marks flexible loops, tags, and crystal-packing effects,
so the metrics are a directional check rather than a disorder benchmark.
These metrics were added after the smoke checks and before any test
evaluation ran.

### Historical results requiring recomputation

Every head scored the same folded samples of the 512 standard test targets,
five samples per target, at three recycling loops and 50 diffusion steps.
Production `esmfold2` folded and scored its own five samples of the same
targets. No target was skipped in any evaluation.

| Metric | 300M v2 | 300M pilot | 300M donor | 600M v2 | 600M pilot | 600M donor | Production |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pLDDT/lDDT Spearman | 0.881 | 0.818 | -0.459 | 0.854 | 0.803 | -0.035 | 0.687 |
| pTM/TM-score Spearman | 0.838 | 0.789 | -0.039 | 0.863 | 0.741 | -0.093 | 0.751 |
| ipTM/DockQ Spearman | 0.816 | 0.729 | -0.163 | 0.853 | 0.633 | -0.384 | 0.759 |
| Atom pLDDT MAE | 0.0796 | 0.0925 | 0.3987 | 0.0808 | 0.0904 | 0.3426 | 0.0809 |
| Calibration error | 0.0038 | 0.0222 | 0.3605 | 0.0050 | 0.0046 | 0.3087 | 0.0477 |
| pLDDT cross-entropy | 2.729 | 2.971 | 7.011 | 2.689 | 2.792 | 6.025 | 3.262 |
| PAE cross-entropy | 2.815 | 2.958 | 4.132 | 2.748 | 2.929 | 5.285 | 3.131 |
| Within-target lDDT accuracy | 0.589 | 0.504 | 0.425 | 0.480 | 0.473 | 0.452 | 0.767 |
| Within-target ipTM/DockQ accuracy | 0.603 | 0.572 | 0.500 | 0.501 | 0.449 | 0.546 | 0.759 |
| Top-1 regret | 0.0210 | 0.0260 | 0.0236 | 0.0175 | 0.0201 | 0.0162 | 0.0177 |
| Random-choice regret | 0.0247 | 0.0247 | 0.0247 | 0.0186 | 0.0186 | 0.0186 | 0.0321 |
| Disorder AUROC | 0.858 | 0.851 | 0.447 | 0.874 | 0.874 | 0.620 | 0.888 |
| Resolved residue mean pLDDT | 0.752 | 0.764 | 0.375 | 0.790 | 0.784 | 0.430 | 0.864 |
| Unresolved residue mean pLDDT | 0.492 | 0.493 | 0.410 | 0.505 | 0.510 | 0.400 | 0.565 |

Within-target accuracy counts sample pairs whose measured quality differs by
at least 0.01 lDDT or 0.05 DockQ. The 300M samples give 409 lDDT pairs and 474
ipTM pairs, the 600M samples 279 and 405, and the production samples 510 and
758. Each model scores its own samples, so the pair counts differ.

Historical, uncorrected paired 95% intervals for the v2 heads, from one bootstrap of 1,000 target
resamples shared by every head:

| Metric | 300M v2 | 600M v2 |
| --- | ---: | ---: |
| pLDDT/lDDT Spearman | 0.852 to 0.903 | 0.821 to 0.882 |
| pTM/TM-score Spearman | 0.805 to 0.865 | 0.835 to 0.886 |
| ipTM/DockQ Spearman | 0.767 to 0.854 | 0.818 to 0.880 |
| Atom pLDDT MAE | 0.0769 to 0.0823 | 0.0773 to 0.0846 |
| Calibration error | 0.0024 to 0.0076 | 0.0030 to 0.0102 |
| Within-target lDDT accuracy | 0.507 to 0.672 | 0.392 to 0.567 |
| Within-target ipTM/DockQ accuracy | 0.537 to 0.667 | 0.422 to 0.586 |
| Top-1 regret | 0.0165 to 0.0263 | 0.0144 to 0.0208 |
| Disorder AUROC | 0.836 to 0.879 | 0.854 to 0.894 |

Spearman correlation between each head's confidence and production
`esmfold2`'s confidence, over the per-target mean of each model's five
samples. The two models fold different samples, so this compares per-target
scores rather than scores of one structure. ipTM uses the 320 multi-chain
targets.

| Score | 300M v2 | 300M pilot | 300M donor | 600M v2 | 600M pilot | 600M donor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean pLDDT | 0.690 | 0.690 | -0.208 | 0.713 | 0.717 | 0.191 |
| pTM | 0.806 | 0.814 | -0.089 | 0.790 | 0.823 | 0.236 |
| ipTM | 0.740 | 0.747 | -0.174 | 0.696 | 0.674 | -0.110 |

Against production over the same targets, the 300M v2 head reports -0.081 mean
pLDDT, -0.039 pTM, and +0.006 ipTM on average, and the 600M v2 head reports
-0.055, -0.020, and +0.018.

Historical gate outcomes, pending recomputation:

| Gate | 300M | 600M |
| --- | --- | --- |
| 1. Against the pilot | pass | fail |
| 2. Sample selection | pass | fail |
| 3. Production parity | fail | fail |

These are the original gate decisions, not corrected acceptance results.
The full gate analysis remains pending raw prediction recovery and
recomputation. Neither head is approved for publication, and no v2 weights
are published.

Per-head numbers, paired intervals, per-stratum results including the 64 long
targets, split identity, and training configuration are recorded in
[docs/evidence/confidence/esmfold2_300-v2.json](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/resolve/07cd9e4fee7aeb18ff9d2ce2078f9092fa8ed3f3/docs/evidence/confidence/esmfold2_300-v2.json)
and
[docs/evidence/confidence/esmfold2_600-v2.json](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/resolve/07cd9e4fee7aeb18ff9d2ce2078f9092fa8ed3f3/docs/evidence/confidence/esmfold2_600-v2.json).

### Kernels

The frozen folding model runs with cuEquivariance triangle kernels and no
chunking. Heads keep the default PyTorch kernels, unchunked up to 1,024 tokens
and in 256-row chunks above. Chunking does not change head outputs. The kernel
check compared settings on one 512-token monomer and one 1,024-token dimer
(300M, BF16 autocast, FP32 parameters, SDPA, four samples):

| Setting at 1,024 tokens | Fold | Training step | Step peak memory |
| --- | ---: | ---: | ---: |
| PyTorch, 32-row chunks | 59.0 s | 40.0 s | 68.2 GiB |
| PyTorch, unchunked | 12.7 s | 6.7 s | 39.7 GiB |
| cuEquivariance, unchunked | 7.1 s | 2.9 s | 30.7 GiB |

Pair representations from cuEquivariance differed from the 32-row PyTorch
fold by 2.2% (relative norm). An unchunked PyTorch fold of the same input
differed by 2.1%, so the difference is within run-to-run variation. On one
stored rollout, cuEquivariance head logits differed by at most 0.22, and
unchunked PyTorch head logits were identical.

### Smoke checks

- Parity: on one monomer, one homodimer, and one heterodimer, true
  coordinates and resolved-atom masks matched the pilot's name-based mapping
  exactly, and per-atom lDDT labels differed by at most 0.0013. Head logits
  from the reconstructed inputs matched the native confidence path of the
  same forward call exactly.
- Training rate (300M, 50 steps, four samples, ranking on): 64 targets drawn
  by the training sampler took 5.37 s each (standard error 0.66) at a mean of
  389 tokens, about 42 updates per hour. Each model trains for at most 20.5
  hours with 780 planned updates; two hours of its budget stay free for the
  test evaluation.
- Probes (300M, 15 steps, learning rate `1e-4`, 40 planned updates, 30
  minutes, 64 validation targets): with the ranking loss, 34 updates reached
  validation cross-entropy 6.231 (pLDDT 3.102, PAE 3.129), target pLDDT
  Spearman 0.881, and calibration error 0.028. Without it, 36 updates reached
  6.203, 0.887, and 0.028. The donor head started at 11.184, -0.435, and 0.350.
  Within-target accuracy stayed near chance in both probes (lDDT 0.40 over 25
  pairs; ipTM 0.44 and 0.45 over 77 pairs), so the ranking loss showed no
  early effect either way and stays in the recipe.
- Throughput (300M, fast kernels, four samples, 15 steps): folding took 1.3 s at 384
  tokens, 6.1 s at 1,024 tokens, and 23.1 s at 2,048 tokens. The 600M model
  took 1.3 s, 6.2 s, and 23.0 s. The training step at 1,024 tokens took 6.6 s
  for 600M with 40.1 GiB peak memory.
- Overfit (300M, 12 fixed targets of at most 256 tokens, learning rate
  `3e-4`, 150 updates): pLDDT cross-entropy fell from 8.89 to 1.12 and PAE
  cross-entropy from 4.72 to 1.93. All 7 ranked ipTM pairs were ordered
  correctly, against 1 of 7 before training. No sample pair differed in lDDT
  by the 0.02 evaluation margin, so the recorded result shows `passed: false`
  under the earlier pLDDT-only criterion; the criterion now counts pLDDT and
  ipTM pairs together.
- Samples of one target rarely differ in lDDT: spreads were 0.002 to 0.005 at
  128 to 1,024 tokens. Interface quality varies more; one 1,536-token complex
  had true ipTM from 0.27 to 0.54. The ranking loss therefore acts mostly on
  multi-chain targets.

### Commands

Run from the repository root on the controlling machine. The SSH helper
copies source files, delivers `HF_TOKEN` and `WANDB_API_KEY` without printing
them, starts detached tmux jobs, and waits for a job to exit:

```bash
python -m tools.confidence.ssh --host ubuntu@<address> --identity <key path> sync
python -m tools.confidence.ssh --host ubuntu@<address> --identity <key path> start train-300 train --model esmfold2_300 --run v2 --planned-updates <updates> --hours 21
python -m tools.confidence.ssh --host ubuntu@<address> --identity <key path> wait train-300
```

The workstation stages are `pilot-artifacts`, `pool`, `splits`, `smoke`,
`train`, `evaluate`, `reference`, and `download-reference`. `evaluate` and
`reference` accept `--split validation --limit <n>` for a dry run that never
folds or scores a test target.

The SSH `wait` command returns the recorded job exit status after printing its
log tail, so a failed remote stage also fails the controlling shell command.
Job names use letters, digits, underscores, and hyphens, beginning with a letter
or digit.

### Preserve evaluation records

Use one evaluation ID to group the two model evaluations and the production
reference. A missing ID generates a new one. Each model destination is reserved
before model loading, and existing destinations are refused:

```bash
python -m tools.confidence.host evaluate --model esmfold2_300 --run v2 --evaluation-id review-rerun
python -m tools.confidence.host evaluate --model esmfold2_600 --run v2 --evaluation-id review-rerun
python -m tools.confidence.host reference --evaluation-id review-rerun
```

These commands allocate GPU work and score the already spent test split. They
are documented for an explicitly authorized rerun, not as a prerequisite for
CPU metric correction. Outputs live under
`~/data/confidence-v2/evaluation/review-rerun/<model>/`. They retain raw sample
predictions, skipped targets, split inputs, source and environment identity,
and the hashes of the exact head checkpoint snapshots loaded for evaluation.
Completion and failure records distinguish a finished evaluation from a partial
run. Checkpoint snapshots stay local unless separately approved for publication.

Verify and export an evaluation for durable storage:

```bash
python -m tools.confidence.experiment_artifacts verify --evaluation-dir <evaluation/model>
python -m tools.confidence.experiment_artifacts export --evaluation-dir <evaluation/model> --output-dir <new-public-directory>
```

The export checks the recorded hashes and copies the JSON records and
identities into a new directory. It excludes checkpoint weights and makes no
network calls. Review the bundle, publish it to the public artifact dataset,
and pin its revision and file identities before closing the compute host.

### Recompute metrics from saved predictions

The v2 calculation module uses NumPy and SciPy without importing training,
folding, Torch, or the target-pool loader. Given saved records for
`esmfold2_300`, `esmfold2_600`, and `esmfold2`, recompute summaries, paired
bootstrap intervals, acceptance gates, and production agreement on a CPU:

```bash
python -m tools.confidence.recompute --evaluation-dir <evaluation-group> --output-dir <new-correction-directory> --evidence-dir docs/evidence/confidence
```

`--evidence-dir` is optional. When supplied, corrected evidence copies go under
the new output directory; original evidence is preserved. The command records
input and calculation-source hashes, versions, bootstrap seed, and cohort
counts. It does not refold structures or change weights. The test set remains
spent, and validation correlations used during checkpoint selection still need
their own saved-cache rescore. Historical v2 raw predictions remain unavailable,
so this command cannot currently correct those published historical numbers.
