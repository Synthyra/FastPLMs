# ESMFold2 confidence-head training

ESMFold2-300 and ESMFold2-600 include Synthyra-trained native confidence heads
in their checkpoint weights, enabled by default. They return pLDDT, PAE, pTM,
and iPTM without downloading a separate head. The current release is **v1**;
**pilot** denotes the earlier baseline. The backbone and folding model remain
frozen during confidence training.

## Training and data

Both heads completed 780 updates, about 18 hours per model, using the final
exponential moving-average checkpoint. Training logs are available for
[300M](https://wandb.ai/lhallee/fastplms-confidence/runs/9558b6d23daf) and
[600M](https://wandb.ai/lhallee/fastplms-confidence/runs/820d2cfa56c0).
Final validation cross-entropy was 5.51839 and 5.39807, respectively.

Targets come from the `rcsb` and `rcsb_multimer` configurations of
[AtlasFold-Data](https://huggingface.co/datasets/Synthyra/AtlasFold-Data), pinned
in the evidence records. Eligible experimental structures have resolution at
most 4.0 Å, standard amino acids, and at least four resolved C-alpha atoms per
chain. Whole-chain spatial subsets limit large assemblies to the token budget.
Training covers monomers, dimers, and larger complexes up to 1,024 tokens;
1,025–2,048-token targets are evaluated separately and are not training targets.

MMseqs2 clustering uses 40% sequence identity and 80% coverage. Targets sharing
a chain cluster form a component. Validation and test components exclude pilot
records; training also excludes clusters represented in the pilot final test.
The prepared pool contains 545,078 targets, 106,860 unique chain sequences,
and 35,543 clusters, with 475,969 eligible training targets. This pool size is
not the number of distinct targets visited during training.

| Stratum | Train | Validation | Standard test | Long test |
| --- | ---: | ---: | ---: | ---: |
| Monomer, up to 256 tokens | 229,619 | 32 | 64 | 0 |
| Monomer, 257–512 tokens | 141,576 | 32 | 64 | 0 |
| Monomer, 513–1,024 tokens | 31,318 | 32 | 64 | 0 |
| Homodimer | 36,652 | 48 | 96 | 0 |
| Heterodimer | 11,300 | 48 | 96 | 0 |
| Three or more chains | 25,504 | 64 | 128 | 0 |
| Long, 1,025–2,048 tokens | 0 | 0 | 0 | 64 |

Each update samples 16 targets with equal monomer/complex probability and
inverse-cluster weighting. The frozen model generates four samples per target,
using three recycling loops and 50 diffusion steps. Chain permutations and
symmetric-atom assignments are resolved separately for every sample before
computing confidence labels. The objective is pLDDT cross-entropy plus PAE
cross-entropy plus 0.5 times the within-target ranking loss.

AdamW uses weight decay 0.01 and gradient clipping at 1.0. The learning rate
warms up for 78 updates to 0.0001, then decays to 0.00001. EMA decay is
0.9871794871794872. Validation uses 256 targets; training completes the planned
schedule without early stopping. Checkpoints and exact training identities are
persisted throughout training.

## Evaluation methods

The headline evaluation contains 512 standard targets with five predictions
per target, three recycling loops, and 50 diffusion steps. Both small models
also completed all 64 long targets. Production ESMFold2 completed all standard
targets and 46 long targets; 18 long targets exceeded GPU memory. Long-target
comparisons with production therefore require the shared subset or an explicit
statement that the target sets differ.

**This is a previously used test split, not a new untouched benchmark.** The
current campaign reconstructs the earlier protocol with a new clustered split;
it does not reproduce the original target assignments or promise identical
weights across GPU architectures. The current metrics are recomputed from
preserved predictions with corrected tied ranks and target-bootstrap intervals.

Folding uses SDPA, FP32 parameters, BF16 autocast, BF16 backbone computation,
and fast folding kernels. Seeds follow the recorded target order. Metrics
compare pLDDT with all-atom lDDT, pTM with C-alpha TM-align TM-score, and ipTM
with DockQ averaged over native interfaces. Headline correlations use individual
diffusion samples; each bootstrap draw keeps a target's samples together.
Intervals below are 95% intervals
from 1,000 target-bootstrap resamples. Comparisons between heads on the same
model use the same bootstrap samples.

Ranking accuracy compares samples of one target, requiring quality gaps of at
least 0.02 lDDT or 0.05 DockQ. Top-1 regret is the quality gap between the best
sample and the confidence-selected sample, using lDDT for monomers and DockQ
for complexes. Three complexes with undefined DockQ are omitted from selection
metrics. Confidence and error values use a 0–1 scale; exported CIF pLDDT uses
0–100. Lower error, cross-entropy, and regret are better; higher correlation,
ranking accuracy, and AUROC are better.

## Standard evaluation results

| Metric | ESMFold2-300 | ESMFold2-600 |
| --- | ---: | ---: |
| pLDDT–all-atom lDDT Spearman | 0.88003 [0.85090, 0.90258] | 0.85185 [0.81670, 0.87953] |
| pTM–TM-score Spearman | 0.84004 [0.80621, 0.86693] | 0.86219 [0.83103, 0.88565] |
| ipTM–DockQ Spearman | 0.81403 [0.76544, 0.85119] | 0.84507 [0.81196, 0.87120] |
| Atom pLDDT mean absolute error | 0.07867 [0.07617, 0.08124] | 0.07950 [0.07625, 0.08277] |
| Calibration error, 10 bins | 0.00426 [0.00241, 0.00831] | 0.00567 [0.00293, 0.01027] |
| pLDDT cross-entropy | 2.72722 [2.68388, 2.77285] | 2.68398 [2.62732, 2.73986] |
| PAE cross-entropy | 2.81677 [2.76512, 2.86575] | 2.75040 [2.69624, 2.79981] |
| Within-target pLDDT ranking accuracy | 0.58505 [0.50357, 0.66201] | 0.50962 [0.41981, 0.59490] |
| Within-target ipTM/DockQ ranking accuracy | 0.57173 [0.50567, 0.63475] | 0.50993 [0.43450, 0.58957] |
| Top-1 selection regret | 0.02312 [0.01814, 0.02899] | 0.01946 [0.01503, 0.02496] |
| Random-selection regret | 0.02556 | 0.02029 |
| Unresolved-residue detection AUROC | 0.85591 [0.83319, 0.87786] | 0.87270 [0.85106, 0.89298] |

Unresolved-residue detection uses missing C-alpha density as a proxy for
disorder. Missing density also reflects flexible loops, tags, and crystal
packing; this is not a definitive disorder benchmark. Full per-stratum,
resolved/unresolved residue, baseline, and paired-difference statistics are
available in the evidence records.

## Agreement with production ESMFold2

These Spearman correlations compare each model's mean confidence over five
samples per target. pLDDT and pTM use 512 targets; ipTM uses the 320 multichain
targets. Differences are the small model minus production.

| Confidence score | ESMFold2-300 correlation | ESMFold2-600 correlation | 300M mean difference | 600M mean difference |
| --- | ---: | ---: | ---: | ---: |
| Mean pLDDT | 0.68423 | 0.71212 | -0.07777 | -0.05294 |
| pTM | 0.79666 | 0.78433 | -0.03602 | -0.01344 |
| ipTM | 0.73161 | 0.68299 | +0.01022 | +0.02579 |

Production predicts its own structures. These values describe agreement in
confidence across targets, not agreement on identical structures or evidence
that the smaller models fold more accurately. The recovered production records
contain the summaries and sufficient statistics needed for these comparisons;
production inference was not rerun.

## Long-target results

Both columns contain the same 64 long targets, reported separately from the
headline evaluation. These estimates do not include the production model.

| Metric | ESMFold2-300 | ESMFold2-600 |
| --- | ---: | ---: |
| pLDDT–all-atom lDDT Spearman | 0.92216 | 0.88840 |
| pTM–TM-score Spearman | 0.64717 | 0.77608 |
| ipTM–DockQ Spearman | 0.51071 | 0.77263 |
| Atom pLDDT mean absolute error | 0.09081 | 0.09278 |
| Calibration error, 10 bins | 0.01900 | 0.01609 |
| pLDDT cross-entropy | 3.17160 | 3.01707 |
| PAE cross-entropy | 2.41084 | 2.62461 |
| Within-target pLDDT ranking accuracy | 0.60000 | 0.70000 |
| Within-target ipTM/DockQ ranking accuracy | 0.50538 | 0.73684 |
| Top-1 selection regret | 0.03346 | 0.01436 |
| Random-selection regret | 0.02759 | 0.01795 |
| Unresolved-residue detection AUROC | 0.85143 | 0.87522 |

## Prespecified acceptance criteria

The experiment recorded three scientific criteria before evaluation. They
remain part of the research record and do not imply production equivalence.

| Criterion | ESMFold2-300 | ESMFold2-600 |
| --- | --- | --- |
| Improves on pilot without a significant regression | Pass | Fail |
| Reliable within-target sample selection | Pass | Fail |
| Within production tolerances | Fail | Fail |

The pilot comparison requires pLDDT/lDDT correlation not to decrease,
calibration error not to increase, and no significant regression among the
reported metrics. The 600M calibration point estimate is slightly worse than
pilot; its paired difference interval includes zero.

Sample selection requires both ranking-accuracy lower confidence bounds to
exceed 0.5 and top-1 regret to improve on random selection. Both 600M lower
bounds remain below 0.5. Production tolerances allow correlations to decrease
by at most 0.03, calibration error to increase by at most 0.01, and ranking
accuracy to decrease by at most 0.03. Both models meet the correlation and
calibration tolerances but miss the ranking tolerances. Confidence quality
across targets does not establish reliable selection between near-tied samples
of one target.

## Records and reproducibility

The [public artifact dataset](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts)
stores the training reports, raw evaluation records, manifests, exact checkpoint
identities, and statistical analysis. The
[verified release evidence](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/d7039a56e732e2de7685abefdbea57dfc6a9da13/docs/evidence/confidence)
includes both completed GPU artifact checks. `evidence.toml` pins the release reports
at `docs/evidence/confidence/esmfold2_300-v1.json` and
`docs/evidence/confidence/esmfold2_600-v1.json`; fetch them with:

```bash
python -m tools.artifacts.evidence_store fetch
```

The final heads are merged into the original frozen model state. Packaging
verifies every non-head tensor is unchanged and every embedded head tensor
matches the evaluated checkpoint. Offline reload checks cover default-enabled
confidence, exact head identity, confidence ranges, seeded coordinate equality,
and CIF confidence export. Both models passed all six checks on an RTX PRO
6000 Blackwell using two 56-residue chains, one and two diffusion samples,
15 sampling steps, three recycling loops, seed 17, SDPA, FP32 parameters, and
BF16 computation. This is a focused packaging and inference check, not a full
structure benchmark. The manifest preserves the original frozen training
base independently of the published checkpoint revision.

For traceability, these immutable historical locations retain their original
names. They do not name additional public confidence releases:

- [Preparation and source records](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/c7aa9df21f9142e8a0c3303ba4340c0a16674e66/confidence-v2/v2-reproduction-20260922).
- [Production raw evaluation](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/156619fb1b15a58e1f112bcd5ef0aeeac18d4bf2/confidence-v2/v2-reproduction-20260922/public/evaluation/esmfold2).
- [300M raw evaluation](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/309f353b07e0e46de4d77a5266d4eddd695538e3/confidence-v2/v2-reproduction-20260922/public/evaluation/esmfold2_300).
- [600M raw evaluation](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/6e62186cd36b9047cc4691980076be9f76482192/confidence-v2/v2-reproduction-20260922/public/evaluation/esmfold2_600).
- [Pilot evidence](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/07cd9e4fee7aeb18ff9d2ce2078f9092fa8ed3f3/docs/evidence/confidence).
- [Earlier experimental chronicle](https://github.com/Synthyra/FastPLMs/blob/9e9ecb0/docs/confidence_training.md), including pilot settings, historical workstation runs, and debugging records.

The earlier GH200 checkpoints and raw predictions could not be recovered.
Their historical reports remain marked `requires_recomputation` and must not
be cited as corrected current results. The current v1 heads have new training
and evaluation records. Archived raw keys and operational filenames containing
`v2` remain unchanged to preserve source hashes and auditability.

## Remote workflows

Confidence tests and compute run remotely. Use the bounded CPU verification
workflow for code changes:

```bash
python -m tools.verification.cpu --output-root artifacts/verification/confidence
```

The existing launcher uses authenticated Modal, Hugging Face, and W&B SDKs.
For a new campaign with an explicitly authorized compute budget:

```bash
PYTHONPATH=src:. python -m tools.confidence.launch_v2 prepare --campaign <new-campaign>
PYTHONPATH=src:. python -m tools.confidence.launch_v2 start --campaign <new-campaign>
```

RTX PRO 6000 Blackwell had the lowest measured training cost per update in the
bounded seven-GPU comparison. The panel used 16 targets, four samples, three
loops, 50 diffusion steps, FP32 weights, and BF16 autocast. The repeated
measurement was about 86 seconds and $0.09 per update for each model, excluding
startup, validation, uploads, and final evaluation. This is a panel result,
not a guaranteed full-run cost. The
[hardware comparison archive](https://huggingface.co/datasets/Synthyra/FastPLMs-artifacts/tree/1e0efe9637e3850065a1831dcb4ecc142e9852ca/confidence-v2/v2-reproduction-20260922)
contains all seven GPU measurements and their environments.

Evaluation destinations are immutable and preserve checkpoint snapshots,
requests, raw target records, and completion manifests. A resumed interrupted
evaluation verifies saved records and keeps the original target seeds and
remaining budget. Verify and export the evidence before closing compute:

```bash
python -m tools.confidence.experiment_artifacts verify --evaluation-dir <evaluation/model>
python -m tools.confidence.experiment_artifacts export --evaluation-dir <evaluation/model> --output-dir <new-public-directory>
```

`tools/confidence/v2_analysis.py` computes metrics without loading a model.
Reanalysis of saved predictions uses a fresh output directory:

```bash
python -m tools.confidence.recompute --evaluation-dir <evaluation-group> --output-dir <new-correction-directory> --evidence-dir docs/evidence/confidence
```

The archived summaries support correlation, calibration, ranking, and bootstrap
analysis. They do not preserve full confidence arrays, logits, or predicted
coordinates. Analyses requiring those outputs need a separately planned run.
