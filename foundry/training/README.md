---
id: foundry.training
kind: library_module
status: active
aliases: [foundry.training]
---
# foundry.training

Training helpers that do not depend on the model: seeding, gradient clipping, the names
trainers use for precisions, and the off-diagonal negative masks of paired batches.

| Name | Does |
|---|---|
| `set_seed(seed)` | Seeds `random`, NumPy's global generator, and torch, whose `manual_seed` seeds every device |
| `clip_grad_norm(model, max_norm)` | Clips the gradients to a global L2 norm and returns the norm before clipping |
| `AutoGradClipper(model, clip_percentile=10, history_length=1000000)` | `clip_gradients()` records the gradient norm and, from the tenth call on, clips to that percentile of the recorded norms and returns it; before then it returns 0.0 and leaves the gradients alone |
| `precision_map` | `bf16`, `fp16`, and `fp32` to their torch dtypes |
| `pair_masks.OffDiagonalMask(tables, device, seed=0)` | `batch(rows, step)` returns the `(b, b)` bool `(positive, eligible)` masks of a paired batch, the interface of the original Atlas `PositiveMask`. Cell `(i, j)` pairs row `i`'s left protein with row `j`'s right protein; the diagonal is the positive. A candidate negative lies in its row's taxonomy pair, is removed when its two proteins sit in the clusters of a known positive's two proteins (either orientation), and the rest are thinned per metadata stratum. Every check is a gather or `searchsorted` over tables, on the device |
| `pair_masks.MaskTables`, `load_tables(path)` | The arrays a mask reads: each protein's species, 0.7 cluster, source bits, length bin, locations and GO-BP terms, the sorted keys of the known positive protein pairs and cluster pairs, and the keep fraction of each of 80 strata. `production_suite.ppi_batch_mask` builds and saves them for a release |

Promoted on 2026-09-24. `set_seed` was copied in ten places, `AutoGradClipper` in six, and the
rest in the `training.py` that Atlas, DatasetDev, and synth each held. The copies that now
import from here:

- Atlas's `utils/training.py`, which keeps `initialize_external_services` and its trainer classes
  (`TrainerLogger`, `ModelCheckpointer`; they subclass `LocalMetricLogger` and `StateCheckpointer`
  since 2026-10-06). DatasetDev's copy, which nothing imported, was retired on 2026-09-26, and
  synth's `synth.utilities.training`, which Atlas's supersets, on 2026-10-06.
- confounders' `training/utils.py` and `data/biogrid.py`, adversarial_invariance's `utils.py`,
  speedrunning_plms' `training/schedules_and_timing.py` (named `training/utils.py` until
  2026-10-05), contact_esmc6b's `atlasv2/runtime.py`, DSM's `evaluation/utils.py`, and
  embedding_translation's `helpers/utils.py`.

Two behaviors changed. Atlas's, DatasetDev's, and synth's `set_seed` printed
`[pipeline] set_seed() called with seed=<seed>`, and no longer do. adversarial_invariance's and
speedrunning_plms' clippers returned None before their tenth call, and now return 0.0, as the
others did. adversarial_invariance never reads the value, and speedrunning_plms only tests it.
contact_esmc6b's and speedrunning_plms' extra CUDA seeding repeated what `torch.manual_seed`
already does. ProJEPA's `set_seed` stays in ProJEPA, because it leaves NumPy unseeded.

Added on 2026-10-01: `pair_masks`, for the rule that a PPI release holds positives only and the
negatives of a batch are built in the batch
([decision](../../docs/decisions/serving/production_suite/2026-10-01_ppi_positives_only.md),
[convention](../../docs/conventions/paired_sampling.md)).

Tests: `tests/tier1_unit/test_training.py` and `tests/tier1_unit/test_training_pair_masks.py`, with
each project's own tests as the guard on the names it imports.

## The trainer

Promoted on 2026-10-05 from Atlas's trainer package. Import the submodules directly, because
`foundry.training.trainer` is not re-exported from the package `__init__`.

| Module | Holds |
|---|---|
| `trainer.Trainer` | The one loop: patching, batching, grad accumulation, DDP, precision, clipping, divergence checks, early stopping, resume, `reset_for_trial`. Subclasses supply `get_model`, `get_datasets`, `get_data_loaders`, `train_step`, `eval_step`, and may override the hooks, among them `eval_interval` (optimizer steps between validations), `initial_best_metric` and `compile_model` (static shapes by default) |
| `config.TrainingArguments` | Neutral defaults; Atlas's subclass sets its own lr, batch, patch and tracked metric |
| `prefetch` | `PatchAccumulator`, `SizedPatchAccumulator` (counts a loader of known length and flags its last group, so the epoch's leftover gradient steps at its end), `KeyedPatchAccumulator`, `SynchronousGroupPrefetcher`, `AsyncGroupPrefetcher`, `build_prefetcher` |
| `callbacks`, `exceptions`, `compilation`, `distributed`, `schedulers`, `metric_logger`, `checkpointing` | Callback protocol and `PeriodicTestEvaluation`, `TrainingDivergedError`, compile mode, DDP helpers, `WarmupCosineScheduler`, `LocalMetricLogger`, `StateCheckpointer` |

Quirks kept so Atlas results are unchanged: an epoch-limited run starts one epoch more than `num_epochs`
and stops after its first micro-step; when the loader length is a multiple of the patch group size, a
leftover gradient at epoch end joins the next epoch's first step; the gradient norm is measured only when
clipping is on; the default optimizer skips parameters with `requires_grad` False; schedulers take
`step(current_step=...)`.

confounders' `BiogridBinaryTrainer` runs on it since 2026-10-06, 294 seeded runs of the old and new trainer equal
([the move](../../docs/decisions/interaction/confounders/2026-10-06_trainer_on_foundry.md)); its metric logger and
checkpointer are small classes of its own, because `metrics.log` and `best_model.pth` keep the paper's formats.

Atlas's `TrainerLogger` and `ModelCheckpointer` (`utils/training.py`) subclass `LocalMetricLogger` and
`StateCheckpointer` since 2026-10-06 and add only what Atlas needs: the logger starts or joins a W&B run and makes it
the sink, and the checkpointer exports the inference artifact and uploads the best checkpoint to the Hub. Two
behaviors changed: checkpoints are written atomically, and the `logger_init` row of `metrics.log` no longer names the
project and run (the `logger_initialized` row still does).

`TODO(foundry_trainer)`: three trainers are not single-model loops and run their own: Protify's HF `Trainer` (which
also trains InterpNet), embedding_translation's `GANTrainer` and `ModelTrainer` (a generator and discriminators, each
with an optimizer), and base_model_distillation's `MultiRecipeTrainer` (one step for several students). They move
when `Trainer` can drive several models and optimizers.

Tests: `tests/tier1_unit/test_training_trainer.py`, `test_training_prefetch.py`, `test_training_components.py`,
and Atlas's `tests/test_trainer_parity.py`.
