---
id: foundry.logging
kind: library_module
status: active
aliases: [foundry.logging]
---
# foundry.logging

How a run reaches Weights and Biases, and what survives it locally. An experiment is the
recorded intent and a run is the W&B object inside it, so a run's group is the experiment id,
`YYYY-MM-DD_slug`, and `ws capture` pulls the group back into `experiments/` by that id.

Every run follows the [W&B convention](../../docs/conventions/wandb.md), which `Run` enforces: a few
readable charts per training job, and no run for tuning, evaluation or diagnostics.

```python
@track(project="atlas-ppi", job_type="train")
def train(config: Config, *, run: Run) -> dict[str, float]:
    for step, batch in enumerate(run.batches(loader)):   # piloted with WS_PILOT=1
        run.log({"train/loss": loss}, step=step)
    return {"test/auroc": auroc}
```

What it does for you: loads the secrets file when the W&B key is missing, seeds from `config.seed`,
pilots the first loop (`WS_PILOT=1` stops it after the timed window and makes no W&B run), and
writes `cost/seconds`, `cost/peak_gpu_gib` and the pilot's `cost/step_seconds`,
`cost/input_wait_share` and `cost/projected_hours` into the summary, outside the key budget.

| Name | Does |
|---|---|
| `@track(project=..., job_type=..., **run_options)` | Wraps a training function `f(config, *, run)`: opens a `Run` under `directory=` or `config.output_dir / "wandb"`, revised from the function's file, and finishes it `completed` with the returned scalars as summary, or `failed` when it raises |
| `Run(directory, config, *, project, revision=None, group, job_type, max_keys=40, system_metrics=False, pilot=None, ...)` | One resumable W&B run, also a context manager. Sends the config redacted, with `source_revision` added, and W&B's git capture and system charts off. Resumes from `run_file` with `resume="must"`. Appends every row to `history.jsonl`, fsynced, and sends W&B only its real scalars, refusing nested values, malformed keys and keys past the budget. Tuning, evaluation, diagnostic, package and release jobs stay local unless `mode` is passed, and a local-only record is never resumed online. A pilot keeps its records in `pilots/<time>/`, so the full run after it starts clean. Keys have at most three `/` segments, and a dot only joins digits (`coverage@0.95`). `finish(status=...)` writes `run_summary.json` and exits 0 for a planned stop, 1 for `failed`. W&B's own files go to `directory` when it is named `wandb`, else to `directory/wandb/`, never `wandb/wandb/` (`wandb_root`) |
| `Run.batches(loader, *, total_steps=None, items=None, item_unit="items")` | The loop under a `foundry.pilot.Pilot` timed from when the run opened: a pilot stops after the window, a full run goes on; the report goes to `pilot.json` and the summary. Later loops pass through |
| `Run.table(name, rows)` | Per-entity values as one W&B table, one key of the budget, also written to `tables/` |
| `Run.trainer_callback()`, `trainer_metrics(logs)` | Hugging Face Trainer logs into the run as `train/`, `val/` and `test/`, without throughput counters; set `report_to="none"` |
| `source_revision(root, include=("src", "scripts", "configs", "pyproject.toml"))` | `Revision`: the git commit for a clean checkout rooted at `root`, else a tree hash. The tree hash normalizes line endings and skips READMEs and AGENTS.md, so the workspace, a Windows clone, and a Linux clone agree |
| `revision_of(path)` | The revision of the nearest project whose sources hold `path`, else of the directory holding it |
| `redact(values)`, `is_credential(name)` | Drop credential-shaped keys at any depth. `token` and `key` count only as the last segment, so `hf_token` goes and `token_budget` stays |

Promoted from contact_esmc6b's `DurableExperiment` and `Experiment`, with
base_model_distillation's step metric and planned stops, and `ws capture`'s redaction.
`foundry` never imports `tools`; `tools/workspace/capture.py` imports `redact` from here.

**Stays in the projects:** numerical-backend identity, artifact receipts, and rank checks. A
non-main rank passes `mode="disabled"`. The secrets loader is `foundry.secrets_env`.

New runs receive `lifecycle:active` and a `role:` tag derived from their recorded job type,
unless the caller already supplies those tags. An explicit group wins; otherwise a valid
`config.experiment_id` or the run directory's dated experiment parent supplies the group.
Unrecognized identifiers do not invent an experiment. Roles are shared with `ws wandb` through
`organization.py`; job types themselves stay unchanged. Project routing and archive policy
live in the [W&B guide](../../docs/guides/wandb.md#organization-and-archives).

Tests: `tests/tier1_unit/test_logging_*.py`, with a stand-in for `wandb`, so they never reach
the network.
