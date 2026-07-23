# Fine-tuning

FastPLMs models follow Transformers `PreTrainedModel` conventions, so
compatible family and task-head combinations can use Trainer, Accelerate,
distributed, and adapter workflows. Core training dependencies are isolated in
the `train` extra. FastPLMs 1.0 supports Python 3.11-3.14 with PyTorch 2.13 and
Transformers 5.13. CPU is suitable for tiny contract runs; practical checkpoint
fine-tuning normally requires a CUDA accelerator whose memory fits the selected
model, optimizer state, and batch.

```bash
uv sync --extra train
```

The runnable classification and regression example currently targets ESM2,
whose artifacts advertise `AutoModelForSequenceClassification`. Plotting is
disabled by default, so a normal run needs only the `train` extra. To request
plots, add the separate reporting dependency and `--plot-results`:

```bash
uv run \
  --extra train \
  --extra reporting \
  python examples/fine_tuning.py \
  --task classification \
  --model_path Synthyra/ESM2-8M \
  --model-revision 185ecbd45665d050a8dae326d91886d330c5f9d0 \
  --classification-dataset-source GleghornLab/DL2_reg \
  --classification-dataset-revision 7e18f1b98859b0a3e3da283f63d0a153b774cf1f \
  --attn-backend sdpa \
  --output-dir artifacts/fine-tuning \
  --batch_size 8 \
  --epochs 2 \
  --seed 7 \
  --full-determinism \
  --plot-results
```

Omit `--extra reporting` and `--plot-results` for the default plot-free run.
Generated plots are saved at 300 dpi as
`<task-output>/classification_results.png` or
`<task-output>/regression_results.png`. Existing plot files are never replaced.

Regression uses three independently pinned dataset snapshots:

```bash
uv run \
  --extra train \
  python examples/fine_tuning.py \
  --task regression \
  --model_path Synthyra/ESM2-8M \
  --model-revision 185ecbd45665d050a8dae326d91886d330c5f9d0 \
  --regression-train-dataset-source Synthyra/ProteinProteinAffinity \
  --regression-train-dataset-revision f4a51e5e9f2c2a0185693f9fbcffc02d9dae08db \
  --regression-validation-dataset-source Synthyra/AffinityBenchmarkv5.5 \
  --regression-validation-dataset-revision 826ccfb1488d52b7b361802fbde161373247d084 \
  --regression-test-dataset-source Synthyra/haddock_benchmark \
  --regression-test-dataset-revision 4e22f014745728fca2d9c10f2f2cfd5a29a4981c \
  --attn-backend sdpa \
  --output-dir artifacts/fine-tuning \
  --seed 7 \
  --full-determinism
```

The exact shipped sources above use those pinned revisions automatically.
Custom remote sources reject omitted revisions, branches, and tags. For a fully
local run, pass existing local model and dataset directory paths through
`--model_path` and the corresponding `--*-dataset-source` options. Revisions
may be omitted for local directories because the example records an immutable
SHA-256 over the complete tree and rejects any tree that changes between model
initialization and final persistence. Pre-populate every pinned model and
dataset in the cache before starting a network-isolated Hub-backed run.
Local dataset directories must be layouts accepted by `datasets.load_dataset`;
the example does not currently accept arbitrary `Dataset.save_to_disk()` trees.

The classification source must resolve to a `DatasetDict` with the required
`train`, `valid`, and `test` splits. Every split must be non-empty and
contain a `seqs` column of non-empty protein strings plus an integer `labels`
column. Training labels must be the contiguous zero-based set `0..K-1`;
validation and test labels must be subsets of the training labels. A local
file-backed layout can therefore use `train.csv`, `valid.csv`, and `test.csv`
in one directory, each with this header:

```text
seqs,labels
MKT...,0
GAV...,1
```

Regression uses three separate sources. Each source must expose a non-empty
`train` split with non-empty string columns `SeqA` and `SeqB` and a `labels`
column containing only finite real numbers. A local directory for each source
can contain `train.csv` with this header:

```text
SeqA,SeqB,labels
MKT...,GAV...,-8.42
```

All split, column, and label contracts are checked before model initialization.
Malformed data cannot allocate a model or begin training.

## Sequence tasks

Tokenize proteins with the tokenizer belonging to the exact checkpoint revision.
Construct labels only for the task being trained and mask padding or ignored
positions explicitly. For residue-level tasks, align labels to biological
residues rather than assuming tokenizer position equals residue position.
For paired proteins, filter using the exact tokenizer encoding including
special tokens. The collator applies the same `max_length` with longest-first
truncation at the boundary. Throughout the CLI and manifest, `max_length` is an
encoded token budget, not a biological-residue count. It includes all
tokenizer-added BOS, EOS, separator, and other special tokens for a single
sequence or complete pair.

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "Synthyra/ESM2-150M"
model_revision = "979e0880dfc9e0c0080839b83d9d2dc05b92786a"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    revision=model_revision,
    trust_remote_code=True,
)
model = AutoModelForMaskedLM.from_pretrained(
    model_id,
    revision=model_revision,
    trust_remote_code=True,
    attn_implementation="sdpa",
)
```

E1 is the exception: it has no tokenizer and must use its native raw-sequence
preparation adapter.

## Precision and attention

Select a declared attention backend explicitly for a reproducible training run.
Do not rely on a silent fallback. Record the resolved backend, Torch and
Transformers versions, CUDA environment, checkpoint revision, tokenizer hashes,
dtype, optimizer, seed, and data fingerprint.

The runnable example exposes `eager`, `sdpa`, and `flex_attention` through
`--attn-backend` and defaults to `sdpa`. It intentionally rejects
FlashAttention because the compact CLI does not expose the explicit BF16 CUDA
model-loading and placement policy that Flash training requires. The current
GH200/aarch64 validation environment is not expected to provide compatible
bundled Flash kernels, and this workflow does not recommend source-building
them. Prefer SDPA for the default training path and use Flex only on supported
Torch platforms. Any Flash gradient evidence remains a separate,
device-specific compliance result. `--output-dir` names a parent directory;
regression and classification are written into separate task- and
LoRA-specific children so Trainer state, final artifacts, and manifests do not
collide.

Every selected task child must be absent before the command starts. The CLI
preflights all selected children, then each training function atomically
creates its own child before loading a model or dataset. An existing file,
directory, or broken symlink is rejected rather than reused. If an in-process
run fails, its newly reserved child is removed; an interrupted process leaves a
reservation marker and partial output that a later run refuses to mix. Inspect
and explicitly relocate or remove such a failed run before retrying.

ESMFold2 FP8 is inference-only. Gradient-enabled ESMC execution reloads BF16,
and canonical training checkpoints contain only BF16 or FP32 weights. Runtime
Transformer Engine quantization state is never serialized.

## Parameter-efficient adaptation

Low-rank adapters are appropriate when full-model optimization is unnecessary.
Record exact target module names and verify that only intended parameters have
`requires_grad=True`. A separately trained task head, such as `classifier`, must
also be listed in PEFT's `modules_to_save`; setting `requires_grad=True` alone
does not include that head in the adapter checkpoint. Save adapter configuration
and base checkpoint revision together. The runnable example enforces this by
requiring the adapter configuration and safetensors payload, reloading into the
same immutable base, and comparing every persisted adapter and task-head tensor
hash. Full fine-tuning verifies the complete model state, including buffers. It
separately compares logits from the prepared Trainer and the
independently reloaded model on the first one or two held-out rows.

LoRA is enabled by default in the runnable example. Pass `--no-use-lora` to
fine-tune the full model instead.

## Reproducibility and evaluation

Split homologous proteins at an identity threshold appropriate to the task to
reduce sequence leakage. Report class balance, length distribution, ambiguous
residue handling, truncation, and any structure-derived labels. Use task-specific
metrics and retain per-sequence predictions so aggregate improvements can be
audited.

The CLI rejects moving Hub references before loading a model or dataset. The
example refuses a pre-existing task output before loading either source and
writes `run_manifest.json` with the requested immutable model source;
resolved base-weight and FastPLMs runtime revisions; tokenizer identity and
vocabulary hash; attention backend; parameter and compute dtypes; Trainer
device, optimizer, and scheduler; Torch, Transformers, PEFT, Datasets, CUDA,
Python, and package environment; command line and normalized configuration;
seeds and deterministic settings; adapter configuration and trainable target
modules; and each dataset's immutable source, split, consumed columns, row
count, and `ordered_rows_sha256` over ordered post-filter values.

Final persistence is part of the run contract. The example saves into a
temporary sibling directory, requires the expected PEFT configuration and
safetensors weights, reloads that staged artifact against the same immutable
base, and refuses to continue if any persisted training-state hash changes. A
deterministic reload check then compares the
prepared Trainer's logits with independently reloaded-model logits for the
first `min(2, len(test_dataset))` rows, using the same collator, device, and
autocast policy with dtype-specific tolerances. Only after both checks pass is
the staging directory atomically renamed to `final_model`, without overwriting
an existing artifact. The manifest records the final
tree SHA-256, verified parameter hashes, `reload_verified: true`, and the
`held_out_inference` comparison metrics. The subsequent full
`trainer.predict(test_dataset)` evaluation deliberately remains on the original
prepared Trainer; it is not a full-dataset pass through the independently
reloaded instance. Retain the manifest, final artifact, and per-sequence
evaluation outputs together when making biological claims.

The runnable minimal pattern is in [`examples/fine_tuning.py`](../examples/fine_tuning.py).
Model-family compliance establishes that the starting checkpoint is represented
correctly; it does not validate a fine-tuned model's biological claims.
