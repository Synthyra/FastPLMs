# Fine-tuning

FastPLMs models follow Transformers `PreTrainedModel` conventions, so
compatible family and task-head combinations can use Trainer, Accelerate,
distributed, and adapter workflows. Core training dependencies are isolated in
the `train` extra.

```bash
uv sync --extra train
```

The runnable classification and regression example currently targets ESM2,
whose artifacts advertise `AutoModelForSequenceClassification`. It also uses
plotting and evaluation packages, which can be resolved without adding them to
the runtime package:

```bash
uv run \
  --extra train \
  --with matplotlib \
  --with scikit-learn \
  --with scipy \
  --with seaborn \
  python examples/fine_tuning.py \
  --task classification \
  --model_path Synthyra/ESM2-8M \
  --batch_size 8 \
  --epochs 2
```

## Sequence tasks

Tokenize proteins with the tokenizer belonging to the exact checkpoint revision.
Construct labels only for the task being trained and mask padding or ignored
positions explicitly. For residue-level tasks, align labels to biological
residues rather than assuming tokenizer position equals residue position.

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "Synthyra/ESM2-150M"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(
    model_id,
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

ESMFold2 FP8 is inference-only. Gradient-enabled ESMC execution reloads BF16,
and canonical training checkpoints contain only BF16 or FP32 weights. Runtime
Transformer Engine quantization state is never serialized.

## Parameter-efficient adaptation

Low-rank adapters are appropriate when full-model optimization is unnecessary.
Record exact target module names and verify that only intended parameters have
`requires_grad=True`. Save adapter configuration and base checkpoint revision
together. Reload into the same architecture and run a held-out inference check
before treating the artifact as usable.

## Reproducibility and evaluation

Split homologous proteins at an identity threshold appropriate to the task to
reduce sequence leakage. Report class balance, length distribution, ambiguous
residue handling, truncation, and any structure-derived labels. Use task-specific
metrics and retain per-sequence predictions so aggregate improvements can be
audited.

The runnable minimal pattern is in [`examples/fine_tuning.py`](../examples/fine_tuning.py).
Model-family compliance establishes that the starting checkpoint is represented
correctly; it does not validate a fine-tuned model's biological claims.
