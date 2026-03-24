# Fine-Tuning Guide

FastPLMs models can be fine-tuned for downstream tasks using LoRA (recommended) or full fine-tuning via the HuggingFace `Trainer` API. This guide is based on `fastplms/fine_tuning_example.py`.

## Quick Start

```python
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

# Load with a classification head
model = AutoModelForSequenceClassification.from_pretrained(
    "Synthyra/ESM2-150M",
    trust_remote_code=True,
    num_labels=1,  # regression
)

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

## Dataset Classes

### Single-Sequence Classification/Regression

`SequenceDatasetHF` wraps a HuggingFace `datasets` split:

```python
from datasets import load_dataset

dataset = load_dataset("your/dataset")
train_data = SequenceDatasetHF(
    dataset["train"],
    col_name="sequence",
    label_col="label",
    max_length=2048,
)
```

### Protein-Protein Interaction (Pair)

`PairDatasetHF` handles two-sequence inputs (e.g., PPI binding affinity):

```python
train_data = PairDatasetHF(
    dataset["train"],
    col_a="protein_a",
    col_b="protein_b",
    label_col="pkd",
    max_length=2048,
)
```

For pair tasks, a custom collate function tokenizes both sequences and concatenates them with the tokenizer's separator handling.

## Trainer Configuration

The example uses a shared `BASE_TRAINER_KWARGS` dict:

```python
BASE_TRAINER_KWARGS = {
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "save_strategy": "steps",
    "save_steps": 500,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "report_to": "none",
    "label_names": ["labels"],
}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=1e-4,
    **BASE_TRAINER_KWARGS,
)
```

## Metrics

### Regression (Spearman Correlation)

```python
from scipy.stats import spearmanr

def compute_regression_metrics(eval_pred: EvalPrediction):
    predictions = eval_pred.predictions.flatten()
    labels = eval_pred.label_ids.flatten()
    rho, p_value = spearmanr(predictions, labels)
    return {"spearman_rho": rho, "spearman_p": p_value}
```

### Classification (Confusion Matrix)

```python
from sklearn.metrics import confusion_matrix

def compute_classification_metrics(eval_pred: EvalPrediction):
    predictions = eval_pred.predictions.argmax(axis=-1)
    labels = eval_pred.label_ids
    cm = confusion_matrix(labels, predictions)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}
```

## Training

```python
from transformers import Trainer, EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_regression_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
```

## LoRA Configuration

The recommended defaults from `fastplms/fine_tuning_example.py`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `r` | 16 | Rank of LoRA matrices |
| `lora_alpha` | 32 | Scaling factor |
| `target_modules` | `"all-linear"` | Applies LoRA to all linear layers (valid in peft >= 0.7) |
| `lora_dropout` | 0.1 | Dropout on LoRA paths |
| `bias` | `"none"` | Do not train bias terms |
| `task_type` | `"SEQ_CLS"` | Sequence classification task |

## Model-Specific Notes

- **ESM2, ESM++, DPLM, DPLM2**: Standard tokenizer-based fine-tuning via `AutoModelForSequenceClassification`
- **E1**: Requires custom collation because it uses sequence mode (no standard tokenizer). You need to override the Trainer's data collation to call `model.model.prep_tokens.get_batch_kwargs()` instead of the standard tokenizer
- **ESMC (ESM++)**: Ensure `sequence_id` is included in the forward pass inputs when batching

## Saving and Loading

```python
# Save LoRA adapter
model.save_pretrained("./lora_adapter")

# Load later
from peft import PeftModel

base_model = AutoModelForSequenceClassification.from_pretrained(
    "Synthyra/ESM2-150M",
    trust_remote_code=True,
    num_labels=1,
)
model = PeftModel.from_pretrained(base_model, "./lora_adapter")
```
