#! /usr/bin/env python3
"""
This script shows how to fine-tune a Synthyra FastPLM model for protein
sequence regression or classification.
For regression we look at the binding affinity of two proteins (pkd)
For classification we look at the solubility of a protein (membrane bound or not)
"""

import argparse
import contextlib
import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping
from functools import wraps
from importlib import metadata
from numbers import Integral, Real
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_MODEL = "Synthyra/ESM2-8M"
DEFAULT_MODEL_REVISION = "185ecbd45665d050a8dae326d91886d330c5f9d0"
DEFAULT_CLASSIFICATION_DATASET = "GleghornLab/DL2_reg"
DEFAULT_CLASSIFICATION_DATASET_REVISION = "7e18f1b98859b0a3e3da283f63d0a153b774cf1f"
DEFAULT_REGRESSION_TRAIN_DATASET = "Synthyra/ProteinProteinAffinity"
DEFAULT_REGRESSION_TRAIN_DATASET_REVISION = "f4a51e5e9f2c2a0185693f9fbcffc02d9dae08db"
DEFAULT_REGRESSION_VALIDATION_DATASET = "Synthyra/AffinityBenchmarkv5.5"
DEFAULT_REGRESSION_VALIDATION_DATASET_REVISION = "826ccfb1488d52b7b361802fbde161373247d084"
DEFAULT_REGRESSION_TEST_DATASET = "Synthyra/haddock_benchmark"
DEFAULT_REGRESSION_TEST_DATASET_REVISION = "4e22f014745728fca2d9c10f2f2cfd5a29a4981c"
CLASSIFIER_MODULE_NAME = "classifier"
EXAMPLE_ATTENTION_BACKENDS = ("eager", "sdpa", "flex_attention")
_OUTPUT_RESERVATION_FILE = ".fastplms-output-reservation.json"
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_P = ParamSpec("_P")
_R = TypeVar("_R")
_PINNED_DEFAULT_REVISIONS = {
    ("model", DEFAULT_MODEL): DEFAULT_MODEL_REVISION,
    ("dataset", DEFAULT_CLASSIFICATION_DATASET): DEFAULT_CLASSIFICATION_DATASET_REVISION,
    ("dataset", DEFAULT_REGRESSION_TRAIN_DATASET): (DEFAULT_REGRESSION_TRAIN_DATASET_REVISION),
    ("dataset", DEFAULT_REGRESSION_VALIDATION_DATASET): (
        DEFAULT_REGRESSION_VALIDATION_DATASET_REVISION
    ),
    ("dataset", DEFAULT_REGRESSION_TEST_DATASET): DEFAULT_REGRESSION_TEST_DATASET_REVISION,
}

# Shared arguments for the trainer
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


def _output_path_exists(path: Path) -> bool:
    """Return true for files, directories, and broken symlinks."""

    return os.path.lexists(path)


@contextlib.contextmanager
def _reserved_output_directory(output_dir: str | Path) -> Iterator[Path]:
    """Atomically reserve a new run directory and clean it after a failed run."""

    destination = Path(output_dir).expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            "Refusing to start because the task output path already exists and could "
            f"mix state from another run: {destination}"
        ) from error

    reservation_token = uuid.uuid4().hex
    reservation_path = destination / _OUTPUT_RESERVATION_FILE
    try:
        with reservation_path.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "reservation_token": reservation_token,
                },
                handle,
                sort_keys=True,
            )
            handle.write("\n")
    except BaseException:
        destination.rmdir()
        raise

    try:
        yield destination
    except BaseException as error:
        try:
            reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
            if reservation.get("reservation_token") != reservation_token:
                raise RuntimeError(
                    "The output reservation identity changed while the run was active; "
                    f"preserving {destination} for manual inspection."
                )
            shutil.rmtree(destination)
        except BaseException as cleanup_error:
            error.add_note(
                "FastPLMs could not clean the failed run's reserved output directory: "
                f"{cleanup_error}"
            )
        raise
    else:
        reservation_path.unlink()


def _guard_training_output(
    *,
    lora_default: str,
    full_default: str,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Reserve a task output before the decorated function performs any work."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        function_signature = inspect.signature(function)

        @wraps(function)
        def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            bound = function_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            output_dir = bound.arguments["output_dir"]
            if output_dir is None:
                output_dir = lora_default if bound.arguments["use_lora"] else full_default
            with _reserved_output_directory(output_dir) as reserved:
                bound.arguments["output_dir"] = reserved
                return function(*bound.args, **bound.kwargs)

        return guarded

    return decorate


def _ensure_output_paths_available(paths: list[Path]) -> None:
    """Preflight all requested task outputs before a multi-task CLI starts."""

    collisions = [str(path) for path in paths if _output_path_exists(path)]
    if collisions:
        raise FileExistsError(
            "Refusing to start because task output paths already exist and could mix "
            f"prior state: {collisions}"
        )


def _ensure_classifier_persistence(lora_config: Any) -> Any:
    """Ensure PEFT saves the independently trained classification head."""
    modules_to_save = list(lora_config.modules_to_save or ())
    if CLASSIFIER_MODULE_NAME not in modules_to_save:
        modules_to_save.append(CLASSIFIER_MODULE_NAME)
    lora_config.modules_to_save = modules_to_save
    return lora_config


# Dataset classes
class PairDatasetHF(TorchDataset):
    """
    Dataset class for protein pair data (e.g., protein-protein interactions).

    Args:
        data: The dataset containing protein sequences and labels
        col_a: Column name for the first protein sequence
        col_b: Column name for the second protein sequence
        label_col: Column name for the labels
        max_length: Encoded token budget for the complete pair, including
            tokenizer-added separator and special tokens
    """

    def __init__(
        self, dataset: Any, col_a: str, col_b: str, label_col: str, max_length: int = 2048
    ):
        self.seqs_a = dataset[col_a]
        self.seqs_b = dataset[col_b]
        self.labels = dataset[label_col]
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.seqs_a)

    def __getitem__(self, idx: int) -> tuple[str, str, float | int]:
        # Token-budget filtering and any defensive truncation belong to the
        # tokenizer-aware collator. Slicing both strings to max_length could
        # still create a pair longer than the model context and ignores special
        # tokens inserted between the proteins.
        seq_a = self.seqs_a[idx]
        seq_b = self.seqs_b[idx]
        label = self.labels[idx]
        return seq_a, seq_b, label


class SequenceDatasetHF(TorchDataset):
    """
    Dataset class for single protein sequence data.

    Args:
        dataset: The dataset containing protein sequences and labels
        col_name: Column name for the protein sequences
        label_col: Column name for the labels
        max_length: Encoded token budget including tokenizer-added special tokens
    """

    def __init__(
        self,
        dataset: Any,
        col_name: str = "seqs",
        label_col: str = "labels",
        max_length: int = 2048,
    ):
        self.seqs = dataset[col_name]
        self.labels = dataset[label_col]
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> tuple[str, float | int]:
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq, label


def _tokenization_kwargs(max_length: int | None, *, pair: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "padding": "longest",
        "return_tensors": "pt",
    }
    if max_length is not None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        kwargs.update(
            {
                "truncation": "longest_first" if pair else True,
                "max_length": max_length,
            }
        )
        if max_length % 8 == 0:
            kwargs["pad_to_multiple_of"] = 8
    else:
        kwargs["pad_to_multiple_of"] = 8
    return kwargs


def _encoded_length(tokenizer: Any, sequence: str, pair: str | None = None) -> int:
    encoded = tokenizer(
        sequence,
        pair,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


def _fits_token_budget(
    tokenizer: Any,
    sequence: str,
    pair: str | None,
    max_length: int,
) -> bool:
    """Include tokenizer-added control tokens in the context-length gate."""

    return _encoded_length(tokenizer, sequence, pair) <= max_length


class PairCollator:
    """
    Collator for protein pair data that handles tokenization and tensor conversion.

    Args:
        tokenizer: The tokenizer to use for encoding sequences
        regression: Whether this is a regression task (True) or classification (False)
        max_length: Encoded token budget for the complete pair, including
            tokenizer-added separator and special tokens
    """

    def __init__(
        self,
        tokenizer: Any,
        regression: bool = False,
        max_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.regression = regression
        self.max_length = max_length

    def __call__(self, batch: list[tuple[str, str, float | int]]) -> dict[str, torch.Tensor]:
        seqs_a, seqs_b, labels = zip(*batch, strict=True)
        labels = torch.tensor(labels)
        labels = labels.float() if self.regression else labels.long()
        tokenized = self.tokenizer(
            seqs_a,
            seqs_b,
            **_tokenization_kwargs(self.max_length, pair=True),
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }


class SequenceCollator:
    """
    Collator for single protein sequence data that handles tokenization and tensor conversion.

    Args:
        tokenizer: The tokenizer to use for encoding sequences
        regression: Whether this is a regression task (True) or classification (False)
        max_length: Encoded token budget including tokenizer-added special tokens
    """

    def __init__(
        self,
        tokenizer: Any,
        regression: bool = False,
        max_length: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.regression = regression
        self.max_length = max_length

    def __call__(self, batch: list[tuple[str, float | int]]) -> dict[str, torch.Tensor]:
        seqs, labels = zip(*batch, strict=True)
        labels = torch.tensor(labels)
        labels = labels.float() if self.regression else labels.long()
        tokenized = self.tokenizer(
            seqs,
            **_tokenization_kwargs(self.max_length, pair=False),
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }


def _tree_sha256(path: Path) -> str:
    """Hash every regular byte in a local model or dataset tree."""

    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Local reproducibility source must be a directory: {root}")
    digest = hashlib.sha256()
    files = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for file_path in files:
        if file_path.is_symlink():
            raise ValueError(f"Local reproducibility sources may not contain symlinks: {file_path}")
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_path.stat().st_size.to_bytes(8, "big"))
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _immutable_source_identity(
    source: str,
    revision: str | None,
    *,
    source_kind: str,
) -> dict[str, Any]:
    """Resolve a Hub commit or immutable local tree, rejecting moving refs."""

    local_path = Path(source).expanduser()
    if local_path.exists():
        return {
            "kind": "local_tree",
            "source_kind": source_kind,
            "path": str(local_path.resolve(strict=True)),
            "tree_sha256": _tree_sha256(local_path),
        }
    if local_path.is_absolute() or source.startswith((".", "~")):
        raise FileNotFoundError(f"Local {source_kind} source does not exist: {source}")
    if revision is None:
        revision = _PINNED_DEFAULT_REVISIONS.get((source_kind, source))
    if revision is None or _IMMUTABLE_REVISION.fullmatch(revision) is None:
        raise ValueError(
            f"Remote {source_kind} source {source!r} requires an immutable 40-character "
            "Hub commit revision; branches, tags, and omitted revisions are rejected."
        )
    return {
        "kind": "hub",
        "source_kind": source_kind,
        "repo_id": source,
        "revision": revision.lower(),
    }


def _load_dataset_immutable(
    source: str,
    revision: str | None,
    *,
    split: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    identity = _immutable_source_identity(
        source,
        revision,
        source_kind="dataset",
    )
    kwargs: dict[str, Any] = {}
    if identity["kind"] == "hub":
        kwargs["revision"] = identity["revision"]
    if split is not None:
        kwargs["split"] = split
    return load_dataset(source, **kwargs), identity


def _available_columns(dataset: Any) -> set[str]:
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None and not isinstance(column_names, Mapping):
        return {str(column) for column in column_names}
    if isinstance(dataset, Mapping):
        return {str(column) for column in dataset}
    features = getattr(dataset, "features", None)
    if isinstance(features, Mapping):
        return {str(column) for column in features}
    raise TypeError("Dataset must expose column_names, features, or mapping keys.")


def _require_dataset_columns(
    dataset: Any,
    *,
    split: str,
    required: tuple[str, ...],
) -> None:
    available = _available_columns(dataset)
    missing = sorted(set(required).difference(available))
    if missing:
        raise ValueError(
            f"Dataset split {split!r} is missing required columns {missing}; "
            f"available columns are {sorted(available)}."
        )
    if len(dataset) == 0:
        raise ValueError(f"Dataset split {split!r} must contain at least one row.")


def _validate_sequence_column(dataset: Any, *, split: str, column: str) -> None:
    for row_index, sequence in enumerate(dataset[column]):
        if not isinstance(sequence, str) or not sequence.strip():
            raise ValueError(
                f"Dataset split {split!r} column {column!r} row {row_index} must "
                "contain a non-empty protein sequence string."
            )


def _classification_label_set(dataset: Any, *, split: str) -> set[int]:
    labels: set[int] = set()
    for row_index, label in enumerate(dataset["labels"]):
        if isinstance(label, bool) or not isinstance(label, Integral):
            raise ValueError(
                f"Dataset split {split!r} label at row {row_index} must be an "
                f"integer, got {label!r}."
            )
        labels.add(int(label))
    return labels


def _validate_classification_dataset_dict(data: Any) -> int:
    """Validate the complete classification schema before model initialization."""

    if not isinstance(data, Mapping):
        raise TypeError(
            "Classification data must be a DatasetDict-style mapping with train, "
            "valid, and test splits."
        )
    required_splits = ("train", "valid", "test")
    missing_splits = [split for split in required_splits if split not in data]
    if missing_splits:
        raise ValueError(
            "Classification data is missing required splits: "
            f"{missing_splits}; expected train, valid, and test."
        )

    label_sets: dict[str, set[int]] = {}
    for split in required_splits:
        dataset = data[split]
        _require_dataset_columns(
            dataset,
            split=split,
            required=("seqs", "labels"),
        )
        _validate_sequence_column(dataset, split=split, column="seqs")
        label_sets[split] = _classification_label_set(dataset, split=split)

    train_labels = label_sets["train"]
    expected_train_labels = set(range(len(train_labels)))
    if train_labels != expected_train_labels:
        raise ValueError(
            "Classification training labels must be contiguous zero-based integers; "
            f"observed {sorted(train_labels)}, expected {sorted(expected_train_labels)}."
        )
    for split in ("valid", "test"):
        unseen = sorted(label_sets[split].difference(train_labels))
        if unseen:
            raise ValueError(
                f"Classification split {split!r} contains labels absent from train: {unseen}."
            )
    return len(train_labels)


def _validate_regression_dataset(dataset: Any, *, split: str) -> None:
    """Validate a protein-pair regression split before model initialization."""

    _require_dataset_columns(
        dataset,
        split=split,
        required=("SeqA", "SeqB", "labels"),
    )
    _validate_sequence_column(dataset, split=split, column="SeqA")
    _validate_sequence_column(dataset, split=split, column="SeqB")
    for row_index, label in enumerate(dataset["labels"]):
        if (
            isinstance(label, bool)
            or not isinstance(label, Real)
            or not math.isfinite(float(label))
        ):
            raise ValueError(
                f"Regression split {split!r} label at row {row_index} must be a "
                f"finite real number, got {label!r}."
            )


def _require_non_empty_filtered_split(dataset: Any, *, split: str) -> None:
    if len(dataset) == 0:
        raise ValueError(f"Dataset split {split!r} has no rows within the encoded token budget.")


def _verify_training_source_unchanged(
    model: Any,
    model_name: str,
    model_revision: str | None,
) -> dict[str, Any]:
    """Fail if the exact local tree loaded at initialization has since drifted."""

    expected = getattr(model, "_fastplms_training_source_identity", None)
    if not isinstance(expected, Mapping):
        raise RuntimeError(
            "The model is missing its initialization-time source identity; initialize it "
            "with initialize_model() before saving a reproducible training artifact."
        )
    observed = _immutable_source_identity(
        model_name,
        model_revision,
        source_kind="model",
    )
    if dict(expected) != observed:
        raise RuntimeError(
            "The training model source changed after initialization. Refusing to save or "
            f"record a stale artifact identity: expected={dict(expected)}, observed={observed}."
        )
    return observed


def _effective_attention_backend(model: Any) -> str | None:
    config = model.config
    for field in ("attn_backend", "attention_backend", "_attn_implementation"):
        value = getattr(config, field, None)
        if value is not None:
            return str(getattr(value, "value", value))
    return None


# Get the model ready, with or without LoRA
def initialize_model(
    model_name: str,
    num_labels: int,
    use_lora: bool = True,
    lora_config: Any = None,
    model_revision: str | None = None,
    attn_backend: str = "sdpa",
) -> tuple[Any, Any]:
    """
    Initialize a model with optional LoRA support

    Args:
        model_name: Name or path of the pretrained model
        num_labels: Number of labels for the task (1 for regression)
        use_lora: Whether to use LoRA for fine-tuning
        lora_config: Custom LoRA configuration (optional)
        model_revision: Immutable Hub commit for a remote model
        attn_backend: Explicit eager, SDPA, or Flex implementation

    Returns:
        model: The initialized model
        tokenizer: The model's tokenizer
    """
    if attn_backend not in EXAMPLE_ATTENTION_BACKENDS:
        raise ValueError(
            f"The fine-tuning example supports {EXAMPLE_ATTENTION_BACKENDS}, got "
            f"{attn_backend!r}. FlashAttention training requires an explicit BF16 CUDA "
            "loading and placement policy that this compact CLI does not expose."
        )

    print(f"Loading model {model_name} with {num_labels} labels...")

    source_identity = _immutable_source_identity(
        model_name,
        model_revision,
        source_kind="model",
    )
    revision_kwargs = (
        {"revision": source_identity["revision"]} if source_identity["kind"] == "hub" else {}
    )

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
        num_labels=num_labels,
        attn_implementation=attn_backend,
        **revision_kwargs,
    )
    effective_backend = _effective_attention_backend(model)
    if effective_backend != attn_backend:
        raise RuntimeError(
            f"Requested attention backend {attn_backend!r}, but the loaded model "
            f"configured {effective_backend!r}. Refusing to train with ambiguous dispatch."
        )
    tokenizer = model.tokenizer

    # Apply LoRA if requested
    if use_lora:
        # Default LoRA configuration if none provided
        if lora_config is None:
            # Target modules for the ESM2 sequence-classification artifacts.
            target_modules = ["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense"]

            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.01,
                bias="none",
                target_modules=target_modules,
                modules_to_save=[CLASSIFIER_MODULE_NAME],
            )

        # A custom configuration must preserve the task head too. Marking the
        # head trainable without modules_to_save would omit it from an adapter
        # checkpoint and restore an untrained head after reload.
        lora_config = _ensure_classifier_persistence(lora_config)

        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)

        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {non_trainable_params}")
        print(
            f"Percentage of parameters being trained: {100 * trainable_params / total_params:.2f}%"
        )

    model._fastplms_training_source_identity = source_identity
    return model, tokenizer


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=str)
    if value is None or isinstance(value, (bool, float, int, str)):
        return value
    return str(value)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    vocab = get_vocab() if callable(get_vocab) else getattr(tokenizer, "vocab", None)
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    return {
        "class": type(tokenizer).__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "revision": (
            init_kwargs.get("revision") or getattr(tokenizer, "_commit_hash", None)
            if isinstance(init_kwargs, dict)
            else getattr(tokenizer, "_commit_hash", None)
        ),
        "vocab_size": len(vocab) if isinstance(vocab, dict) else None,
        "vocab_sha256": _sha256_json(vocab) if isinstance(vocab, dict) else None,
        "special_token_ids": {
            name: getattr(tokenizer, f"{name}_token_id", None)
            for name in ("bos", "cls", "eos", "mask", "pad", "sep", "unk")
        },
    }


def _ordered_rows_sha256(dataset: Any, columns: tuple[str, ...]) -> str:
    """Hash ordered post-filter rows and only the columns consumed by training."""

    digest = hashlib.sha256()
    digest.update(_sha256_json(list(columns)).encode("ascii"))
    row_count = 0
    for row_count, row in enumerate(dataset, start=1):
        if not isinstance(row, Mapping):
            raise TypeError("Dataset iteration must yield row mappings.")
        missing = [column for column in columns if column not in row]
        if missing:
            raise KeyError(f"Dataset row is missing required columns: {missing}")
        payload = json.dumps(
            _json_safe({column: row[column] for column in columns}),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(row_count.to_bytes(8, "big"))
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    if row_count != len(dataset):
        raise RuntimeError(
            "Dataset length changed while hashing ordered training rows: "
            f"expected {len(dataset)}, observed {row_count}."
        )
    return digest.hexdigest()


def _dataset_identity(
    dataset: Any,
    *,
    columns: tuple[str, ...],
    source: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    fingerprint = getattr(dataset, "_fingerprint", None)
    info = getattr(dataset, "info", None)
    return {
        "source": _json_safe(source),
        "split": split,
        "columns": list(columns),
        "ordered_rows_sha256": _ordered_rows_sha256(dataset, columns),
        "library_fingerprint_advisory": fingerprint,
        "rows": len(dataset),
        "builder_name": getattr(info, "builder_name", None),
        "config_name": getattr(info, "config_name", None),
        "version": (
            str(info.version)
            if info is not None and getattr(info, "version", None) is not None
            else None
        ),
    }


def _write_training_manifest(
    output_dir: str,
    *,
    task: str,
    model: Any,
    tokenizer: Any,
    model_name: str,
    model_revision: str | None,
    seed: int,
    max_length: int,
    use_lora: bool,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_epochs: float,
    full_determinism: bool,
    datasets: dict[str, Any],
    dataset_contracts: dict[str, dict[str, Any]],
    training_arguments: TrainingArguments,
    patience: int,
    final_artifact: Mapping[str, Any],
    requested_attention_backend: str = "sdpa",
) -> None:
    """Persist the execution identity needed to reproduce a training run."""

    config = model.config
    model_configuration = _json_safe(config.to_dict())
    parameter = next(iter(model.parameters()), None)
    attention_backend = _effective_attention_backend(model)
    adapter_configuration = {
        str(name): _json_safe(adapter_config.to_dict())
        for name, adapter_config in (getattr(model, "peft_config", None) or {}).items()
    }
    if training_arguments.bf16:
        compute_dtype = "torch.bfloat16"
    elif training_arguments.fp16:
        compute_dtype = "torch.float16"
    else:
        compute_dtype = "torch.float32"
    manifest = {
        "schema_version": 1,
        "task": task,
        "command": list(sys.argv),
        "model": {
            "requested": model_name,
            "requested_source": _verify_training_source_unchanged(
                model,
                model_name,
                model_revision,
            ),
            "resolved": getattr(config, "_name_or_path", None),
            "revision": getattr(config, "_commit_hash", None),
            "weights_revision": getattr(config, "fastplms_weights_revision", None),
            "runtime_revision": getattr(config, "fastplms_runtime_revision", None),
            "attention_backend": attention_backend,
            "requested_attention_backend": requested_attention_backend,
            "effective_attention_backend": attention_backend,
            "parameter_dtype": str(parameter.dtype) if parameter is not None else None,
            "configuration": model_configuration,
            "configuration_sha256": _sha256_json(model_configuration),
            "adapters": adapter_configuration,
            "adapter_configuration_sha256": _sha256_json(adapter_configuration),
        },
        "tokenizer": _tokenizer_identity(tokenizer),
        "training": {
            "seed": seed,
            "max_length": max_length,
            "max_length_semantics": (
                "encoded token budget including tokenizer-added special and pair tokens"
            ),
            "use_lora": use_lora,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_single_process_batch_size": (batch_size * gradient_accumulation_steps),
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "full_determinism": full_determinism,
            "device": str(training_arguments.device),
            "compute_dtype": compute_dtype,
            "optimizer": getattr(
                training_arguments.optim,
                "value",
                str(training_arguments.optim),
            ),
            "scheduler": getattr(
                training_arguments.lr_scheduler_type,
                "value",
                str(training_arguments.lr_scheduler_type),
            ),
            "warmup_steps": training_arguments.warmup_steps,
            "weight_decay": training_arguments.weight_decay,
            "early_stopping_patience": patience,
            "evaluation_strategy": str(training_arguments.eval_strategy),
            "eval_steps": training_arguments.eval_steps,
            "save_strategy": str(training_arguments.save_strategy),
            "save_steps": training_arguments.save_steps,
            "logging_strategy": str(training_arguments.logging_strategy),
            "logging_steps": training_arguments.logging_steps,
            "load_best_model_at_end": training_arguments.load_best_model_at_end,
            "metric_for_best_model": training_arguments.metric_for_best_model,
            "greater_is_better": training_arguments.greater_is_better,
            "label_names": list(training_arguments.label_names or ()),
            "report_to": list(training_arguments.report_to or ()),
            "filtering": "tokenizer encoding including added special tokens",
            "truncation": "longest_first",
        },
        "datasets": {
            split_name: _dataset_identity(
                dataset,
                columns=tuple(dataset_contracts[split_name]["columns"]),
                source=dataset_contracts[split_name]["source"],
                split=str(dataset_contracts[split_name]["split"]),
            )
            for split_name, dataset in datasets.items()
        },
        "final_artifact": _json_safe(final_artifact),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "datasets": _package_version("datasets"),
            "fastplms": _package_version("fastplms"),
            "accelerate": _package_version("accelerate"),
            "cuda": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(training_arguments.device)
                if training_arguments.device.type == "cuda"
                else None
            ),
            "cuda_capability": (
                list(torch.cuda.get_device_capability(training_arguments.device))
                if training_arguments.device.type == "cuda"
                else None
            ),
            "cuda_driver": (
                getattr(torch._C, "_cuda_getDriverVersion", lambda: None)()
                if training_arguments.device.type == "cuda"
                else None
            ),
        },
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _persisted_parameter_hashes(model: Any, *, use_lora: bool) -> dict[str, str]:
    if not use_lora:
        hashes = {
            name: _tensor_sha256(value)
            for name, value in model.state_dict().items()
            if isinstance(value, torch.Tensor)
        }
    else:
        hashes = {
            name: _tensor_sha256(parameter)
            for name, parameter in model.named_parameters()
            if "lora_" in name or "modules_to_save" in name
        }
    if not hashes:
        raise RuntimeError("No persisted model state was found for final-artifact verification.")
    return hashes


def _primary_prediction_tensor(predictions: Any) -> torch.Tensor:
    """Normalize Trainer and model prediction containers to a CPU tensor."""

    value = predictions[0] if isinstance(predictions, tuple) else predictions
    if not isinstance(value, (np.ndarray, torch.Tensor)):
        raise TypeError(
            "Held-out verification expected logits as a NumPy array or Torch tensor, "
            f"got {type(value).__name__}."
        )
    return torch.as_tensor(value).detach().cpu()


def _held_out_reload_verification(
    trainer: Trainer,
    reloaded_model: Any,
    *,
    verification_dataset: Any,
    data_collator: Any,
) -> dict[str, Any]:
    """Compare prepared-Trainer and independently reloaded held-out logits."""

    row_count = len(verification_dataset)
    if row_count < 1:
        raise ValueError("Final-artifact verification requires a non-empty held-out dataset.")
    held_out_rows = [verification_dataset[index] for index in range(min(2, row_count))]
    prediction_output = trainer.predict(held_out_rows)  # type: ignore[arg-type]
    prediction_values = getattr(prediction_output, "predictions", None)
    if prediction_values is None:
        prediction_values = prediction_output[0]
    expected = _primary_prediction_tensor(prediction_values)

    batch = data_collator(held_out_rows)
    if not isinstance(batch, Mapping):
        raise TypeError("The verification data collator must return a mapping.")
    device = torch.device(getattr(trainer.args, "device", "cpu"))
    prepared_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    reloaded_model = reloaded_model.to(device).eval()
    use_bf16 = bool(getattr(trainer.args, "bf16", False))
    use_fp16 = bool(getattr(trainer.args, "fp16", False))
    autocast_dtype: torch.dtype | None = None
    if use_bf16 and device.type in {"cpu", "cuda"}:
        autocast_dtype = torch.bfloat16
    elif use_fp16 and device.type == "cuda":
        autocast_dtype = torch.float16
    with torch.inference_mode():
        if autocast_dtype is None:
            output = reloaded_model(**prepared_batch)
        else:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                output = reloaded_model(**prepared_batch)
    observed = _primary_prediction_tensor(output.logits if hasattr(output, "logits") else output[0])
    if observed.shape != expected.shape:
        raise RuntimeError(
            "Final model reload changed held-out prediction shape: "
            f"expected {tuple(expected.shape)}, observed {tuple(observed.shape)}."
        )
    if use_bf16:
        rtol, atol = 5e-3, 5e-3
    elif use_fp16:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-6
    expected_float = expected.float()
    observed_float = observed.float()
    try:
        torch.testing.assert_close(
            observed_float,
            expected_float,
            rtol=rtol,
            atol=atol,
        )
    except AssertionError as error:
        raise RuntimeError(
            "Final model reload changed held-out logits beyond the configured "
            f"dtype tolerance (rtol={rtol}, atol={atol})."
        ) from error
    absolute_error = (observed_float - expected_float).abs()
    relative_error = absolute_error / expected_float.abs().clamp_min(atol)
    return {
        "rows": len(held_out_rows),
        "shape": list(expected.shape),
        "device": str(device),
        "autocast_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": float(absolute_error.max().item()),
        "max_relative_error": float(relative_error.max().item()),
    }


def _reload_final_model(
    artifact_dir: Path,
    *,
    model_name: str,
    model_revision: str | None,
    num_labels: int,
    use_lora: bool,
    attn_backend: str = "sdpa",
) -> Any:
    source = _immutable_source_identity(
        model_name,
        model_revision,
        source_kind="model",
    )
    revision_kwargs = {"revision": source["revision"]} if source["kind"] == "hub" else {}
    if use_lora:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=num_labels,
            attn_implementation=attn_backend,
            **revision_kwargs,
        )
        return PeftModel.from_pretrained(
            base_model,
            artifact_dir,
            local_files_only=True,
        )
    return AutoModelForSequenceClassification.from_pretrained(
        artifact_dir,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=attn_backend,
    )


def _save_reload_verify_final_artifact(
    trainer: Trainer,
    tokenizer: Any,
    *,
    output_dir: str,
    model_name: str,
    model_revision: str | None,
    num_labels: int,
    use_lora: bool,
    verification_dataset: Any,
    data_collator: Any,
    attn_backend: str = "sdpa",
) -> dict[str, Any]:
    """Atomically save, locally reload, and verify trained adapter/head state."""

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / "final_model"
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing final artifact: {final_dir}")
    base_source = _verify_training_source_unchanged(
        trainer.model,
        model_name,
        model_revision,
    )
    expected_hashes = _persisted_parameter_hashes(trainer.model, use_lora=use_lora)
    staging = Path(tempfile.mkdtemp(prefix=".final-model-", dir=output_root))
    try:
        trainer.save_model(os.fspath(staging))
        save_tokenizer = getattr(tokenizer, "save_pretrained", None)
        if not callable(save_tokenizer):
            raise TypeError("The training tokenizer must implement save_pretrained().")
        save_tokenizer(staging)
        if use_lora and not (staging / "adapter_config.json").is_file():
            raise RuntimeError("Trainer did not save the required PEFT adapter configuration.")
        if not any(staging.glob("*.safetensors")):
            raise RuntimeError("Final artifact does not contain safetensors weights.")
        reloaded = _reload_final_model(
            staging,
            model_name=model_name,
            model_revision=model_revision,
            num_labels=num_labels,
            use_lora=use_lora,
            attn_backend=attn_backend,
        ).eval()
        observed_hashes = _persisted_parameter_hashes(reloaded, use_lora=use_lora)
        if observed_hashes != expected_hashes:
            missing = sorted(set(expected_hashes).difference(observed_hashes))
            unexpected = sorted(set(observed_hashes).difference(expected_hashes))
            changed = sorted(
                name
                for name in set(expected_hashes).intersection(observed_hashes)
                if expected_hashes[name] != observed_hashes[name]
            )
            raise RuntimeError(
                "Final model reload changed persisted training state: "
                f"missing={missing}, unexpected={unexpected}, changed={changed}."
            )
        inference_verification = _held_out_reload_verification(
            trainer,
            reloaded,
            verification_dataset=verification_dataset,
            data_collator=data_collator,
        )
        metadata_payload = {
            "schema_version": 1,
            "base_source": base_source,
            "use_lora": use_lora,
            "verified_parameter_sha256": expected_hashes,
            "held_out_inference": inference_verification,
        }
        (staging / "artifact_metadata.json").write_text(
            json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, final_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return {
        "path": str(final_dir),
        "tree_sha256": _tree_sha256(final_dir),
        "verified_parameter_sha256": expected_hashes,
        "reload_verified": True,
        "held_out_inference": inference_verification,
    }


# For computing performance metrics, it's fairly straightforward to add more metrics here
def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks for ties without requiring the reporting extra."""

    flattened = np.asarray(values).reshape(-1)
    order = np.argsort(flattened, kind="mergesort")
    sorted_values = flattened[order]
    ranks = np.empty(flattened.size, dtype=np.float64)
    start = 0
    while start < flattened.size:
        stop = start + 1
        while stop < flattened.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def _spearman_correlation(predictions: np.ndarray, labels: np.ndarray) -> float:
    prediction_ranks = _rankdata(predictions)
    label_ranks = _rankdata(labels)
    if prediction_ranks.size < 2:
        return float("nan")
    correlation = np.corrcoef(prediction_ranks, label_ranks)[0, 1]
    return float(correlation)


def compute_metrics_regression(p: EvalPrediction) -> dict[str, float]:
    """Compute Spearman correlation for regression tasks."""
    predictions, labels = p.predictions, p.label_ids
    predictions = predictions[0] if isinstance(predictions, tuple) else predictions
    return {
        "spearman_correlation": _spearman_correlation(
            predictions,
            cast(np.ndarray, labels),
        )
    }


def compute_metrics_classification(p: EvalPrediction) -> dict[str, float]:
    """Compute accuracy for classification tasks"""
    predictions, labels = p.predictions, p.label_ids
    predictions = predictions[0] if isinstance(predictions, tuple) else predictions
    predictions = np.argmax(predictions, axis=-1)

    accuracy = (predictions.flatten() == labels.flatten()).mean()

    return {"accuracy": accuracy}


# For plotting the results, it's fairly straightforward to add more plots here
def _save_figure_exclusive(figure: Any, output_path: str | Path) -> Path:
    """Save one PNG without replacing an existing report artifact."""

    destination = Path(output_path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"Plot output directory does not exist: {destination.parent}")
    try:
        handle = destination.open("xb")
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to overwrite an existing result plot: {destination}"
        ) from error
    try:
        with handle:
            figure.savefig(handle, format="png", dpi=300)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def plot_regression_results(
    preds: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    task_name: str = "Regression",
) -> float:
    """
    Plot regression results with Spearman correlation

    Args:
        preds: Predicted values
        labels: True values
        output_path: New PNG path inside this task's reserved output directory
        task_name: Name of the task for the plot title

    Returns:
        correlation: Spearman correlation coefficient
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr

    # Calculate Spearman correlation
    correlation, p_value = spearmanr(preds, labels)

    # Create scatter plot
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=labels, y=preds, alpha=0.6, ax=axis)

    # Add regression line
    sns.regplot(x=labels, y=preds, scatter=False, color="red", ax=axis)

    axis.set_title(f"{task_name} - Spearman Correlation: {correlation:.3f} (p={p_value:.3e})")
    axis.set_xlabel("True Values")
    axis.set_ylabel("Predicted Values")

    # Add correlation text
    axis.annotate(
        f"rho = {correlation:.3f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    figure.tight_layout()
    try:
        _save_figure_exclusive(figure, output_path)
    finally:
        plt.close(figure)
    return correlation


def plot_classification_results(
    trainer: Trainer,
    test_dataset: Any,
    output_path: str | Path,
    task_name: str = "Classification",
) -> float:
    """
    Plot classification results with confusion matrix

    Args:
        trainer: The trained model trainer
        test_dataset: Dataset to evaluate on
        output_path: New PNG path inside this task's reserved output directory
        task_name: Name of the task for the plot title

    Returns:
        accuracy: Classification accuracy
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    # Get predictions
    predictions, labels, _ = trainer.predict(test_dataset)
    preds = predictions[0] if isinstance(predictions, tuple) else predictions
    pred_values = np.argmax(preds, axis=1)

    # Calculate accuracy
    accuracy = (pred_values == labels).mean()

    # Create confusion matrix
    cm = confusion_matrix(labels, pred_values)

    # Plot confusion matrix
    figure, axis = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=axis)

    axis.set_title(f"{task_name} - Accuracy: {accuracy:.3f}")
    figure.tight_layout()
    try:
        _save_figure_exclusive(figure, output_path)
    finally:
        plt.close(figure)

    return accuracy


# Training functions
@_guard_training_output(
    lora_default="./results_regression_lora",
    full_default="./results_regression",
)
def train_regression_model(
    model_name: str = DEFAULT_MODEL,
    model_revision: str | None = None,
    train_dataset_source: str = DEFAULT_REGRESSION_TRAIN_DATASET,
    train_dataset_revision: str | None = None,
    validation_dataset_source: str = DEFAULT_REGRESSION_VALIDATION_DATASET,
    validation_dataset_revision: str | None = None,
    test_dataset_source: str = DEFAULT_REGRESSION_TEST_DATASET,
    test_dataset_revision: str | None = None,
    use_lora: bool = True,
    custom_lora_config: Any = None,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    num_epochs: int = 10,
    max_length: int = 1024,
    gradient_accumulation_steps: int = 1,
    patience: int = 3,
    seed: int = 42,
    full_determinism: bool = False,
    plot_results: bool = False,
    attn_backend: str = "sdpa",
    output_dir: str | Path | None = None,
) -> tuple[Trainer, Any]:
    """
    Train a regression model for protein-protein affinity prediction

    Args:
        model_name: Name or path of the pretrained model
        use_lora: Whether to use LoRA for fine-tuning
        custom_lora_config: Custom LoRA configuration (optional)
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        num_epochs: Number of epochs for training
        max_length: Encoded token budget for the complete protein pair,
            including tokenizer-added separator and special tokens
        gradient_accumulation_steps: Number of gradient accumulation steps
        patience: Number of evaluation calls without improvement before
            training stops
        seed: Shared model, data-loader, and training seed
        full_determinism: Request Transformers deterministic algorithms
        plot_results: Generate a reporting-extra scatter plot after training
        attn_backend: Explicit eager, SDPA, or Flex implementation
        output_dir: New task-specific output directory; existing paths are rejected

    Returns:
        trainer: The trained model trainer
        test_dataset: The test dataset used for evaluation
    """
    print("Loading datasets for regression task...")
    if max_length <= 0:
        raise ValueError("max_length must be a positive encoded token budget.")
    set_seed(seed)

    # Validate every source contract before allocating or initializing a model.
    train_data, train_source = _load_dataset_immutable(
        train_dataset_source,
        train_dataset_revision,
        split="train",
    )
    valid_data, validation_source = _load_dataset_immutable(
        validation_dataset_source,
        validation_dataset_revision,
        split="train",
    )
    test_data, test_source = _load_dataset_immutable(
        test_dataset_source,
        test_dataset_revision,
        split="train",
    )
    _validate_regression_dataset(train_data, split="train")
    _validate_regression_dataset(valid_data, split="validation")
    _validate_regression_dataset(test_data, split="test")

    # Resolve the tokenizer only after source validation. Raw residue counts
    # omit the special separator/EOS tokens inserted for a protein pair.
    model, tokenizer = initialize_model(
        model_name=model_name,
        model_revision=model_revision,
        num_labels=1,
        use_lora=use_lora,
        lora_config=custom_lora_config,
        attn_backend=attn_backend,
    )

    def _filter_pair_by_length(example: Any) -> bool:
        return _fits_token_budget(
            tokenizer,
            example["SeqA"],
            example["SeqB"],
            max_length,
        )

    train_data = train_data.filter(_filter_pair_by_length)
    valid_data = valid_data.filter(_filter_pair_by_length)
    test_data = test_data.filter(_filter_pair_by_length)
    _require_non_empty_filtered_split(train_data, split="train")
    _require_non_empty_filtered_split(valid_data, split="validation")
    _require_non_empty_filtered_split(test_data, split="test")
    dataset_contracts = {
        "train": {
            "source": train_source,
            "split": "train",
            "columns": ("SeqA", "SeqB", "labels"),
        },
        "validation": {
            "source": validation_source,
            "split": "train",
            "columns": ("SeqA", "SeqB", "labels"),
        },
        "test": {
            "source": test_source,
            "split": "train",
            "columns": ("SeqA", "SeqB", "labels"),
        },
    }

    # Create datasets
    train_dataset = PairDatasetHF(train_data, "SeqA", "SeqB", "labels", max_length=max_length)
    valid_dataset = PairDatasetHF(valid_data, "SeqA", "SeqB", "labels", max_length=max_length)
    test_dataset = PairDatasetHF(test_data, "SeqA", "SeqB", "labels", max_length=max_length)

    # Create data collator
    data_collator = PairCollator(tokenizer, regression=True, max_length=max_length)

    # Define training arguments
    if output_dir is None:
        raise RuntimeError("The output reservation guard did not supply a directory.")
    output_dir = str(Path(output_dir))
    logging_dir = str(Path(output_dir) / "logs")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir=logging_dir,
        learning_rate=learning_rate,
        seed=seed,
        data_seed=seed,
        full_determinism=full_determinism,
        **BASE_TRAINER_KWARGS,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_regression,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    metrics = trainer.evaluate(test_dataset)
    print(f"Initial metrics: {metrics}")
    print("Training regression model...")
    trainer.train()

    final_artifact = _save_reload_verify_final_artifact(
        trainer,
        tokenizer,
        output_dir=output_dir,
        model_name=model_name,
        model_revision=model_revision,
        num_labels=1,
        use_lora=use_lora,
        verification_dataset=test_dataset,
        data_collator=data_collator,
        attn_backend=attn_backend,
    )
    _write_training_manifest(
        output_dir,
        task="protein_pair_regression",
        model=trainer.model,
        tokenizer=tokenizer,
        model_name=model_name,
        model_revision=model_revision,
        seed=seed,
        max_length=max_length,
        use_lora=use_lora,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        full_determinism=full_determinism,
        datasets={"train": train_data, "validation": valid_data, "test": test_data},
        dataset_contracts=dataset_contracts,
        training_arguments=training_args,
        patience=patience,
        final_artifact=final_artifact,
        requested_attention_backend=attn_backend,
    )

    # Evaluate and visualize results
    print("Evaluating and visualizing results...")
    predictions, labels, _prediction_metrics = trainer.predict(test_dataset)
    preds = predictions[0] if isinstance(predictions, tuple) else predictions
    label_values = cast(np.ndarray, labels)
    if plot_results:
        correlation = plot_regression_results(
            preds.flatten(),
            label_values.flatten(),
            Path(output_dir) / "regression_results.png",
            "Protein-Protein Affinity",
        )
    else:
        correlation = _spearman_correlation(preds, label_values)
    print(f"Final Spearman correlation on test set: {correlation:.3f}")
    return trainer, test_dataset


@_guard_training_output(
    lora_default="./results_classification_lora",
    full_default="./results_classification",
)
def train_classification_model(
    model_name: str = DEFAULT_MODEL,
    model_revision: str | None = None,
    dataset_source: str = DEFAULT_CLASSIFICATION_DATASET,
    dataset_revision: str | None = None,
    use_lora: bool = True,
    custom_lora_config: Any = None,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    num_epochs: int = 10,
    max_length: int = 512,
    gradient_accumulation_steps: int = 1,
    patience: int = 3,
    seed: int = 42,
    full_determinism: bool = False,
    plot_results: bool = False,
    attn_backend: str = "sdpa",
    output_dir: str | Path | None = None,
) -> Trainer:
    """
    Train a classification model for protein solubility prediction

    Args:
        model_name: Name or path of the pretrained model
        use_lora: Whether to use LoRA for fine-tuning
        custom_lora_config: Custom LoRA configuration (optional)
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        num_epochs: Number of epochs for training
        max_length: Encoded token budget including tokenizer-added special tokens
        gradient_accumulation_steps: Number of gradient accumulation steps
        patience: Number of evaluation calls without improvement before
            training stops
        seed: Shared model, data-loader, and training seed
        full_determinism: Request Transformers deterministic algorithms
        plot_results: Generate a reporting-extra confusion matrix after training
        attn_backend: Explicit eager, SDPA, or Flex implementation
        output_dir: New task-specific output directory; existing paths are rejected

    Returns:
        trainer: The trained model trainer
    """
    print("Loading datasets for classification task...")
    if max_length <= 0:
        raise ValueError("max_length must be a positive encoded token budget.")
    set_seed(seed)

    data, dataset_source_identity = _load_dataset_immutable(
        dataset_source,
        dataset_revision,
    )
    num_labels = _validate_classification_dataset_dict(data)
    model, tokenizer = initialize_model(
        model_name=model_name,
        model_revision=model_revision,
        num_labels=num_labels,
        use_lora=use_lora,
        lora_config=custom_lora_config,
        attn_backend=attn_backend,
    )

    def _filter_by_length(example: Any) -> bool:
        return _fits_token_budget(tokenizer, example["seqs"], None, max_length)

    # Load datasets
    train_data = data["train"].filter(_filter_by_length)
    valid_data = data["valid"].filter(_filter_by_length)
    test_data = data["test"].filter(_filter_by_length)
    _require_non_empty_filtered_split(train_data, split="train")
    _require_non_empty_filtered_split(valid_data, split="valid")
    _require_non_empty_filtered_split(test_data, split="test")
    filtered_num_labels = _validate_classification_dataset_dict(
        {"train": train_data, "valid": valid_data, "test": test_data}
    )
    if filtered_num_labels != num_labels:
        raise ValueError(
            "Encoded-token filtering removed every training row for at least one "
            "declared class; increase max_length or repair the split before training."
        )
    dataset_contracts = {
        split_name: {
            "source": dataset_source_identity,
            "split": source_split,
            "columns": ("seqs", "labels"),
        }
        for split_name, source_split in (
            ("train", "train"),
            ("validation", "valid"),
            ("test", "test"),
        )
    }

    # Create datasets
    train_dataset = SequenceDatasetHF(train_data, "seqs", "labels", max_length=max_length)
    valid_dataset = SequenceDatasetHF(valid_data, "seqs", "labels", max_length=max_length)
    test_dataset = SequenceDatasetHF(test_data, "seqs", "labels", max_length=max_length)

    # Create data collator
    data_collator = SequenceCollator(tokenizer, regression=False, max_length=max_length)

    # Define training arguments
    if output_dir is None:
        raise RuntimeError("The output reservation guard did not supply a directory.")
    output_dir = str(Path(output_dir))
    logging_dir = str(Path(output_dir) / "logs")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir=logging_dir,
        learning_rate=learning_rate,
        seed=seed,
        data_seed=seed,
        full_determinism=full_determinism,
        **BASE_TRAINER_KWARGS,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_classification,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    metrics = trainer.evaluate(test_dataset)
    print(f"Initial metrics: {metrics}")
    print("Training classification model...")
    trainer.train()

    final_artifact = _save_reload_verify_final_artifact(
        trainer,
        tokenizer,
        output_dir=output_dir,
        model_name=model_name,
        model_revision=model_revision,
        num_labels=num_labels,
        use_lora=use_lora,
        verification_dataset=test_dataset,
        data_collator=data_collator,
        attn_backend=attn_backend,
    )
    _write_training_manifest(
        output_dir,
        task="protein_sequence_classification",
        model=trainer.model,
        tokenizer=tokenizer,
        model_name=model_name,
        model_revision=model_revision,
        seed=seed,
        max_length=max_length,
        use_lora=use_lora,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        full_determinism=full_determinism,
        datasets={"train": train_data, "validation": valid_data, "test": test_data},
        dataset_contracts=dataset_contracts,
        training_arguments=training_args,
        patience=patience,
        final_artifact=final_artifact,
        requested_attention_backend=attn_backend,
    )

    # Evaluate and visualize results
    print("Evaluating and visualizing results...")
    if plot_results:
        accuracy = plot_classification_results(
            trainer,
            test_dataset,
            Path(output_dir) / "classification_results.png",
            "Protein Solubility",
        )
    else:
        predictions, labels, _ = trainer.predict(test_dataset)
        preds = predictions[0] if isinstance(predictions, tuple) else predictions
        label_values = cast(np.ndarray, labels)
        accuracy = float((np.argmax(preds, axis=-1).reshape(-1) == label_values.reshape(-1)).mean())
    print(f"Final accuracy on test set: {accuracy:.3f}")

    return trainer


MODEL_LIST = [
    "Synthyra/ESM2-8M",
    "Synthyra/ESM2-35M",
    "Synthyra/ESM2-150M",
    "Synthyra/ESM2-650M",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the public fine-tuning command-line interface."""

    parser = argparse.ArgumentParser(description="Train models for protein tasks")
    parser.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification", "both"],
        default="both",
        help="Task to train model for",
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL, help="Path to the model to train"
    )
    parser.add_argument(
        "--model-revision",
        help=(
            "Immutable 40-character Hub commit for the model; the shipped default model "
            "is pinned automatically, and custom remote models require this argument"
        ),
    )
    parser.add_argument(
        "--classification-dataset-source",
        default=DEFAULT_CLASSIFICATION_DATASET,
    )
    parser.add_argument(
        "--classification-dataset-revision",
        help="Immutable 40-character Hub commit for the classification dataset",
    )
    parser.add_argument(
        "--regression-train-dataset-source",
        default=DEFAULT_REGRESSION_TRAIN_DATASET,
    )
    parser.add_argument("--regression-train-dataset-revision")
    parser.add_argument(
        "--regression-validation-dataset-source",
        default=DEFAULT_REGRESSION_VALIDATION_DATASET,
    )
    parser.add_argument("--regression-validation-dataset-revision")
    parser.add_argument(
        "--regression-test-dataset-source",
        default=DEFAULT_REGRESSION_TEST_DATASET,
    )
    parser.add_argument("--regression-test-dataset-revision")
    parser.add_argument(
        "--use-lora",
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LoRA (pass --no-use-lora to fine-tune the full model)",
    )
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of epochs for training")
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "Maximum encoded token count, including tokenizer-added special and pair "
            "separator tokens"
        ),
    )
    parser.add_argument(
        "--attn-backend",
        choices=EXAMPLE_ATTENTION_BACKENDS,
        default="sdpa",
        help=(
            "Explicit eager, SDPA, or Flex implementation recorded in the run manifest; "
            "FlashAttention training needs a separate explicit BF16 CUDA loading policy"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/fine-tuning"),
        help=("Parent for isolated task runs; each selected task path must not already exist"),
    )
    parser.add_argument(
        "--grad_accum", type=int, default=1, help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help=("Number of evaluation calls without improvement before early stopping"),
    )
    parser.add_argument("--seed", type=int, default=42, help="Training and data-loader seed")
    parser.add_argument(
        "--full-determinism",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Transformers deterministic algorithms (potentially slower)",
    )
    parser.add_argument(
        "--plot-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate 300 dpi result plots using the reporting extra",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run sequence classification, protein-pair regression, or both."""

    args = build_parser().parse_args(argv)
    planned_output_dirs: list[Path] = []
    if args.task in ("regression", "both"):
        planned_output_dirs.append(
            args.output_dir / ("regression_lora" if args.use_lora else "regression")
        )
    if args.task in ("classification", "both"):
        planned_output_dirs.append(
            args.output_dir / ("classification_lora" if args.use_lora else "classification")
        )
    _ensure_output_paths_available(planned_output_dirs)

    # Print training configuration
    print("\n" + "=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"Task: {args.task}")
    print(f"Model revision: {args.model_revision}")
    print(f"Using LoRA: {args.use_lora}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Max encoded token budget: {args.max_length}")
    print(f"Attention backend: {args.attn_backend}")
    print(f"Output root: {args.output_dir}")
    print(f"Gradient Accumulation Steps: {args.grad_accum}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Seed: {args.seed}")
    print(f"Full determinism: {args.full_determinism}")
    print("=" * 50 + "\n")

    # Train regression model if required
    if args.task in ["regression", "both"]:
        print("\n" + "=" * 50)
        print("TRAINING REGRESSION MODEL")
        print("=" * 50)
        train_regression_model(
            model_name=args.model_path,
            model_revision=args.model_revision,
            train_dataset_source=args.regression_train_dataset_source,
            train_dataset_revision=args.regression_train_dataset_revision,
            validation_dataset_source=args.regression_validation_dataset_source,
            validation_dataset_revision=args.regression_validation_dataset_revision,
            test_dataset_source=args.regression_test_dataset_source,
            test_dataset_revision=args.regression_test_dataset_revision,
            use_lora=args.use_lora,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            max_length=args.max_length,
            gradient_accumulation_steps=args.grad_accum,
            patience=args.patience,
            seed=args.seed,
            full_determinism=args.full_determinism,
            plot_results=args.plot_results,
            attn_backend=args.attn_backend,
            output_dir=(args.output_dir / ("regression_lora" if args.use_lora else "regression")),
        )

    # Train classification model if required
    if args.task in ["classification", "both"]:
        print("\n" + "=" * 50)
        print("TRAINING CLASSIFICATION MODEL")
        print("=" * 50)
        train_classification_model(
            model_name=args.model_path,
            model_revision=args.model_revision,
            dataset_source=args.classification_dataset_source,
            dataset_revision=args.classification_dataset_revision,
            use_lora=args.use_lora,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            max_length=args.max_length,
            gradient_accumulation_steps=args.grad_accum,
            patience=args.patience,
            seed=args.seed,
            full_determinism=args.full_determinism,
            plot_results=args.plot_results,
            attn_backend=args.attn_backend,
            output_dir=(
                args.output_dir / ("classification_lora" if args.use_lora else "classification")
            ),
        )

    print("\nTraining completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
