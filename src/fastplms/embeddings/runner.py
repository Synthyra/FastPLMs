"""Model-independent dataset embedding orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .pooling import Pooler
from .storage import (
    append_sqlite_records,
    initialize_sqlite_run,
    load_result,
    load_sqlite_result,
    safetensors_result_exists,
    save_result,
    tensor_sha256,
    update_sqlite_run_metadata,
)
from .types import (
    EmbeddingBatch,
    EmbeddingInput,
    EmbeddingRecord,
    EmbeddingResult,
    LazyTensorReference,
)

_MAX_PARTI_RESIDUES = 2_048


def _validate_parti_length(M: Tensor) -> None:
    """Reject an oversized attention graph before model inference."""

    n_residues = int(M.to(dtype=torch.int64).sum(dim=1).max().item())
    if n_residues > _MAX_PARTI_RESIDUES:
        raise ValueError(f"parti supports at most {_MAX_PARTI_RESIDUES:,} biological residues.")


def select_hidden_state_embeddings(
    last_hidden_state: Tensor,
    hidden_states: tuple[Tensor, ...] | None,
    *,
    hidden_state_index: int = -1,
    store_all_hidden_states: bool = False,
) -> Tensor:
    """Select one hidden state or stack every state without changing values."""
    if store_all_hidden_states:
        if not hidden_states:
            raise ValueError("store_all_hidden_states requires model hidden states.")
        # H has shape (b, n, l, d), where n follows the model's output order.
        return torch.stack(hidden_states, dim=1)
    if hidden_state_index == -1:
        return last_hidden_state
    if not hidden_states:
        raise ValueError("hidden_state_index requires model hidden states.")
    return hidden_states[hidden_state_index]


def parse_fasta(path: str | Path) -> list[EmbeddingInput]:
    """Parse FASTA records while preserving identifiers, order, and duplicates."""

    records: list[EmbeddingInput] = []
    identifier: str | None = None
    sequence_parts: list[str] = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if identifier is not None:
                records.append(EmbeddingInput(identifier, "".join(sequence_parts)))
            identifier = line[1:].strip().split(maxsplit=1)[0]
            if not identifier:
                raise ValueError(f"Missing FASTA identifier on line {line_number}.")
            sequence_parts = []
        else:
            if identifier is None:
                raise ValueError(
                    f"Sequence data precedes the first FASTA header on line {line_number}."
                )
            sequence_parts.append("".join(line.split()))
    if identifier is not None:
        records.append(EmbeddingInput(identifier, "".join(sequence_parts)))
    if not records:
        raise ValueError(f"No FASTA records found in {path}.")
    return records


def _normalize_inputs(
    inputs: Iterable[str | EmbeddingInput | tuple[str, str]] | str | Path,
) -> list[EmbeddingInput]:
    is_fasta_path = isinstance(inputs, Path)
    if isinstance(inputs, str):
        try:
            is_fasta_path = Path(inputs).is_file()
        except OSError:
            is_fasta_path = False
    if is_fasta_path:
        return parse_fasta(inputs)
    if isinstance(inputs, str):
        values: Iterable[str | EmbeddingInput | tuple[str, str]] = [inputs]
    else:
        values = inputs
    records: list[EmbeddingInput] = []
    for position, item in enumerate(values):
        if isinstance(item, EmbeddingInput):
            record = item
        elif isinstance(item, str):
            record = EmbeddingInput(str(position), item)
        elif isinstance(item, tuple) and len(item) == 2:
            record = EmbeddingInput(str(item[0]), str(item[1]))
        else:
            raise TypeError(
                "inputs must contain sequences, EmbeddingInput values, or (id, sequence) tuples."
            )
        records.append(record)
    if not records:
        raise ValueError("inputs must contain at least one sequence.")
    return records


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _attention_backend(model: Any) -> str | None:
    config = getattr(model, "config", None)
    for name in ("_attn_implementation", "attn_implementation", "attn_backend"):
        value = getattr(config, name, None)
        if value:
            return str(value)
    return None


def _attention_kernel_metadata(backend: str | None) -> dict[str, Any] | None:
    if backend not in {"flash_attention_2", "flash_attention_3"}:
        return None
    from fastplms.registry import get_model_registry

    spec = get_model_registry().attention_kernels[backend]
    return {
        "repository": spec.repository,
        "revision": spec.revision,
        "version": spec.version,
        "expected_variant": spec.expected_variant,
        "dtypes": list(spec.dtypes),
    }


def _tokenizer_metadata(model: Any, tokenizer: Any | None) -> dict[str, Any]:
    resolved = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
    if resolved is None:
        return {"mode": "native-sequence"}
    return {
        "mode": "tokenizer",
        "class": f"{resolved.__class__.__module__}.{resolved.__class__.__qualname__}",
        "name_or_path": getattr(resolved, "name_or_path", None),
        "vocab_size": getattr(resolved, "vocab_size", None),
        "special_token_ids": list(getattr(resolved, "all_special_ids", ())),
    }


def _software_versions() -> dict[str, str | None]:
    try:
        import fastplms

        fastplms_version = fastplms.__version__
    except (AttributeError, ImportError):
        fastplms_version = None
    try:
        import safetensors

        safetensors_version = safetensors.__version__
    except ImportError:
        safetensors_version = None
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    return {
        "fastplms": fastplms_version,
        "safetensors": safetensors_version,
        "torch": torch.__version__,
        "transformers": transformers_version,
    }


def _biological_residue_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    tokenizer: Any,
) -> Tensor:
    """Remove padding and tokenizer-declared special tokens from M."""

    M = attention_mask.to(dtype=torch.bool)
    special_ids = tuple(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ()))
    if special_ids:
        specials = torch.tensor(special_ids, device=input_ids.device, dtype=input_ids.dtype)
        M = M & ~torch.isin(input_ids, specials)
    return M


def _generic_embedding_batch(
    model: Any,
    sequences: list[str],
    *,
    tokenizer: Any | None,
    max_length: int | None,
    truncate: bool,
    need_attentions: bool,
    model_kwargs: dict[str, Any],
) -> EmbeddingBatch:
    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    if tokenizer is None:
        tokenizer = getattr(model, "tokenizer", None)

    if tokenizer is None and model_type == "e1":
        output = model._embed(sequences, return_attention_mask=True, **model_kwargs)
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("E1 _embed must return (X, residue_mask).")
        X, M = output
        preparer = getattr(model, "prep_tokens", None)
        if preparer is not None and hasattr(preparer, "get_batch_kwargs"):
            prepared = preparer.get_batch_kwargs(sequences, device=X.device)
            input_ids = prepared["input_ids"]
            boundary_ids = preparer.boundary_token_ids.to(
                device=input_ids.device, dtype=input_ids.dtype
            )
            # E1 wraps each raw sequence in BOS, context-label, terminal-label,
            # and EOS tokens. Only amino-acid rows are biological residues.
            M = M.to(dtype=torch.bool) & ~torch.isin(input_ids, boundary_ids)
        if need_attentions:
            raise ValueError("parti is not available for tokenizer-free E1 embedding.")
        return EmbeddingBatch(X=X, residue_mask=M.to(dtype=torch.bool))
    if tokenizer is None:
        raise ValueError("A tokenizer is required for this model's embedding path.")

    tokenize_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": truncate,
    }
    if max_length is not None:
        tokenize_kwargs["max_length"] = max_length
    sequence_tokenizer = getattr(model, "_tokenize_sequence_batch", None)
    if callable(sequence_tokenizer):
        encoded = sequence_tokenizer(sequences, tokenizer=tokenizer, **tokenize_kwargs)
    else:
        encoded = tokenizer(sequences, **tokenize_kwargs)
    device = _model_device(model)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", input_ids.new_ones(input_ids.shape)).to(device)
    M = _biological_residue_mask(input_ids, attention_mask, tokenizer)
    if need_attentions:
        # Validate l before either the backbone or its quadratic attention graph
        # is materialized. M has shape (b, l).
        _validate_parti_length(M)
    X = model._embed(input_ids, attention_mask, **model_kwargs)
    attentions = None
    if need_attentions:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        attentions = getattr(output, "attentions", None)
        if attentions is None:
            raise ValueError("The model did not return attentions required by parti.")
    return EmbeddingBatch(X=X, residue_mask=M, attentions=attentions)


def _first_metadata_value(*values: Any) -> Any:
    for value in values:
        if isinstance(value, str):
            if value.strip():
                return value
        elif value is not None:
            return value
    return None


def _model_identity_metadata(model: Any) -> dict[str, Any]:
    """Resolve model and checkpoint identity, including local artifact fallbacks."""

    config = getattr(model, "config", None)
    checkpoint_revision = _first_metadata_value(
        getattr(config, "fastplms_checkpoint_revision", None),
        getattr(config, "_commit_hash", None),
    )
    return {
        "model_id": _first_metadata_value(
            getattr(config, "fastplms_model_id", None),
            getattr(config, "_name_or_path", None),
        ),
        "model_revision": _first_metadata_value(
            getattr(config, "_commit_hash", None),
            checkpoint_revision,
        ),
        "checkpoint_repo_id": getattr(config, "fastplms_checkpoint_repo_id", None),
        "checkpoint_revision": checkpoint_revision,
        "checkpoint_hash": _first_metadata_value(
            getattr(model, "checkpoint_hash", None),
            getattr(config, "checkpoint_hash", None),
            getattr(config, "fastplms_checkpoint_hash", None),
        ),
    }


def _run_fingerprint(
    model: Any,
    records: Sequence[EmbeddingInput],
    *,
    pooling: Sequence[str],
    full_embeddings: bool,
    max_length: int | None,
    truncate: bool,
    dtype: torch.dtype | None,
    model_kwargs: dict[str, Any],
    tokenizer_metadata: dict[str, Any],
) -> tuple[str, str]:
    input_payload = [[record.id, record.sequence] for record in records]
    input_fingerprint = hashlib.sha256(
        json.dumps(input_payload, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    attention_backend = _attention_backend(model)
    model_identity = _model_identity_metadata(model)
    payload = {
        "input_fingerprint": input_fingerprint,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        **model_identity,
        "attention_backend": attention_backend,
        "attention_kernel": _attention_kernel_metadata(attention_backend),
        "layer": repr(
            getattr(model, "embedding_layer", model_kwargs.get("hidden_state_index", -1))
        ),
        "projection": getattr(model, "embedding_projection", None),
        "esmc_source": getattr(model, "_esmc_source", None),
        "esmc_revision": getattr(model, "_esmc_source_revision", None),
        "esmc_files": getattr(model, "_esmc_source_files", None),
        "token_policy": getattr(model, "embedding_token_policy", None),
        "tokenizer": tokenizer_metadata,
        "fastplms_version": _software_versions()["fastplms"],
        "pooling": list(pooling),
        "full_embeddings": full_embeddings,
        "max_length": max_length,
        "truncate": truncate,
        "dtype": str(dtype) if dtype is not None else None,
        "model_kwargs": {key: repr(value) for key, value in sorted(model_kwargs.items())},
        "residue_mask_policy": "attention-mask-minus-special-tokens",
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return input_fingerprint, run_fingerprint


def _output_exists(path: str | Path, format: str) -> bool:
    path = Path(path)
    if format == "sqlite":
        return path.is_file()
    return safetensors_result_exists(path)


def _output_descriptor(position: int, record: EmbeddingRecord) -> dict[str, Any]:
    tensor = record.tensor
    if isinstance(tensor, LazyTensorReference):
        dtype = tensor.dtype
        shape = tensor.shape
        digest = tensor.sha256
    else:
        dtype = str(tensor.dtype).removeprefix("torch.")
        shape = tuple(tensor.shape)
        digest = tensor_sha256(tensor)
    return {
        "position": position,
        "id": record.id,
        "dtype": dtype,
        "shape": shape,
        "sha256": digest,
    }


def embed_dataset(
    model: Any,
    inputs: Iterable[str | EmbeddingInput | tuple[str, str]] | str | Path,
    *,
    batch_size: int = 2,
    pooling: str | Sequence[str] | None = ("mean",),
    full_embeddings: bool = False,
    output: str | Path | None = None,
    format: str = "safetensors",
    resume: bool = True,
    tokenizer: Any | None = None,
    max_length: int | None = None,
    truncate: bool = True,
    dtype: torch.dtype | None = torch.float32,
    shard_size: int = 2 * 1024**3,
    **model_kwargs: Any,
) -> EmbeddingResult:
    """Embed protein sequences with stable ordering and residue-only pooling."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive when provided.")
    records = _normalize_inputs(inputs)
    pooling_names = (
        () if pooling is None else ((pooling,) if isinstance(pooling, str) else tuple(pooling))
    )
    if full_embeddings:
        if pooling_names not in {(), ("mean",)}:
            raise ValueError("full_embeddings=True cannot be combined with pooling.")
        pooling_names = ()
    elif not pooling_names:
        raise ValueError("pooling is required unless full_embeddings=True.")

    unsupported = set(getattr(model, "embedding_unsupported_pooling", ()))
    requested_unsupported = unsupported.intersection(pooling_names)
    if requested_unsupported:
        raise ValueError(
            f"{model.__class__.__name__} does not support pooling operations "
            f"{sorted(requested_unsupported)}."
        )

    input_fingerprint, run_fingerprint = _run_fingerprint(
        model,
        records,
        pooling=pooling_names,
        full_embeddings=full_embeddings,
        max_length=max_length,
        truncate=truncate,
        dtype=dtype,
        model_kwargs=model_kwargs,
        tokenizer_metadata=_tokenizer_metadata(model, tokenizer),
    )
    existing: EmbeddingResult | None = None
    start_position = 0
    if output is not None and resume and _output_exists(output, format):
        existing = load_result(output, format=format)
        if existing.metadata.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                "Existing embeddings were produced by a different run fingerprint; "
                "choose another output or set resume=False."
            )
        expected = [(record.id, record.sequence) for record in records]
        observed = [(record.id, record.sequence) for record in existing]
        if observed == expected and existing.metadata.get("complete", True):
            return existing
        if observed != expected[: len(observed)]:
            raise ValueError(
                "Existing embeddings are not an ordered prefix of the requested inputs."
            )
        start_position = len(observed)

    sqlite_run_id: str | None = None
    if output is not None and format == "sqlite":
        sqlite_run_id = initialize_sqlite_run(
            output,
            {
                "format_version": 1,
                "run_fingerprint": run_fingerprint,
                "input_fingerprint": input_fingerprint,
                "complete": False,
            },
            resume=resume,
        )

    pooler = Pooler(pooling_names) if pooling_names else None
    attention_backend = _attention_backend(model)
    output_records: list[EmbeddingRecord] = (
        [] if sqlite_run_id is not None else list(existing or ())
    )
    output_descriptors = [
        _output_descriptor(position, record) for position, record in enumerate(existing or ())
    ]
    pool_slices: dict[str, tuple[int, int]] = {}
    if existing and pooler is not None:
        pooled_width = existing[0].load_tensor().shape[-1]
        if pooled_width % len(pooling_names) != 0:
            raise ValueError("Stored pooled width is inconsistent with pooling metadata.")
        pool_slices = pooler.output_slices(pooled_width // len(pooling_names))
    need_attentions = "parti" in pooling_names

    with torch.inference_mode():
        for start in range(start_position, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            sequences = [
                record.sequence[:max_length]
                if truncate and max_length is not None
                else record.sequence
                for record in batch_records
            ]
            custom_batch = getattr(model, "_embedding_batch", None)
            if custom_batch is not None:
                batch = custom_batch(sequences, **model_kwargs)
                if not isinstance(batch, EmbeddingBatch):
                    raise TypeError("_embedding_batch must return EmbeddingBatch.")
            else:
                batch = _generic_embedding_batch(
                    model,
                    sequences,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    truncate=truncate,
                    need_attentions=need_attentions,
                    model_kwargs=model_kwargs,
                )
            X = batch.X
            M = batch.residue_mask.to(device=X.device, dtype=torch.bool)
            if need_attentions:
                # Custom embedding adapters own their inference path, so retain
                # the same fail-closed length contract at their boundary.
                _validate_parti_length(M)
            if X.ndim != 3 or M.shape != X.shape[:2]:
                raise ValueError(
                    "Embedding batches must provide X with shape (b, l, d) and "
                    "residue_mask with shape (b, l)."
                )
            if dtype is not None:
                X = X.to(dtype=dtype)

            if full_embeddings:
                values = [X_i[M_i].detach().cpu() for X_i, M_i in zip(X, M, strict=True)]
            else:
                assert pooler is not None
                Y = pooler(
                    X,
                    M,
                    attentions=batch.attentions,
                    attention_backend=attention_backend,
                )
                pool_slices = pooler.output_slices(X.shape[-1])
                values = list(Y.detach().cpu().unbind(0))
            new_records = [
                EmbeddingRecord(record.id, record.sequence, value)
                for record, value in zip(batch_records, values, strict=True)
            ]
            output_descriptors.extend(
                _output_descriptor(start + offset, record)
                for offset, record in enumerate(new_records)
            )
            if output is not None and sqlite_run_id is not None:
                append_sqlite_records(output, sqlite_run_id, start, new_records)
            else:
                output_records.extend(new_records)

    software_versions = _software_versions()
    projection = getattr(model, "embedding_projection", None)
    resolved_layer = getattr(
        model,
        "embedding_layer",
        model_kwargs.get("hidden_state_index", -1),
    )
    token_policy = getattr(
        model,
        "embedding_token_policy",
        {
            "unit": "residue",
            "include": ["biological residues"],
            "exclude": [
                "BOS",
                "EOS",
                "padding",
                "chain delimiters",
                "non-protein tokens",
            ],
        },
    )
    model_identity = _model_identity_metadata(model)
    metadata: dict[str, Any] = {
        "format_version": 1,
        "run_fingerprint": run_fingerprint,
        "input_fingerprint": input_fingerprint,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        **model_identity,
        "dtype": str(dtype).removeprefix("torch.") if dtype is not None else "model",
        "attention_backend": attention_backend,
        "attention_kernel": _attention_kernel_metadata(attention_backend),
        "layer": resolved_layer,
        "projection": projection,
        "esmc_source": getattr(model, "_esmc_source", None),
        "esmc_revision": getattr(model, "_esmc_source_revision", None),
        "esmc_files": getattr(model, "_esmc_source_files", None),
        "token_policy": token_policy,
        "tokenizer": _tokenizer_metadata(model, tokenizer),
        "pooling": list(pooling_names),
        "pool_slices": pool_slices,
        "full_embeddings": full_embeddings,
        "max_length": max_length,
        "truncate": truncate,
        "truncation": {"enabled": truncate, "max_length": max_length},
        "residue_mask_policy": "biological-residues-only",
        "outputs": output_descriptors,
        "tensor_hashes": [item["sha256"] for item in output_descriptors],
        "storage_format": format if output is not None else "memory",
        "software": software_versions,
        "torch_version": software_versions["torch"],
        "transformers_version": software_versions["transformers"],
        "complete": True,
    }
    status = getattr(model, "esmc_precision_status", None)
    if status is not None:
        metadata["esmc_precision"] = status.as_dict() if hasattr(status, "as_dict") else status
    if output is not None and sqlite_run_id is not None:
        update_sqlite_run_metadata(output, sqlite_run_id, metadata)
        return load_sqlite_result(output, run_id=sqlite_run_id)
    result = EmbeddingResult(output_records, metadata)
    if output is not None:
        return save_result(result, output, format=format, shard_size=shard_size)
    return result


class EmbeddingMixin:
    """Small delegation mixin shared by FastPLMs model classes."""

    def embed_dataset(self, inputs: Any, **kwargs: Any) -> EmbeddingResult:
        return embed_dataset(self, inputs, **kwargs)


__all__ = [
    "EmbeddingMixin",
    "embed_dataset",
    "parse_fasta",
    "select_hidden_state_embeddings",
]
