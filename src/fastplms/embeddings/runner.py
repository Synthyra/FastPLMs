"""Coordinate input preparation, run identity, batch execution, and publication."""

from __future__ import annotations

import torch

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from torch import Tensor

from . import identity
from .batches import (
    BatchExecutor,
    _residue_embeddings as _residue_embeddings,
    _temporary_eval,
    select_hidden_state_embeddings as select_hidden_state_embeddings,
)
from .identity import (
    _RUN_FINGERPRINT_SCHEMA_VERSION,
    _adapter_identity_metadata,
    _attention_backend,
    _attention_kernel_metadata,
    _embedding_context,
    _execution_identity_metadata,
    _fingerprint_jsonable,
    _model_identity_metadata,
    _run_fingerprint,
    _tokenizer_metadata,
)
from .inputs import (
    _InputSpool,
    _normalize_inputs,
    _validate_untruncated_lengths,
    iter_fasta as iter_fasta,
    parse_fasta as parse_fasta,
)
from .output import EmbeddingOutput
from .pooling import Pooler
from .types import EmbeddingBatch, EmbeddingInput, EmbeddingResult


_DEFAULT_BATCH_WINDOW_MULTIPLIER = 16
_SUPPORTED_STORAGE_FORMATS = frozenset({"safetensors", "sqlite"})


def embed_dataset(
    model: Any,
    inputs: (Iterable[str | EmbeddingInput | tuple[str, str]] | Mapping[str, str] | str | Path),
    *,
    batch_size: int = 2,
    pooling: str | Sequence[str] | None = None,
    full_embeddings: bool = False,
    output: str | Path | None = None,
    format: str = "safetensors",
    resume: bool = True,
    tokenizer: Any | None = None,
    max_length: int | None = None,
    truncate: bool = True,
    dtype: torch.dtype | None = torch.float32,
    shard_size: int = 2 * 1024**3,
    model_state_fingerprint: str | None = None,
    batch_window_size: int | None = None,
    max_tokens_per_batch: int | None = None,
    hidden_state_source: str = "encoder",
    decoder_inputs: Sequence[str] | None = None,
    decoder_input_ids: Tensor | None = None,
    decoder_attention_mask: Tensor | None = None,
    _embedding_batch_fn: Callable[..., EmbeddingBatch] | None = None,
    _embedding_batch_identity: Mapping[str, Any] | None = None,
    _allowed_unsupported_pooling: Sequence[str] = (),
    **model_kwargs: Any,
) -> EmbeddingResult:
    """Embed protein sequences with stable ordering and residue-only pooling."""

    for name, value in (
        ("batch_size", batch_size),
        ("shard_size", shard_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be a positive integer.")
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    for optional_name, optional_value in (
        ("max_length", max_length),
        ("max_tokens_per_batch", max_tokens_per_batch),
        ("batch_window_size", batch_window_size),
    ):
        if optional_value is not None and (
            not isinstance(optional_value, int) or isinstance(optional_value, bool)
        ):
            raise TypeError(f"{optional_name} must be a positive integer when provided.")
        if optional_value is not None and optional_value <= 0:
            raise ValueError(f"{optional_name} must be a positive integer when provided.")
    for name, value in (
        ("full_embeddings", full_embeddings),
        ("resume", resume),
        ("truncate", truncate),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean.")
    if not isinstance(format, str):
        raise TypeError("format must be a string.")
    if output is not None and not isinstance(output, (str, Path)):
        raise TypeError("output must be a path or None.")
    if model_state_fingerprint is not None and (
        not isinstance(model_state_fingerprint, str) or not model_state_fingerprint
    ):
        raise ValueError("model_state_fingerprint must be a non-empty string when provided.")
    if hidden_state_source not in {"encoder", "decoder"}:
        raise ValueError("hidden_state_source must be 'encoder' or 'decoder'.")
    hidden_state_index = model_kwargs.get("hidden_state_index", -1)
    if not isinstance(hidden_state_index, int) or isinstance(hidden_state_index, bool):
        raise TypeError("hidden_state_index must be an integer.")
    store_all_hidden_states = model_kwargs.get("store_all_hidden_states", False)
    if not isinstance(store_all_hidden_states, bool):
        raise TypeError("store_all_hidden_states must be a boolean.")
    if decoder_input_ids is not None:
        if not isinstance(decoder_input_ids, Tensor):
            raise TypeError("decoder_input_ids must be a tensor.")
        if decoder_input_ids.is_meta:
            raise ValueError("decoder_input_ids cannot be a meta tensor.")
        if decoder_input_ids.ndim != 2 or decoder_input_ids.shape[1] == 0:
            raise ValueError("decoder_input_ids must have non-empty shape (batch, sequence).")
        if decoder_input_ids.dtype not in {torch.int32, torch.int64}:
            raise TypeError("decoder_input_ids must use torch.int32 or torch.int64.")
    if decoder_attention_mask is not None:
        if not isinstance(decoder_attention_mask, Tensor):
            raise TypeError("decoder_attention_mask must be a tensor.")
        if decoder_attention_mask.is_meta:
            raise ValueError("decoder_attention_mask cannot be a meta tensor.")
        if decoder_attention_mask.is_complex() or not bool(
            torch.isfinite(decoder_attention_mask).all()
        ):
            raise ValueError("decoder_attention_mask must contain finite binary values.")
        if not bool(((decoder_attention_mask == 0) | (decoder_attention_mask == 1)).all()):
            raise ValueError("decoder_attention_mask must contain finite binary values.")
    pooling_names = (
        (("mean",) if not full_embeddings else ())
        if pooling is None
        else ((pooling,) if isinstance(pooling, str) else tuple(pooling))
    )
    if full_embeddings and pooling is not None:
        raise ValueError("full_embeddings=True cannot be combined with pooling.")
    if not full_embeddings and not pooling_names:
        raise ValueError("pooling is required unless full_embeddings=True.")
    pooler = Pooler(pooling_names) if pooling_names else None

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if format == "pth" or (output is not None and Path(output).suffix.lower() == ".pth"):
        raise ValueError("Writing pickle-based .pth embeddings is not supported.")
    if format not in _SUPPORTED_STORAGE_FORMATS:
        raise ValueError("format must be 'safetensors' or 'sqlite'.")
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive when provided.")
    if max_tokens_per_batch is not None and max_tokens_per_batch <= 0:
        raise ValueError("max_tokens_per_batch must be positive when provided.")
    if not isinstance(dtype, (torch.dtype, type(None))):
        raise TypeError("dtype must be a torch.dtype or None.")
    if batch_window_size is not None and batch_window_size <= 0:
        raise ValueError("batch_window_size must be positive when provided.")
    if _embedding_batch_fn is not None and not callable(_embedding_batch_fn):
        raise TypeError("_embedding_batch_fn must be callable when provided.")
    if _embedding_batch_fn is not None and _embedding_batch_identity is None:
        raise ValueError(
            "_embedding_batch_identity is required with _embedding_batch_fn so persisted "
            "runs bind the family-specific embedding behavior."
        )
    if _embedding_batch_identity is not None and not isinstance(_embedding_batch_identity, Mapping):
        raise TypeError("_embedding_batch_identity must be a mapping when provided.")
    if isinstance(_allowed_unsupported_pooling, (str, bytes)) or not isinstance(
        _allowed_unsupported_pooling, Sequence
    ):
        raise TypeError("_allowed_unsupported_pooling must be a sequence of pooler names.")
    if not all(isinstance(name, str) for name in _allowed_unsupported_pooling):
        raise TypeError("_allowed_unsupported_pooling must contain only strings.")
    allowed_unsupported_pooling = frozenset(_allowed_unsupported_pooling)
    if allowed_unsupported_pooling and _embedding_batch_fn is None:
        raise ValueError(
            "_allowed_unsupported_pooling is only valid with a family-specific _embedding_batch_fn."
        )
    resolved_batch_window_size = (
        batch_size * _DEFAULT_BATCH_WINDOW_MULTIPLIER
        if batch_window_size is None
        else batch_window_size
    )
    if resolved_batch_window_size < batch_size:
        raise ValueError("batch_window_size must be at least batch_size.")
    records = _normalize_inputs(inputs, disk_backed=output is not None)
    _validate_untruncated_lengths(
        records,
        max_length=max_length,
        truncate=truncate,
    )
    pooling_names = (
        (("mean",) if not full_embeddings else ())
        if pooling is None
        else ((pooling,) if isinstance(pooling, str) else tuple(pooling))
    )
    if full_embeddings:
        if pooling is not None:
            raise ValueError("full_embeddings=True cannot be combined with pooling.")
    elif not pooling_names:
        raise ValueError("pooling is required unless full_embeddings=True.")
    store_all_hidden_states = bool(model_kwargs.get("store_all_hidden_states", False))
    if store_all_hidden_states and not full_embeddings:
        raise ValueError("store_all_hidden_states=True requires full_embeddings=True.")

    unsupported = set(getattr(model, "embedding_unsupported_pooling", ()))
    unknown_pooling_overrides = allowed_unsupported_pooling.difference(unsupported)
    if unknown_pooling_overrides:
        raise ValueError(
            "_allowed_unsupported_pooling may only override poolers declared unsupported "
            f"by the model; unknown overrides: {sorted(unknown_pooling_overrides)}."
        )
    unsupported.difference_update(allowed_unsupported_pooling)
    requested_unsupported = unsupported.intersection(pooling_names)
    if requested_unsupported:
        raise ValueError(
            f"{model.__class__.__name__} does not support pooling operations "
            f"{sorted(requested_unsupported)}."
        )

    # Constructing the pooler validates names and duplicate operations before
    # any checkpoint hashing, tokenization, or inference occurs.
    pooler = Pooler(pooling_names) if pooling_names else None
    embedding_context, normalized_decoder_inputs = _embedding_context(
        model,
        records,
        hidden_state_source=hidden_state_source,
        decoder_inputs=decoder_inputs,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        model_kwargs=model_kwargs,
    )
    if _embedding_batch_identity is not None:
        embedding_context["family_adapter"] = _fingerprint_jsonable(_embedding_batch_identity)
        if allowed_unsupported_pooling:
            embedding_context["family_adapter_pooling_override"] = sorted(
                allowed_unsupported_pooling
            )

    # A pending automatic attention request settles here, inside the caller's
    # autocast context, so the fingerprint records the backend that executes.
    attention_resolution = getattr(model, "attention_resolution", None)
    if attention_resolution is not None and attention_resolution.deferred:
        model.resolve_attn_implementation()

    tokenizer_metadata = _tokenizer_metadata(model, tokenizer)
    (
        input_fingerprint,
        run_fingerprint,
        resolved_model_state_fingerprint,
        model_state_fingerprint_source,
    ) = _run_fingerprint(
        model,
        records,
        pooling=pooling_names,
        full_embeddings=full_embeddings,
        max_length=max_length,
        truncate=truncate,
        dtype=dtype,
        model_kwargs=model_kwargs,
        tokenizer_metadata=tokenizer_metadata,
        model_state_fingerprint=model_state_fingerprint,
        persist_output=output is not None,
        embedding_context=embedding_context,
        batch_size=batch_size,
        batch_window_size=resolved_batch_window_size,
        max_tokens_per_batch=max_tokens_per_batch,
    )
    destination = EmbeddingOutput(
        records,
        output=output,
        format=format,
        resume=resume,
        shard_size=shard_size,
        run_fingerprint=run_fingerprint,
        input_fingerprint=input_fingerprint,
        model_state_fingerprint=resolved_model_state_fingerprint,
        model_state_fingerprint_source=model_state_fingerprint_source,
        pooler=pooler,
        pooling_names=pooling_names,
    )
    if destination.completed is not None:
        return destination.completed

    attention_backend = _attention_backend(model)
    executor = BatchExecutor(
        model=model,
        batch_size=batch_size,
        max_tokens_per_batch=max_tokens_per_batch,
        max_length=max_length,
        truncate=truncate,
        model_kwargs=model_kwargs,
        hidden_state_source=hidden_state_source,
        normalized_decoder_inputs=normalized_decoder_inputs,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        _embedding_batch_fn=_embedding_batch_fn,
        tokenizer=tokenizer,
        store_all_hidden_states=store_all_hidden_states,
        full_embeddings=full_embeddings,
        dtype=dtype,
        pooler=pooler,
        attention_backend=attention_backend,
        need_attentions="parti" in pooling_names,
    )
    pool_slices = destination.pool_slices
    with _temporary_eval(model), torch.inference_mode():
        for window_start in range(
            destination.start_position, len(records), resolved_batch_window_size
        ):
            window_stop = min(window_start + resolved_batch_window_size, len(records))
            window_records = records[window_start:window_stop]
            if not isinstance(window_records, Sequence):
                raise RuntimeError("The immutable embedding spool returned a non-sequence window.")
            new_records, pool_slices = executor.run_window(
                window_records, window_start=window_start
            )
            destination.append(window_start, new_records)

    software_versions = identity._software_versions()
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
        "fingerprint_schema_version": _RUN_FINGERPRINT_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "input_fingerprint": input_fingerprint,
        "model_state_fingerprint": resolved_model_state_fingerprint,
        "model_state_fingerprint_source": model_state_fingerprint_source,
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
        "tokenizer": tokenizer_metadata,
        **embedding_context,
        "pooling": list(pooling_names),
        "pool_slices": pool_slices,
        "full_embeddings": full_embeddings,
        "max_length": max_length,
        "truncate": truncate,
        "truncation": {"enabled": truncate, "max_length": max_length},
        "batching": {
            "batch_size": batch_size,
            "batch_window_size": resolved_batch_window_size,
            "max_tokens_per_batch": max_tokens_per_batch,
            "input_storage": ("disk-spool" if isinstance(records, _InputSpool) else "memory"),
            "ordering": "bounded-length-bucketed-stable-output",
            "resume_commit_granularity": (
                "not-applicable"
                if output is None
                else "batch-window"
                if format == "sqlite"
                else "shard-flush"
            ),
        },
        "residue_mask_policy": "biological-residues-only",
        "record_count": len(records),
        "descriptor_index": (
            "memory-metadata"
            if output is None
            else "sqlite-records"
            if format == "sqlite"
            else "safetensors-generation-index"
        ),
        "storage_format": format if output is not None else "memory",
        "software": software_versions,
        "execution": _execution_identity_metadata(model),
        "adapter": _adapter_identity_metadata(model),
        "torch_version": software_versions["torch"],
        "transformers_version": software_versions["transformers"],
        "complete": True,
    }
    if destination.output_descriptors is not None:
        metadata["outputs"] = destination.output_descriptors
        metadata["tensor_hashes"] = [item["sha256"] for item in destination.output_descriptors]
    status = getattr(model, "esmc_precision_status", None)
    if status is not None:
        metadata["esmc_precision"] = status.as_dict() if hasattr(status, "as_dict") else status
    return destination.finish(metadata)


class EmbeddingMixin:
    """Small delegation mixin shared by FastPLMs model classes."""

    def embed_dataset(self, inputs: Any, **kwargs: Any) -> EmbeddingResult:
        return embed_dataset(self, inputs, **kwargs)


__all__ = [
    "EmbeddingMixin",
    "embed_dataset",
    "iter_fasta",
    "parse_fasta",
    "select_hidden_state_embeddings",
]
