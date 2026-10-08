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
    TapExecutor,
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
from .pooling import POOLING_SEMANTICS, Pooler
from .taps import Tap, TapPlan, plan_taps
from .types import (
    EmbeddingBatch, EmbeddingInput, EmbeddingResult, TapRecord, TapResult, TapRunReceipt,
)


_DEFAULT_BATCH_WINDOW_MULTIPLIER = 16
_SUPPORTED_STORAGE_FORMATS = frozenset({"safetensors", "sqlite"})


def embed_dataset(
    model: Any,
    inputs: (Iterable[str | EmbeddingInput | tuple[str, str]] | Mapping[str, str] | str | Path),
    *,
    batch_size: int = 2,
    pooling: str | Sequence[str] | None = None,
    full_embeddings: bool = False,
    taps: Sequence[Tap] | None = None,
    tap_sink: Callable[[Sequence[TapRecord], Mapping[str, str]], None] | None = None,
    require_residue_identity: bool = False,
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
) -> EmbeddingResult | TapResult | TapRunReceipt:
    """Embed protein sequences with stable ordering and residue-only pooling.

    ``taps`` instead returns a ``TapResult``: each tap's output from one forward pass per
    batch, kept in memory. With ``tap_sink``, deliver bounded windows to the callback and return
    a ``TapRunReceipt`` without retaining their tensors. The callback receives ordered records
    and the run/input fingerprints; it must release the records to preserve bounded memory.
    """

    # decoder_input_ids, decoder_attention_mask: (n_records, l_decoder), aligned with the inputs
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
        ("require_residue_identity", require_residue_identity),
        ("resume", resume),
        ("truncate", truncate),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean.")
    if require_residue_identity and taps is None:
        raise ValueError("Residue identity validation requires a tap plan.")
    if tap_sink is not None and (taps is None or not callable(tap_sink)):
        raise ValueError("tap_sink requires a tap plan and a callable destination.")
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
    tap_plan = (
        _tap_plan(
            model,
            taps,
            pooling=pooling,
            full_embeddings=full_embeddings,
            output=output,
            model_kwargs=model_kwargs,
            family_adapter=(
                _embedding_batch_fn is not None or _embedding_batch_identity is not None
            ),
        )
        if taps is not None
        else None
    )
    pooling_names = _requested_pooling(
        pooling,
        full_embeddings=full_embeddings,
        taps_requested=tap_plan is not None,
    )
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
    records = _normalize_inputs(inputs, disk_backed=output is not None or tap_sink is not None)
    _validate_untruncated_lengths(
        records,
        max_length=max_length,
        truncate=truncate,
    )
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
    requested_pooling = set(pooling_names)
    if tap_plan is not None:
        requested_pooling.update(tap_plan.pooling_names)
    requested_unsupported = unsupported.intersection(requested_pooling)
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
    tap_identity = tap_plan.identity() if tap_plan is not None else None
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
        taps=tap_identity,
    )
    attention_backend = _attention_backend(model)
    if tap_plan is not None:
        tap_records, tap_pool_slices = _embed_tap_windows(
            model,
            records,
            TapExecutor(
                model=model,
                plan=tap_plan,
                batch_size=batch_size,
                max_tokens_per_batch=max_tokens_per_batch,
                max_length=max_length,
                truncate=truncate,
                tokenizer=tokenizer,
                dtype=dtype,
                attention_backend=attention_backend,
                require_residue_identity=require_residue_identity,
            ),
            window_size=resolved_batch_window_size,
            sink=tap_sink,
            run_identity={
                "run_fingerprint": run_fingerprint, "input_fingerprint": input_fingerprint,
            },
        )
        metadata = _run_metadata(
            model,
            records,
            run_fingerprint=run_fingerprint,
            input_fingerprint=input_fingerprint,
            model_state_fingerprint=resolved_model_state_fingerprint,
            model_state_fingerprint_source=model_state_fingerprint_source,
            dtype=dtype,
            attention_backend=attention_backend,
            layer=None,
            tokenizer_metadata=tokenizer_metadata,
            embedding_context=embedding_context,
            pooling_names=pooling_names,
            pool_slices={},
            full_embeddings=full_embeddings,
            max_length=max_length,
            truncate=truncate,
            batch_size=batch_size,
            batch_window_size=resolved_batch_window_size,
            max_tokens_per_batch=max_tokens_per_batch,
            output=output,
            format=format,
            descriptor_index="not-recorded",
            taps={
                "plan": tap_identity,
                "stop_after_layer": tap_plan.deepest_layer,
                "pool_slices": tap_pool_slices,
            },
        )
        if tap_sink is not None:
            metadata["storage_format"] = "tap-sink"
            return TapRunReceipt(len(records), metadata)
        return TapResult(tap_records, metadata)
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

    metadata = _run_metadata(
        model,
        records,
        run_fingerprint=run_fingerprint,
        input_fingerprint=input_fingerprint,
        model_state_fingerprint=resolved_model_state_fingerprint,
        model_state_fingerprint_source=model_state_fingerprint_source,
        dtype=dtype,
        attention_backend=attention_backend,
        layer=getattr(
            model,
            "embedding_layer",
            model_kwargs.get("hidden_state_index", -1),
        ),
        tokenizer_metadata=tokenizer_metadata,
        embedding_context=embedding_context,
        pooling_names=pooling_names,
        pool_slices=pool_slices,
        full_embeddings=full_embeddings,
        max_length=max_length,
        truncate=truncate,
        batch_size=batch_size,
        batch_window_size=resolved_batch_window_size,
        max_tokens_per_batch=max_tokens_per_batch,
        output=output,
        format=format,
        descriptor_index=(
            "memory-metadata"
            if output is None
            else "sqlite-records"
            if format == "sqlite"
            else "safetensors-generation-index"
        ),
    )
    if destination.output_descriptors is not None:
        metadata["outputs"] = destination.output_descriptors
        metadata["tensor_hashes"] = [item["sha256"] for item in destination.output_descriptors]
    return destination.finish(metadata)


def _tap_plan(
    model: Any,
    taps: Sequence[Tap],
    *,
    pooling: str | Sequence[str] | None,
    full_embeddings: bool,
    output: str | Path | None,
    model_kwargs: Mapping[str, Any],
    family_adapter: bool,
) -> TapPlan:
    """Validate a tap request against the arguments it excludes and the model's hidden states."""

    excluded = [
        name
        for name, requested in (
            ("pooling", pooling is not None),
            ("full_embeddings", full_embeddings),
            ("hidden_state_index", "hidden_state_index" in model_kwargs),
            ("store_all_hidden_states", "store_all_hidden_states" in model_kwargs),
        )
        if requested
    ]
    if excluded:
        raise ValueError(
            f"taps= cannot be combined with {', '.join(excluded)}; each tap names its own "
            "layer and pooling."
        )
    if model_kwargs:
        raise ValueError(
            f"taps= takes no model keyword arguments; received {sorted(model_kwargs)}."
        )
    if family_adapter:
        raise ValueError(
            "taps= runs the model's own one-pass path; _embedding_batch_fn and "
            "_embedding_batch_identity do not apply."
        )
    if output is not None:
        raise ValueError(
            "taps= returns its records in memory; omit output=. To persist them, call "
            "embed_into_features, which writes each tap into the feature store of its key and "
            "embeds only the sequences that store lacks."
        )
    if getattr(model, "embedding_tap_support", False) is not True:
        raise ValueError(
            f"{model.__class__.__name__} does not support taps=. One-pass taps need a model "
            "family that implements them, such as ESM++ (ESMC)."
        )
    return plan_taps(taps, int(model.embedding_tap_state_count))


def _requested_pooling(
    pooling: str | Sequence[str] | None,
    *,
    full_embeddings: bool,
    taps_requested: bool,
) -> tuple[str, ...]:
    """Pooler names of a single-output run: mean by default, none for residues or taps."""

    if taps_requested:
        return ()
    if full_embeddings:
        if pooling is not None:
            raise ValueError("full_embeddings=True cannot be combined with pooling.")
        return ()
    names = (
        ("mean",)
        if pooling is None
        else ((pooling,) if isinstance(pooling, str) else tuple(pooling))
    )
    if not names:
        raise ValueError("pooling is required unless full_embeddings=True.")
    return names


def _embed_tap_windows(
    model: Any,
    records: Sequence[EmbeddingInput],
    executor: TapExecutor,
    *,
    window_size: int,
    sink: Callable[[Sequence[TapRecord], Mapping[str, str]], None] | None = None,
    run_identity: Mapping[str, str] | None = None,
) -> tuple[list[TapRecord], dict[str, dict[str, tuple[int, int]]]]:
    """Run bounded windows in source order, retaining tensors only without a sink."""

    tap_records: list[TapRecord] = []
    pool_slices: dict[str, dict[str, tuple[int, int]]] = {}
    with _temporary_eval(model), torch.inference_mode():
        for window_start in range(0, len(records), window_size):
            window_stop = min(window_start + window_size, len(records))
            window_records = records[window_start:window_stop]
            if not isinstance(window_records, Sequence):
                raise RuntimeError("The immutable embedding spool returned a non-sequence window.")
            new_records, pool_slices = executor.run_window(
                window_records, window_start=window_start
            )
            if sink is None:
                tap_records.extend(new_records)
            else:
                sink(new_records, dict(run_identity or {}))
            # Release the previous window before allocating the next, including on CPU.
            del new_records
    return tap_records, pool_slices


def _run_metadata(
    model: Any,
    records: Sequence[EmbeddingInput],
    *,
    run_fingerprint: str,
    input_fingerprint: str,
    model_state_fingerprint: str | None,
    model_state_fingerprint_source: str,
    dtype: torch.dtype | None,
    attention_backend: str | None,
    layer: Any,
    tokenizer_metadata: dict[str, Any],
    embedding_context: Mapping[str, Any],
    pooling_names: Sequence[str],
    pool_slices: Mapping[str, tuple[int, int]],
    full_embeddings: bool,
    max_length: int | None,
    truncate: bool,
    batch_size: int,
    batch_window_size: int,
    max_tokens_per_batch: int | None,
    output: str | Path | None,
    format: str,
    descriptor_index: str,
    taps: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything a finished run records so that it can be reproduced and resumed."""

    software_versions = identity._software_versions()
    projection = getattr(model, "embedding_projection", None)
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
        "model_state_fingerprint": model_state_fingerprint,
        "model_state_fingerprint_source": model_state_fingerprint_source,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        **model_identity,
        "dtype": str(dtype).removeprefix("torch.") if dtype is not None else "model",
        "attention_backend": attention_backend,
        "attention_kernel": _attention_kernel_metadata(attention_backend),
        "layer": layer,
        "projection": projection,
        "esmc_source": getattr(model, "_esmc_source", None),
        "esmc_revision": getattr(model, "_esmc_source_revision", None),
        "esmc_files": getattr(model, "_esmc_source_files", None),
        "token_policy": token_policy,
        "tokenizer": tokenizer_metadata,
        **embedding_context,
        "pooling": list(pooling_names),
        "pooling_semantics": dict(POOLING_SEMANTICS),
        "pool_slices": pool_slices,
        "full_embeddings": full_embeddings,
        "max_length": max_length,
        "truncate": truncate,
        "truncation": {"enabled": truncate, "max_length": max_length},
        "retained_positions": (
            "biological_residues_in_input_order_after_optional_prefix_crop_before_forward"
        ),
        "batching": {
            "batch_size": batch_size,
            "batch_window_size": batch_window_size,
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
        "descriptor_index": descriptor_index,
        "storage_format": format if output is not None else "memory",
        "software": software_versions,
        "execution": _execution_identity_metadata(model),
        "adapter": _adapter_identity_metadata(model),
        "torch_version": software_versions["torch"],
        "transformers_version": software_versions["transformers"],
        "complete": True,
    }
    if taps is not None:
        metadata["taps"] = taps
    status = getattr(model, "esmc_precision_status", None)
    if status is not None:
        metadata["esmc_precision"] = status.as_dict() if hasattr(status, "as_dict") else status
    return metadata


class EmbeddingMixin:
    """Small delegation mixin shared by FastPLMs model classes."""

    def embed_dataset(
        self, inputs: Any, **kwargs: Any,
    ) -> EmbeddingResult | TapResult | TapRunReceipt:
        return embed_dataset(self, inputs, **kwargs)


__all__ = [
    "EmbeddingMixin",
    "embed_dataset",
    "iter_fasta",
    "parse_fasta",
    "select_hidden_state_embeddings",
]
