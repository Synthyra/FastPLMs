"""Deterministic identity for embedding inputs, models, tokenizers, and execution."""

from __future__ import annotations

import hashlib
import json
import platform
import torch
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from torch import Tensor

from .inputs import _InputSpool
from .storage import tensor_sha256
from .types import EmbeddingInput


_RUN_FINGERPRINT_SCHEMA_VERSION = 3
_MODEL_STATE_HASH_CHUNK_BYTES = 16 * 1024**2


def _model_device(model: Any) -> torch.device:
    try:
        return torch.device(next(model.parameters()).device)
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


def _fingerprint_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _fingerprint_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_fingerprint_jsonable(item) for item in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "sha256": tensor_sha256(value),
        }
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {
        "class": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
        "value": str(value),
    }


def _tokenizer_content_sha256(tokenizer: Any) -> str:
    content: dict[str, Any] = {
        "init_kwargs": getattr(tokenizer, "init_kwargs", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
    }
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        content["vocabulary"] = get_vocab()
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added_vocab):
        content["added_vocabulary"] = get_added_vocab()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_to_str = getattr(backend, "to_str", None)
    if callable(backend_to_str):
        content["backend"] = backend_to_str()
    serialized = json.dumps(
        _fingerprint_jsonable(content),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _tokenizer_metadata(model: Any, tokenizer: Any | None) -> dict[str, Any]:
    resolved = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
    if resolved is None:
        # Raw-sequence families such as E1 retain their loader context on the
        # model/encoder rather than exposing a Transformers tokenizer. Bind the
        # non-secret source policy to resume identity without serializing a Hub
        # token or forcing lazy tokenizer initialization.
        for candidate in (model, getattr(model, "model", None)):
            settings = getattr(candidate, "__dict__", {}).get("_fastplms_tokenizer_kwargs")
            if isinstance(settings, Mapping):
                token_value = settings.get("token")
                return {
                    "mode": "native-sequence",
                    "source": (
                        str(settings.get("tokenizer_source"))
                        if settings.get("tokenizer_source") is not None
                        else None
                    ),
                    "revision": settings.get("revision"),
                    "cache_dir": (
                        str(settings.get("cache_dir"))
                        if settings.get("cache_dir") is not None
                        else None
                    ),
                    "local_files_only": bool(settings.get("local_files_only", False)),
                    "token_policy": (
                        "disabled"
                        if token_value is False
                        else "provided"
                        if token_value is not None
                        else "default"
                    ),
                }
        return {"mode": "native-sequence"}
    return {
        "mode": "tokenizer",
        "class": f"{resolved.__class__.__module__}.{resolved.__class__.__qualname__}",
        "name_or_path": getattr(resolved, "name_or_path", None),
        "vocab_size": getattr(resolved, "vocab_size", None),
        "special_token_ids": list(getattr(resolved, "all_special_ids", ())),
        "content_sha256": _tokenizer_content_sha256(resolved),
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
        "python": platform.python_version(),
        "safetensors": safetensors_version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers_version,
    }


def _adapter_identity_metadata(model: Any) -> dict[str, Any] | None:
    """Return deterministic PEFT/adapter identity without tensor payloads."""

    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, Mapping) or not peft_config:
        return None
    configurations: dict[str, Any] = {}
    for name, config in sorted(peft_config.items(), key=lambda item: str(item[0])):
        to_dict = getattr(config, "to_dict", None)
        if callable(to_dict):
            value = to_dict()
        else:
            try:
                value = vars(config)
            except TypeError:
                value = config
        configurations[str(name)] = _fingerprint_jsonable(value)
    active_adapters = getattr(model, "active_adapters", None)
    if callable(active_adapters):
        active_adapters = active_adapters()
    return {
        "active": _fingerprint_jsonable(active_adapters),
        "configurations": configurations,
    }


def _execution_identity_metadata(model: Any) -> dict[str, Any]:
    """Capture runtime policy that can change persisted numerical results."""

    parameter_dtypes = sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in getattr(model, "parameters", lambda: ())()
        }
    )
    return {
        "device": _model_device(model).type,
        "hf_device_map": _fingerprint_jsonable(getattr(model, "hf_device_map", None)),
        "parameter_dtypes": parameter_dtypes,
        "software": _software_versions(),
    }


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
        "weights_revision": getattr(config, "fastplms_weights_revision", None),
        "runtime_revision": getattr(config, "fastplms_runtime_revision", None),
        "source_tree_sha256": getattr(config, "fastplms_source_tree_sha256", None),
        "runtime_bundle_sha256": getattr(config, "fastplms_runtime_bundle_sha256", None),
    }


def _bounded_tensor_chunks(X: Tensor, max_elements: int) -> Iterable[Tensor]:
    """Yield X in logical row-major order without materializing a full copy."""

    # X: (...)
    if X.numel() == 0:
        return
    if X.ndim == 0:
        yield X
        return
    trailing_elements = 1
    for size in X.shape[1:]:
        trailing_elements *= int(size)
    if trailing_elements <= max_elements:
        rows_per_chunk = max(1, max_elements // trailing_elements)
        for start in range(0, X.shape[0], rows_per_chunk):
            yield X[start : start + rows_per_chunk]  # (chunk_rows, ...)
        return
    for row in X:
        yield from _bounded_tensor_chunks(row, max_elements)


def _model_state_sha256(model: Any) -> str:
    """Hash named parameters and persistent buffers using bounded CPU copies."""

    # Never cache this digest from tensor identity or ``Tensor._version``.
    # ``Parameter.data`` and independent tensor aliases can mutate shared storage
    # without changing either signal, while persisted resume identity must bind
    # the authoritative bytes visible at the start of this run.
    state = model.state_dict(keep_vars=True)
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, Tensor):
            raise TypeError(f"Model state entry {name!r} is not a tensor.")
        if value.is_meta:
            raise ValueError(
                f"Cannot fingerprint meta-device model state entry {name!r}; pass "
                "model_state_fingerprint with a caller-owned state identity."
            )
        header = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype).removeprefix("torch."),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        max_elements = max(1, _MODEL_STATE_HASH_CHUNK_BYTES // value.element_size())
        for chunk in _bounded_tensor_chunks(value.detach(), max_elements):
            cpu_chunk = chunk.to(device="cpu").contiguous()  # chunk.shape
            digest.update(cpu_chunk.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _input_sha256(records: Iterable[EmbeddingInput]) -> str:
    """Hash an ordered input stream without constructing a duplicate JSON payload."""

    precomputed = getattr(records, "input_fingerprint", None)
    if isinstance(precomputed, str):
        return precomputed
    digest = hashlib.sha256()
    count = 0
    for record in records:
        count += 1
        for value in (record.id, record.sequence):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    digest.update(count.to_bytes(8, "big"))
    return digest.hexdigest()


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
    model_state_fingerprint: str | None,
    persist_output: bool,
    embedding_context: Mapping[str, Any],
    batch_size: int,
    batch_window_size: int,
    max_tokens_per_batch: int | None,
) -> tuple[str, str, str | None, str]:
    input_fingerprint = _input_sha256(records)
    attention_backend = _attention_backend(model)
    model_identity = _model_identity_metadata(model)
    if model_state_fingerprint is None and persist_output:
        resolved_model_state_fingerprint = _model_state_sha256(model)
        model_state_fingerprint_source = "computed"
    elif model_state_fingerprint is not None:
        resolved_model_state_fingerprint = model_state_fingerprint.strip()
        if not resolved_model_state_fingerprint:
            raise ValueError("model_state_fingerprint must not be empty.")
        model_state_fingerprint_source = "caller"
    else:
        resolved_model_state_fingerprint = None
        model_state_fingerprint_source = "not-computed"
    payload = {
        "fingerprint_schema_version": _RUN_FINGERPRINT_SCHEMA_VERSION,
        "input_fingerprint": input_fingerprint,
        "model_state_fingerprint": resolved_model_state_fingerprint,
        "model_state_fingerprint_source": model_state_fingerprint_source,
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
        "adapter": _adapter_identity_metadata(model),
        "execution": _execution_identity_metadata(model),
        "embedding_context": _fingerprint_jsonable(embedding_context),
        "pooling": list(pooling),
        "full_embeddings": full_embeddings,
        "max_length": max_length,
        "truncate": truncate,
        "dtype": str(dtype) if dtype is not None else None,
        "batching": {
            "batch_size": batch_size,
            "batch_window_size": batch_window_size,
            "max_tokens_per_batch": max_tokens_per_batch,
            "input_storage": ("disk-spool" if isinstance(records, _InputSpool) else "memory"),
        },
        "model_kwargs": {
            key: _fingerprint_jsonable(value) for key, value in sorted(model_kwargs.items())
        },
        "residue_mask_policy": "attention-mask-minus-special-tokens",
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return (
        input_fingerprint,
        run_fingerprint,
        resolved_model_state_fingerprint,
        model_state_fingerprint_source,
    )


def _ordered_string_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    digest.update(len(values).to_bytes(8, "big"))
    return digest.hexdigest()


def _embedding_context(
    model: Any,
    records: Sequence[EmbeddingInput],
    *,
    hidden_state_source: str,
    decoder_inputs: Sequence[str] | None,
    decoder_input_ids: Tensor | None,
    decoder_attention_mask: Tensor | None,
    model_kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...] | None]:
    if hidden_state_source not in {"encoder", "decoder"}:
        raise ValueError("hidden_state_source must be 'encoder' or 'decoder'.")
    hidden_state_index = model_kwargs.get("hidden_state_index", -1)
    if not isinstance(hidden_state_index, int) or isinstance(hidden_state_index, bool):
        raise TypeError("hidden_state_index must be an integer.")
    store_all_hidden_states = model_kwargs.get("store_all_hidden_states", False)
    if not isinstance(store_all_hidden_states, bool):
        raise TypeError("store_all_hidden_states must be a boolean.")
    normalized_decoder_inputs: tuple[str, ...] | None = None
    has_decoder_inputs = decoder_inputs is not None
    has_decoder_ids = decoder_input_ids is not None
    if hidden_state_source == "encoder":
        if has_decoder_inputs or has_decoder_ids or decoder_attention_mask is not None:
            raise ValueError("Decoder inputs are only valid when hidden_state_source='decoder'.")
    else:
        if has_decoder_inputs == has_decoder_ids:
            raise ValueError(
                "Decoder embedding requires exactly one of decoder_inputs or decoder_input_ids."
            )
    decoder_input_fingerprint: str | None = None
    if decoder_inputs is not None:
        if isinstance(decoder_inputs, (str, bytes)) or not isinstance(decoder_inputs, Sequence):
            raise TypeError("decoder_inputs must be an aligned sequence of strings.")
        normalized_decoder_inputs = tuple(decoder_inputs)
        if not all(isinstance(value, str) and value for value in normalized_decoder_inputs):
            raise ValueError("decoder_inputs must contain non-empty strings.")
        if len(normalized_decoder_inputs) != len(records):
            raise ValueError("decoder_inputs must align one-to-one with embedding inputs.")
        decoder_input_fingerprint = _ordered_string_sha256(normalized_decoder_inputs)
        if decoder_attention_mask is not None:
            raise ValueError("decoder_attention_mask requires decoder_input_ids.")
    if decoder_input_ids is not None:
        if not isinstance(decoder_input_ids, Tensor) or decoder_input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must have shape (batch, sequence).")
        if decoder_input_ids.shape[0] != len(records):
            raise ValueError("decoder_input_ids must align one-to-one with embedding inputs.")
        if decoder_input_ids.dtype == torch.bool or decoder_input_ids.is_floating_point():
            raise TypeError("decoder_input_ids must use an integer token dtype.")
        decoder_input_fingerprint = tensor_sha256(decoder_input_ids)
    decoder_mask_fingerprint: str | None = None
    if decoder_attention_mask is not None:
        if not isinstance(decoder_attention_mask, Tensor):
            raise TypeError("decoder_attention_mask must be a tensor.")
        if decoder_input_ids is None or decoder_attention_mask.shape != decoder_input_ids.shape:
            raise ValueError("decoder_attention_mask must match decoder_input_ids shape.")
        decoder_mask_fingerprint = tensor_sha256(decoder_attention_mask)

    context: dict[str, Any] = {
        "hidden_state_source": hidden_state_source,
        "hidden_state_index": hidden_state_index,
        "store_all_hidden_states": store_all_hidden_states,
        "decoder_input_fingerprint": decoder_input_fingerprint,
        "decoder_attention_mask_fingerprint": decoder_mask_fingerprint,
        "decoder_alignment": "input-position" if hidden_state_source == "decoder" else None,
    }
    metadata_hook = getattr(model, "_embedding_metadata", None)
    model_metadata: Mapping[str, Any] | None = None
    if callable(metadata_hook):
        model_metadata = metadata_hook(**context)
        if not isinstance(model_metadata, Mapping):
            raise TypeError("_embedding_metadata must return a mapping.")
        context["model_embedding"] = _fingerprint_jsonable(model_metadata)
    if hidden_state_source == "decoder":
        has_decoder_batch = callable(getattr(model, "_embedding_batch", None))
        declares_decoder_stack = (
            model_metadata is not None and model_metadata.get("hidden_state_stack") == "decoder"
        )
        if not has_decoder_batch or not declares_decoder_stack:
            raise ValueError(
                f"{model.__class__.__name__} does not declare decoder embedding support."
            )
    return context, normalized_decoder_inputs
