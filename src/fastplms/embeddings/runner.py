"""Model-independent dataset embedding orchestration."""

from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .pooling import Pooler
from .storage import (
    SafetensorsStreamWriter,
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
_RUN_FINGERPRINT_SCHEMA_VERSION = 3
_MODEL_STATE_HASH_CHUNK_BYTES = 16 * 1024**2
_DEFAULT_BATCH_WINDOW_MULTIPLIER = 16
_SUPPORTED_STORAGE_FORMATS = frozenset({"safetensors", "sqlite"})


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


def iter_fasta(path: str | Path) -> Iterator[EmbeddingInput]:
    """Yield FASTA records in source order without reading the file into memory."""

    identifier: str | None = None
    sequence_parts: list[str] = []
    found_record = False
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    found_record = True
                    yield EmbeddingInput(identifier, "".join(sequence_parts))
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
        found_record = True
        yield EmbeddingInput(identifier, "".join(sequence_parts))
    if not found_record:
        raise ValueError(f"No FASTA records found in {path}.")


def parse_fasta(path: str | Path) -> list[EmbeddingInput]:
    """Parse FASTA records while preserving identifiers, order, and duplicates."""

    return list(iter_fasta(path))


def _normalize_input_item(
    position: int,
    item: str | EmbeddingInput | tuple[str, str],
) -> EmbeddingInput:
    if isinstance(item, EmbeddingInput):
        return item
    if isinstance(item, str):
        return EmbeddingInput(str(position), item)
    if isinstance(item, tuple) and len(item) == 2:
        return EmbeddingInput(str(item[0]), str(item[1]))
    raise TypeError(
        "inputs must contain sequences, EmbeddingInput values, or (id, sequence) tuples."
    )


class _InputSpool(Sequence[EmbeddingInput]):
    """Immutable disk-backed normalized inputs with an incremental digest."""

    def __init__(
        self,
        values: Iterable[str | EmbeddingInput | tuple[str, str]],
    ) -> None:
        self._temporary: tempfile.TemporaryDirectory[str] | None = (
            tempfile.TemporaryDirectory(prefix="fastplms-inputs-")
        )
        self.path = Path(self._temporary.name) / "inputs.sqlite"
        self._connection: sqlite3.Connection | None = sqlite3.connect(self.path)
        self._connection.execute(
            "CREATE TABLE inputs ("
            "position INTEGER PRIMARY KEY, input_id TEXT NOT NULL, sequence TEXT NOT NULL)"
        )
        digest = hashlib.sha256()
        count = 0
        pending: list[tuple[int, str, str]] = []
        try:
            for position, item in enumerate(values):
                record = _normalize_input_item(position, item)
                for value in (record.id, record.sequence):
                    encoded = value.encode("utf-8")
                    digest.update(len(encoded).to_bytes(8, "big"))
                    digest.update(encoded)
                pending.append((position, record.id, record.sequence))
                count += 1
                if len(pending) == 1_024:
                    self._connection.executemany("INSERT INTO inputs VALUES (?, ?, ?)", pending)
                    pending.clear()
            if pending:
                self._connection.executemany("INSERT INTO inputs VALUES (?, ?, ?)", pending)
            if count == 0:
                raise ValueError("inputs must contain at least one sequence.")
            self._connection.commit()
            self._connection.close()
            self._connection = sqlite3.connect(
                f"{self.path.resolve().as_uri()}?mode=ro",
                uri=True,
            )
        except BaseException:
            self.close()
            raise
        digest.update(count.to_bytes(8, "big"))
        self.input_fingerprint = digest.hexdigest()
        self._count = count

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Input spool is closed.")
        return self._connection

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[EmbeddingInput]:
        cursor = self._require_connection().execute(
            "SELECT input_id, sequence FROM inputs ORDER BY position"
        )
        while rows := cursor.fetchmany(1_024):
            for input_id, sequence in rows:
                yield EmbeddingInput(input_id, sequence)

    def __getitem__(
        self, index: int | slice
    ) -> EmbeddingInput | list[EmbeddingInput]:
        connection = self._require_connection()
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            if step != 1:
                return [self[position] for position in range(start, stop, step)]
            rows = connection.execute(
                "SELECT input_id, sequence FROM inputs "
                "WHERE position >= ? AND position < ? ORDER BY position",
                (start, stop),
            ).fetchall()
            return [EmbeddingInput(input_id, sequence) for input_id, sequence in rows]
        position = index + self._count if index < 0 else index
        if position < 0 or position >= self._count:
            raise IndexError(index)
        row = connection.execute(
            "SELECT input_id, sequence FROM inputs WHERE position = ?", (position,)
        ).fetchone()
        if row is None:
            raise IndexError(index)
        return EmbeddingInput(row[0], row[1])

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None
        temporary = getattr(self, "_temporary", None)
        if temporary is not None:
            temporary.cleanup()
            self._temporary = None

    def __del__(self) -> None:
        self.close()


def _normalize_inputs(
    inputs: (Iterable[str | EmbeddingInput | tuple[str, str]] | Mapping[str, str] | str | Path),
    *,
    disk_backed: bool,
) -> Sequence[EmbeddingInput]:
    is_fasta_path = isinstance(inputs, Path)
    if isinstance(inputs, str):
        try:
            is_fasta_path = Path(inputs).is_file()
        except OSError:
            is_fasta_path = False
    should_spool = disk_backed or is_fasta_path or not isinstance(
        inputs, (str, Sequence, Mapping)
    )
    if is_fasta_path:
        values: Iterable[str | EmbeddingInput | tuple[str, str]] = iter_fasta(inputs)
    elif isinstance(inputs, str):
        values: Iterable[str | EmbeddingInput | tuple[str, str]] = [inputs]
    elif isinstance(inputs, Mapping):
        values = inputs.items()
    else:
        values = inputs
    if should_spool:
        return _InputSpool(values)
    records: list[EmbeddingInput] = []
    for position, item in enumerate(values):
        records.append(_normalize_input_item(position, item))
    if not records:
        raise ValueError("inputs must contain at least one sequence.")
    return records


def _validate_untruncated_lengths(
    records: Sequence[EmbeddingInput],
    *,
    max_length: int | None,
    truncate: bool,
) -> None:
    """Fail before inference when a biological-residue limit would be exceeded."""

    if max_length is None or truncate:
        return
    for position, record in enumerate(records):
        residue_count = len(record.sequence)
        if residue_count > max_length:
            raise ValueError(
                f"Input at position {position} with id {record.id!r} has "
                f"{residue_count} biological residues, exceeding max_length={max_length} "
                "while truncate=False."
            )


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
            settings = getattr(candidate, "__dict__", {}).get(
                "_fastplms_tokenizer_kwargs"
            )
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


@contextmanager
def _temporary_eval(model: Any) -> Iterator[None]:
    was_training = getattr(model, "training", None)
    eval_method = getattr(model, "eval", None)
    train_method = getattr(model, "train", None)
    if (
        not isinstance(was_training, bool)
        or not callable(eval_method)
        or not callable(train_method)
    ):
        yield
        return
    eval_method()
    try:
        yield
    finally:
        train_method(was_training)


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
    if max_length is not None and truncate:
        # ``max_length`` is a biological-residue limit. Tokenizer limits include
        # boundary tokens, so reserve their declared width instead of dropping
        # residues at the exact boundary.
        special_token_count = 0
        num_special_tokens_to_add = getattr(tokenizer, "num_special_tokens_to_add", None)
        if callable(num_special_tokens_to_add):
            special_token_count = int(num_special_tokens_to_add(pair=False))
        tokenize_kwargs["max_length"] = max_length + special_token_count
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
        "weights_revision": getattr(config, "fastplms_weights_revision", None),
        "runtime_revision": getattr(config, "fastplms_runtime_revision", None),
        "source_tree_sha256": getattr(config, "fastplms_source_tree_sha256", None),
        "runtime_bundle_sha256": getattr(
            config, "fastplms_runtime_bundle_sha256", None
        ),
    }


def _bounded_tensor_chunks(X: Tensor, max_elements: int) -> Iterable[Tensor]:
    """Yield X in logical row-major order without materializing a full copy."""

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
            yield X[start : start + rows_per_chunk]
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
            cpu_chunk = chunk.to(device="cpu").contiguous()
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
            "input_storage": (
                "disk-spool" if isinstance(records, _InputSpool) else "memory"
            ),
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
            model_metadata is not None
            and model_metadata.get("hidden_state_stack") == "decoder"
        )
        if not has_decoder_batch or not declares_decoder_stack:
            raise ValueError(
                f"{model.__class__.__name__} does not declare decoder embedding support."
            )
    return context, normalized_decoder_inputs


def _planned_batches(
    records: Sequence[EmbeddingInput],
    positions: range,
    *,
    batch_size: int,
    max_tokens_per_batch: int | None,
    max_length: int | None,
    truncate: bool,
) -> Iterator[list[int]]:
    """Length-bucket one bounded window while retaining stable output positions."""

    def effective_length(position: int) -> int:
        length = len(records[position].sequence)
        return min(length, max_length) if truncate and max_length is not None else length

    ordered = sorted(positions, key=lambda position: (-effective_length(position), position))
    batch: list[int] = []
    longest = 0
    for position in ordered:
        length = effective_length(position)
        if max_tokens_per_batch is not None and length > max_tokens_per_batch:
            raise ValueError(
                f"Input at position {position} has {length} residues, exceeding "
                f"max_tokens_per_batch={max_tokens_per_batch}."
            )
        candidate_longest = max(longest, length)
        exceeds_tokens = (
            max_tokens_per_batch is not None
            and candidate_longest * (len(batch) + 1) > max_tokens_per_batch
        )
        if batch and (len(batch) >= batch_size or exceeds_tokens):
            yield batch
            batch = []
            longest = 0
        batch.append(position)
        longest = max(longest, length)
    if batch:
        yield batch


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
    if _embedding_batch_identity is not None and not isinstance(
        _embedding_batch_identity, Mapping
    ):
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
            "_allowed_unsupported_pooling is only valid with a family-specific "
            "_embedding_batch_fn."
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
        embedding_context["family_adapter"] = _fingerprint_jsonable(
            _embedding_batch_identity
        )
        if allowed_unsupported_pooling:
            embedding_context["family_adapter_pooling_override"] = sorted(
                allowed_unsupported_pooling
            )

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
    output_already_exists = output is not None and _output_exists(output, format)
    existing: EmbeddingResult | None = None
    start_position = 0
    if output is not None and resume and output_already_exists:
        if format == "sqlite":
            try:
                existing = load_sqlite_result(output, run_id=run_fingerprint)
            except KeyError:
                existing = load_result(output, format=format)
        else:
            existing = load_result(output, format=format)
        if existing.metadata.get("fingerprint_schema_version") != (_RUN_FINGERPRINT_SCHEMA_VERSION):
            raise ValueError(
                "Existing embeddings use an incompatible run fingerprint schema; "
                "choose another output or set resume=False."
            )
        if existing.metadata.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                "Existing embeddings were produced by a different run fingerprint; "
                "choose another output or set resume=False."
            )
        if len(existing) > len(records):
            raise ValueError(
                "Existing embeddings are not an ordered prefix of the requested inputs."
            )
        prefix_matches = all(
            (observed.id, observed.sequence) == (expected.id, expected.sequence)
            for expected, observed in zip(records, existing, strict=False)
        )
        if not prefix_matches:
            raise ValueError(
                "Existing embeddings are not an ordered prefix of the requested inputs."
            )
        if len(existing) == len(records) and existing.metadata.get("complete", True):
            return existing
        start_position = len(existing)

    sqlite_run_id: str | None = None
    sqlite_replace_on_first_commit = False
    sqlite_initial_metadata: dict[str, Any] | None = None
    if output is not None and format == "sqlite":
        sqlite_initial_metadata = {
            "format_version": 1,
            "fingerprint_schema_version": _RUN_FINGERPRINT_SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "input_fingerprint": input_fingerprint,
            "model_state_fingerprint": resolved_model_state_fingerprint,
            "model_state_fingerprint_source": model_state_fingerprint_source,
            "complete": False,
        }
        sqlite_run_id = run_fingerprint
        if not resume and output_already_exists:
            try:
                load_sqlite_result(output, run_id=run_fingerprint)
            except KeyError:
                pass
            else:
                # Keep an exact prior run readable until replacement inference
                # has produced the first complete commit window.
                sqlite_replace_on_first_commit = True
        if not sqlite_replace_on_first_commit:
            initialize_sqlite_run(
                output,
                sqlite_initial_metadata,
                resume=resume,
            )

    stream_safetensors = output is not None and format == "safetensors"
    attention_backend = _attention_backend(model)
    output_records: list[EmbeddingRecord] = (
        [] if sqlite_run_id is not None or stream_safetensors else list(existing or ())
    )
    output_descriptors: list[dict[str, Any]] | None = [] if output is None else None
    pool_slices: dict[str, tuple[int, int]] = {}
    if existing and pooler is not None:
        pooled_width = existing[0].load_tensor().shape[-1]
        if pooled_width % len(pooling_names) != 0:
            raise ValueError("Stored pooled width is inconsistent with pooling metadata.")
        pool_slices = pooler.output_slices(pooled_width // len(pooling_names))

    safetensors_writer: SafetensorsStreamWriter | None = None
    if stream_safetensors:
        if output is None:
            raise RuntimeError(
                "Safetensors streaming was enabled without an output destination."
            )
        transactional_overwrite = output_already_exists and not resume
        safetensors_writer = SafetensorsStreamWriter(
            output,
            {
                "format_version": 1,
                "fingerprint_schema_version": _RUN_FINGERPRINT_SCHEMA_VERSION,
                "run_fingerprint": run_fingerprint,
                "input_fingerprint": input_fingerprint,
                "model_state_fingerprint": resolved_model_state_fingerprint,
                "model_state_fingerprint_source": model_state_fingerprint_source,
                "complete": False,
            },
            shard_size=shard_size,
            existing=existing or (),
            reuse_existing=bool(resume and existing is not None),
            publish_initial=not transactional_overwrite,
            publish_incremental=not transactional_overwrite,
        )
    need_attentions = "parti" in pooling_names

    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    resolved_tokenizer = tokenizer if tokenizer is not None else getattr(model, "tokenizer", None)
    with _temporary_eval(model), torch.inference_mode():
        for window_start in range(start_position, len(records), resolved_batch_window_size):
            window_stop = min(window_start + resolved_batch_window_size, len(records))
            window_records = records[window_start:window_stop]
            if not isinstance(window_records, Sequence):
                raise RuntimeError("The immutable embedding spool returned a non-sequence window.")
            window_results: dict[int, EmbeddingRecord] = {}
            for local_positions in _planned_batches(
                window_records,
                range(len(window_records)),
                batch_size=batch_size,
                max_tokens_per_batch=max_tokens_per_batch,
                max_length=max_length,
                truncate=truncate,
            ):
                batch_positions = [window_start + position for position in local_positions]
                batch_records = [window_records[position] for position in local_positions]
                sequences = [
                    record.sequence[:max_length]
                    if truncate and max_length is not None
                    else record.sequence
                    for record in batch_records
                ]
                batch_model_kwargs = dict(model_kwargs)
                if model_type == "fast_ankh" or hidden_state_source == "decoder":
                    batch_model_kwargs["hidden_state_source"] = hidden_state_source
                    if normalized_decoder_inputs is not None:
                        batch_model_kwargs["decoder_inputs"] = [
                            normalized_decoder_inputs[position] for position in batch_positions
                        ]
                    if decoder_input_ids is not None:
                        indices = torch.tensor(
                            batch_positions,
                            device=decoder_input_ids.device,
                            dtype=torch.long,
                        )
                        batch_model_kwargs["decoder_input_ids"] = decoder_input_ids.index_select(
                            0, indices
                        )
                    if decoder_attention_mask is not None:
                        indices = torch.tensor(
                            batch_positions,
                            device=decoder_attention_mask.device,
                            dtype=torch.long,
                        )
                        batch_model_kwargs["decoder_attention_mask"] = (
                            decoder_attention_mask.index_select(0, indices)
                        )
                custom_batch = _embedding_batch_fn or getattr(model, "_embedding_batch", None)
                if custom_batch is not None:
                    if model_type == "fast_ankh":
                        batch = custom_batch(
                            sequences,
                            tokenizer=resolved_tokenizer,
                            max_length=max_length,
                            truncate=truncate,
                            need_attentions=need_attentions,
                            **batch_model_kwargs,
                        )
                    else:
                        batch = custom_batch(sequences, **batch_model_kwargs)
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
                        model_kwargs=batch_model_kwargs,
                    )
                X = batch.X
                M = batch.residue_mask.to(device=X.device, dtype=torch.bool)
                if need_attentions:
                    # Custom embedding adapters own their inference path, so retain
                    # the same fail-closed length contract at their boundary.
                    _validate_parti_length(M)
                valid_X_shape = X.ndim == 3 and M.shape == X.shape[:2]
                valid_all_states_shape = (
                    X.ndim == 4
                    and store_all_hidden_states
                    and full_embeddings
                    and M.shape == (X.shape[0], X.shape[2])
                )
                if not (valid_X_shape or valid_all_states_shape):
                    raise ValueError(
                        "Embedding batches must provide X with shape (b, l, d), or "
                        "(b, states, l, d) when storing all hidden states, and "
                        "residue_mask with shape (b, l)."
                    )
                if dtype is not None:
                    X = X.to(dtype=dtype)

                if full_embeddings:
                    if X.ndim == 4:
                        values = [
                            X_i[:, M_i, :].detach().cpu() for X_i, M_i in zip(X, M, strict=True)
                        ]
                    else:
                        values = [X_i[M_i].detach().cpu() for X_i, M_i in zip(X, M, strict=True)]
                else:
                    if pooler is None:
                        raise RuntimeError(
                            "Pooled embedding output was requested without an initialized pooler."
                        )
                    Y = pooler(
                        X,
                        M,
                        attentions=batch.attentions,
                        attention_backend=attention_backend,
                    )
                    pool_slices = pooler.output_slices(X.shape[-1])
                    values = list(Y.detach().cpu().unbind(0))
                for position, record, value in zip(
                    batch_positions, batch_records, values, strict=True
                ):
                    window_results[position] = EmbeddingRecord(record.id, record.sequence, value)

            new_records = [
                window_results[position] for position in range(window_start, window_stop)
            ]
            if output_descriptors is not None:
                output_descriptors.extend(
                    _output_descriptor(window_start + offset, record)
                    for offset, record in enumerate(new_records)
                )
            if output is not None and sqlite_run_id is not None:
                append_sqlite_records(
                    output,
                    sqlite_run_id,
                    window_start,
                    new_records,
                    replace_metadata=(
                        sqlite_initial_metadata
                        if sqlite_replace_on_first_commit
                        else None
                    ),
                )
                sqlite_replace_on_first_commit = False
            elif safetensors_writer is not None:
                safetensors_writer.append(new_records)
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
            "input_storage": (
                "disk-spool" if isinstance(records, _InputSpool) else "memory"
            ),
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
    if output_descriptors is not None:
        metadata["outputs"] = output_descriptors
        metadata["tensor_hashes"] = [item["sha256"] for item in output_descriptors]
    status = getattr(model, "esmc_precision_status", None)
    if status is not None:
        metadata["esmc_precision"] = status.as_dict() if hasattr(status, "as_dict") else status
    if output is not None and sqlite_run_id is not None:
        update_sqlite_run_metadata(output, sqlite_run_id, metadata)
        return load_sqlite_result(output, run_id=sqlite_run_id)
    if safetensors_writer is not None:
        return safetensors_writer.publish(complete=True, metadata=metadata)
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
    "iter_fasta",
    "parse_fasta",
    "select_hidden_state_embeddings",
]
