"""Lossless, reproducible storage for :mod:`fastplms.embeddings`."""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import struct
import numpy as np
import torch
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast, overload
from uuid import uuid4
from torch import Tensor

from .types import (
    EmbeddingRecord,
    EmbeddingResult,
    LazyTensorReference,
)


_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_NAME_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}
DEFAULT_SHARD_SIZE = 2 * 1024**3
_MAX_RECORDS_PER_DESCRIPTOR_SHARD = 1_024
_TENSOR_HASH_CHUNK_BYTES = 16 * 1024**2


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _persistent_metadata(
    metadata: dict[str, Any],
    *,
    descriptor_index: str,
    record_count: int | None = None,
) -> dict[str, Any]:
    """Remove per-record copies from metadata and identify the authoritative index."""

    cleaned_value = _jsonable(metadata)
    if not isinstance(cleaned_value, dict):
        raise TypeError("Embedding metadata must serialize to a JSON object.")
    cleaned: dict[str, Any] = cleaned_value
    cleaned.pop("outputs", None)
    cleaned.pop("tensor_hashes", None)
    cleaned["descriptor_index"] = descriptor_index
    if record_count is not None:
        cleaned["record_count"] = record_count
    return cleaned


def _tensor_bytes(X: Tensor) -> bytes:
    """Return the exact contiguous byte representation of X."""

    # X: (...)
    X = X.detach().cpu().contiguous()  # (...)
    return X.view(torch.uint8).numpy().tobytes()


def _bounded_tensor_chunks(X: Tensor, max_bytes: int) -> Iterator[Tensor]:
    """Yield row-major CPU chunks without materializing one full byte string."""

    # X: (...)
    flattened = X.detach().to(device="cpu").reshape(-1)  # (n,)
    if flattened.numel() == 0:
        return
    chunk_elements = max(1, max_bytes // flattened.element_size())
    for start in range(0, flattened.numel(), chunk_elements):
        chunk = flattened[start : start + chunk_elements]  # (n_chunk,)
        if chunk.stride(0) != 1:
            chunk = chunk.clone(memory_format=torch.contiguous_format)  # (n_chunk,)
        yield chunk  # (n_chunk,)


def _tensor_hash_chunks(X: Tensor) -> Iterator[bytes]:
    for chunk in _bounded_tensor_chunks(X, _TENSOR_HASH_CHUNK_BYTES):
        yield chunk.view(torch.uint8).numpy().tobytes()


def tensor_sha256(X: Tensor) -> str:
    """Hash dtype, shape, and exact tensor bytes."""

    if not isinstance(X, Tensor):
        raise TypeError("X must be a tensor.")
    if X.dtype not in _DTYPE_NAMES:
        raise TypeError(f"Unsupported tensor dtype {X.dtype}.")
    if X.is_meta:
        raise ValueError("Cannot hash a meta tensor without storage.")
    if X.layout != torch.strided:
        raise TypeError("Only strided tensors can be hashed.")
    digest = hashlib.sha256()
    digest.update(_DTYPE_NAMES[X.dtype].encode())
    digest.update(json.dumps(tuple(X.shape)).encode())
    for chunk in _tensor_hash_chunks(X):
        digest.update(chunk)
    return digest.hexdigest()


def _encode_tensor(X: Tensor) -> tuple[str, str, bytes]:
    if X.dtype not in _DTYPE_NAMES:
        raise TypeError(f"Unsupported tensor dtype {X.dtype}.")
    shape = json.dumps(tuple(X.shape), separators=(",", ":"))
    return _DTYPE_NAMES[X.dtype], shape, _tensor_bytes(X)


def _decode_tensor(dtype_name: str, shape_json: str, data: bytes) -> Tensor:
    try:
        dtype = _NAME_DTYPES[dtype_name]
    except KeyError as error:
        raise ValueError(f"Unsupported stored dtype {dtype_name!r}.") from error
    shape = tuple(json.loads(shape_json))
    # uint8 is used only as a byte-level carrier, preserving BF16 bits exactly.
    byte_array = np.frombuffer(data, dtype=np.uint8).copy()  # (n_bytes,)
    X = torch.from_numpy(byte_array).view(dtype)  # (n_elements,)
    return X.reshape(shape).clone()  # shape


def _index_path(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix == ".json":
        return path
    if path.suffix == ".safetensors":
        return path.with_suffix(".json")
    return path / "index.json"


def _run_manifest_path(path: str | Path) -> Path:
    path = Path(path)
    if path.name == "index.json":
        return path.with_name("run.json")
    if path.suffix == ".json":
        return path.with_name(f"{path.stem}.run.json")
    if path.suffix == ".safetensors":
        return path.with_suffix(".run.json")
    return path / "run.json"


def _resolve_index_child(root: Path, relative: str, *, label: str) -> Path:
    relative_path = Path(relative)
    candidate = (root / relative_path).resolve()
    if relative_path.is_absolute() or candidate.parent != root.resolve():
        raise ValueError(f"Safetensors {label} references a file outside its output directory.")
    return candidate


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_authoritative_index(
    path: str | Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Load the index selected by the atomic run-manifest commit record."""

    stable_index_path = _index_path(path)
    run_manifest_path = _run_manifest_path(path)
    if not run_manifest_path.is_file():
        raise ValueError(f"Missing safetensors run manifest: {run_manifest_path}.")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(run_manifest, dict):
        raise ValueError("Safetensors run manifest must contain a JSON object.")
    if run_manifest.get("format") != "fastplms-embedding-run":
        raise ValueError(f"Not a FastPLMs embedding run manifest: {run_manifest_path}.")
    version = run_manifest.get("version")
    index_reference = run_manifest.get("index")
    if not isinstance(index_reference, dict):
        raise ValueError("Safetensors run manifest contains an invalid index reference.")
    if version == 1:
        snapshot = run_manifest.get("index_payload")
        if isinstance(snapshot, dict):
            payload = snapshot
            index_bytes = _canonical_json_bytes(payload)
        elif snapshot is None:
            index_bytes = stable_index_path.read_bytes()
            payload = json.loads(index_bytes.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Safetensors index must contain a JSON object.")
        else:
            raise ValueError("Safetensors run manifest contains an invalid index snapshot.")
        expected = {
            "file": stable_index_path.name,
            "sha256": hashlib.sha256(index_bytes).hexdigest(),
        }
        index_path = stable_index_path
    elif version == 2:
        relative = index_reference.get("file")
        if not isinstance(relative, str):
            raise ValueError("Safetensors run manifest index file is invalid.")
        index_path = _resolve_index_child(stable_index_path.parent, relative, label="run manifest")
        index_bytes = index_path.read_bytes()
        payload = json.loads(index_bytes.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Safetensors generation index must contain a JSON object.")
        if payload.get("version") != 2:
            raise ValueError("Safetensors v2 run manifest must reference a v2 generation index.")
        expected = {
            "file": relative,
            "sha256": hashlib.sha256(index_bytes).hexdigest(),
        }
    else:
        raise ValueError(f"Unsupported safetensors run manifest version {version!r}.")
    if index_reference != expected:
        raise ValueError("Safetensors run manifest does not match its index.")
    if payload.get("format") != "fastplms-embedding-safetensors":
        raise ValueError(f"Not a FastPLMs embedding index: {index_path}.")
    record_count = payload.get("record_count")
    if record_count is None:
        legacy_records = payload.get("records", ())
        if not isinstance(legacy_records, list):
            raise ValueError("Safetensors index contains invalid records.")
        record_count = len(legacy_records)
    if not isinstance(record_count, int) or isinstance(record_count, bool) or record_count < 0:
        raise ValueError("Safetensors record count must be a non-negative integer.")
    if run_manifest.get("record_count") != record_count:
        raise ValueError("Safetensors run manifest record count does not match its index.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Safetensors index metadata must contain a JSON object.")
    if metadata.get("record_count", record_count) != record_count:
        raise ValueError("Safetensors metadata record count does not match its index.")
    if version == 1 and run_manifest.get("metadata") != payload.get("metadata"):
        raise ValueError("Safetensors run manifest metadata does not match its index.")
    return payload, index_path, run_manifest


def safetensors_result_exists(path: str | Path) -> bool:
    """Return whether an authoritative committed safetensors run exists."""

    try:
        _load_authoritative_index(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def _load_safetensor(path: Path, key: str) -> Tensor:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ImportError("Loading embeddings requires the 'safetensors' package.") from error
    with safe_open(path, framework="pt", device="cpu") as handle:
        return cast(Tensor, handle.get_tensor(key))


def _safetensors_shard_prefix(path: str | Path) -> str:
    requested_path = Path(path)
    if requested_path.suffix in {".json", ".safetensors"}:
        return f"{requested_path.stem}-embeddings"
    return "embeddings"


def _authoritative_index_payload(path: str | Path) -> dict[str, Any] | None:
    """Return the last atomically committed generation index when available."""

    try:
        payload, _, _ = _load_authoritative_index(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload


def _referenced_shards(
    index_path: Path,
    payload: dict[str, Any] | None = None,
) -> set[Path]:
    if payload is None:
        payload = _authoritative_index_payload(index_path)
    if payload is None:
        return set()
    shards: set[Path] = set()
    for descriptor_shard in payload.get("descriptor_shards", ()):
        tensor_file = descriptor_shard.get("tensor_file")
        if isinstance(tensor_file, str):
            candidate = _resolve_index_child(
                index_path.parent, tensor_file, label="descriptor index"
            )
            shards.add(candidate)
    for item in payload.get("records", ()):
        relative = item.get("tensor", {}).get("file")
        if not isinstance(relative, str):
            continue
        candidate = (index_path.parent / relative).resolve()
        if candidate.parent == index_path.parent.resolve():
            shards.add(candidate)
    return shards


def _validate_tensor_descriptor(
    tensor: dict[str, Any],
) -> tuple[str, str, tuple[int, ...], str]:
    key = tensor.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("Safetensors descriptor tensor key is invalid.")
    dtype = tensor.get("dtype")
    if not isinstance(dtype, str) or dtype not in _NAME_DTYPES:
        raise ValueError("Safetensors descriptor tensor dtype is invalid.")
    raw_shape = tensor.get("shape")
    if not isinstance(raw_shape, (list, tuple)) or not all(
        isinstance(dimension, int) and not isinstance(dimension, bool) and dimension >= 0
        for dimension in raw_shape
    ):
        raise ValueError("Safetensors descriptor tensor shape is invalid.")
    sha256 = tensor.get("sha256")
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or sha256 != sha256.lower()
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise ValueError("Safetensors descriptor tensor SHA-256 is invalid.")
    return key, dtype, tuple(raw_shape), sha256


def _record_from_safetensors_descriptor(root: Path, item: dict[str, Any]) -> EmbeddingRecord:
    if not isinstance(item, dict):
        raise ValueError("Safetensors record descriptor must contain a JSON object.")
    record_id = item.get("id")
    sequence = item.get("sequence")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("Safetensors descriptor record ID is invalid.")
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("Safetensors descriptor sequence is invalid.")
    tensor = item.get("tensor")
    if not isinstance(tensor, dict):
        raise ValueError("Safetensors descriptor is missing tensor metadata.")
    relative = tensor.get("file")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Safetensors descriptor tensor file is invalid.")
    key, dtype, shape, sha256 = _validate_tensor_descriptor(tensor)
    tensor_path = _resolve_index_child(root, relative, label="descriptor")
    if not tensor_path.is_file():
        raise ValueError(f"Safetensors tensor shard is missing: {relative}.")

    def load_tensor() -> Tensor:
        return _load_safetensor(tensor_path, key)

    reference = LazyTensorReference(
        source=str(tensor_path),
        key=key,
        dtype=dtype,
        shape=shape,
        sha256=sha256,
        _loader=load_tensor,
    )
    return EmbeddingRecord(record_id, sequence, reference)


class _SafetensorsRecordSequence(Sequence[EmbeddingRecord]):
    """Lazy immutable view over bounded descriptor JSONL shards."""

    _fastplms_immutable_sequence = True

    def __init__(self, root: Path, descriptor_shards: Sequence[dict[str, Any]]) -> None:
        if not isinstance(descriptor_shards, (list, tuple)):
            raise ValueError("Safetensors generation index has invalid descriptor shards.")
        self.root = root
        self.shards = tuple(descriptor_shards)
        cumulative: list[int] = []
        total = 0
        for shard in self.shards:
            if not isinstance(shard, dict):
                raise ValueError("Safetensors descriptor shard entry is invalid.")
            relative = shard.get("file")
            declared_count = shard.get("count")
            if (
                not isinstance(declared_count, int)
                or isinstance(declared_count, bool)
                or declared_count < 0
            ):
                raise ValueError("Safetensors descriptor shard count is invalid.")
            declared_sha256 = shard.get("sha256")
            if not isinstance(declared_sha256, str) or len(declared_sha256) != 64:
                raise ValueError("Safetensors descriptor shard SHA-256 is invalid.")
            if not isinstance(relative, str):
                raise ValueError("Safetensors descriptor index file is invalid.")
            descriptor_path = _resolve_index_child(root, relative, label="index")
            tensor_file = shard.get("tensor_file")
            if not isinstance(tensor_file, str):
                raise ValueError("Safetensors descriptor tensor file is invalid.")
            tensor_path = _resolve_index_child(root, tensor_file, label="index")
            if not tensor_path.is_file():
                raise ValueError(f"Safetensors tensor shard is missing: {tensor_file}.")
            digest = hashlib.sha256()
            count = 0
            with descriptor_path.open("rb") as handle:
                for line in handle:
                    digest.update(line)
                    if line.strip():
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            raise ValueError("Safetensors record descriptor must be a JSON object.")
                        item_tensor = item.get("tensor")
                        if not isinstance(item_tensor, dict):
                            raise ValueError("Safetensors descriptor is missing tensor metadata.")
                        item_tensor_file = item_tensor.get("file")
                        if not isinstance(item_tensor_file, str):
                            raise ValueError("Safetensors descriptor tensor file is invalid.")
                        _resolve_index_child(root, item_tensor_file, label="descriptor")
                        if item_tensor_file != tensor_file:
                            raise ValueError(
                                "Safetensors descriptor tensor file does not match its shard."
                            )
                        count += 1
                        _validate_tensor_descriptor(item_tensor)
            if digest.hexdigest() != declared_sha256 or count != declared_count:
                raise ValueError(
                    f"Safetensors descriptor shard failed integrity validation: {relative}."
                )
            total += count
            cumulative.append(total)
        self._cumulative = tuple(cumulative)
        self._count = total

    def __len__(self) -> int:
        return self._count

    def _iter_shard(self, shard_index: int) -> Iterator[EmbeddingRecord]:
        descriptor_path = _resolve_index_child(
            self.root, str(self.shards[shard_index]["file"]), label="index"
        )
        with descriptor_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield _record_from_safetensors_descriptor(self.root, json.loads(line))

    def __iter__(self) -> Iterator[EmbeddingRecord]:
        for shard_index in range(len(self.shards)):
            yield from self._iter_shard(shard_index)

    @overload
    def __getitem__(self, index: int, /) -> EmbeddingRecord: ...

    @overload
    def __getitem__(self, index: slice, /) -> Sequence[EmbeddingRecord]: ...

    def __getitem__(self, index: int | slice) -> EmbeddingRecord | Sequence[EmbeddingRecord]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return [self[position] for position in range(start, stop, step)]
        position = index + self._count if index < 0 else index
        if position < 0 or position >= self._count:
            raise IndexError(index)
        shard_index = bisect_right(self._cumulative, position)
        previous = self._cumulative[shard_index - 1] if shard_index else 0
        local_position = position - previous
        for offset, record in enumerate(self._iter_shard(shard_index)):
            if offset == local_position:
                return record
        raise IndexError(index)


class SafetensorsStreamWriter:
    """Bounded-memory, resumable publisher with immutable retained generations."""

    def __init__(
        self,
        path: str | Path,
        metadata: dict[str, Any],
        *,
        shard_size: int = DEFAULT_SHARD_SIZE,
        existing: Iterable[EmbeddingRecord] = (),
        reuse_existing: bool = False,
        publish_initial: bool = True,
        publish_incremental: bool = True,
    ) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError as error:
            raise ImportError("Saving embeddings requires the 'safetensors' package.") from error
        if shard_size <= 0:
            raise ValueError("shard_size must be positive.")

        self.path = Path(path)
        self.index_path = _index_path(path)
        self.run_manifest_path = _run_manifest_path(path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = _persistent_metadata(
            metadata,
            descriptor_index="safetensors-generation-index",
            record_count=0,
        )
        self.shard_size = shard_size
        self.publish_incremental = publish_incremental
        self._save_file = save_file
        authoritative_payload = _authoritative_index_payload(path)
        prefix = _safetensors_shard_prefix(path)
        # A random generation identity prevents a new writer from reusing a
        # previously published or interrupted generation name. Published files
        # are immutable and remain available to lazy readers until explicit GC.
        self._generation = uuid4().hex
        self._prefix = prefix
        self._shard_index = 0
        self._seed_index = 0
        self._commit_index = 0
        self._descriptor_shards: list[dict[str, Any]] = []
        self._record_count = 0
        self._current: dict[str, Tensor] = {}
        self._pending: list[tuple[EmbeddingRecord, str, str, tuple[int, ...], str]] = []
        self._current_size = 0
        if reuse_existing:
            if authoritative_payload is None:
                raise ValueError("Cannot resume without an authoritative safetensors index.")
            authoritative_metadata = authoritative_payload.get("metadata")
            if not isinstance(authoritative_metadata, dict) or authoritative_metadata.get(
                "run_fingerprint"
            ) != self.metadata.get("run_fingerprint"):
                raise ValueError("Cannot resume a safetensors run with a different fingerprint.")
            expected_prefix_length = (
                len(existing) if isinstance(existing, Sequence) else sum(1 for _ in existing)
            )
            if authoritative_payload.get("version") == 2:
                self._descriptor_shards = list(authoritative_payload.get("descriptor_shards", ()))
                self._record_count = int(authoritative_payload.get("record_count", 0))
            else:
                legacy_records = list(authoritative_payload.get("records", ()))
                self._record_count = len(legacy_records)
                if legacy_records:
                    self._descriptor_shards.extend(self._write_descriptor_seed(legacy_records))
            if expected_prefix_length != self._record_count:
                raise ValueError(
                    "The resumable safetensors prefix does not match the validated "
                    "embedding records."
                )

        if publish_initial:
            self._publish_metadata(complete=False)

    def _write_descriptor_file(
        self,
        name: str,
        descriptors: Sequence[dict[str, Any]],
        *,
        tensor_file: str,
    ) -> dict[str, Any]:
        temporary = self.index_path.parent / f".{name}.tmp"
        destination = self.index_path.parent / name
        if temporary.exists() or destination.exists():
            raise FileExistsError(
                f"Refusing to reuse immutable safetensors generation path {destination}."
            )
        digest = hashlib.sha256()
        with temporary.open("wb") as handle:
            for item in descriptors:
                encoded = (
                    json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
                )
                handle.write(encoded)
                digest.update(encoded)
        temporary.replace(destination)
        return {
            "file": name,
            "sha256": digest.hexdigest(),
            "count": len(descriptors),
            "tensor_file": tensor_file,
        }

    def _write_descriptor_seed(self, records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: list[tuple[str, list[dict[str, Any]]]] = []
        for record in records:
            tensor_file = str(record["tensor"]["file"])
            if (
                not groups
                or groups[-1][0] != tensor_file
                or len(groups[-1][1]) == _MAX_RECORDS_PER_DESCRIPTOR_SHARD
            ):
                groups.append((tensor_file, []))
            groups[-1][1].append(record)
        descriptor_shards: list[dict[str, Any]] = []
        for tensor_file, descriptors in groups:
            self._seed_index += 1
            name = (
                f"{self._prefix}-records-run-{self._generation}-seed-{self._seed_index:05d}.jsonl"
            )
            descriptor_shards.append(
                self._write_descriptor_file(name, descriptors, tensor_file=tensor_file)
            )
        return descriptor_shards

    def _write_shard(self) -> None:
        if not self._current:
            return
        self._shard_index += 1
        name = f"{self._prefix}-run-{self._generation}-{self._shard_index:05d}.safetensors"
        temporary = self.index_path.parent / f".{name}.tmp"
        destination = self.index_path.parent / name
        if temporary.exists() or destination.exists():
            raise FileExistsError(
                f"Refusing to reuse immutable safetensors generation path {destination}."
            )
        self._save_file(self._current, temporary)
        temporary.replace(destination)
        descriptors: list[dict[str, Any]] = []
        for record, key, dtype_name, shape, digest in self._pending:
            descriptors.append(
                {
                    "id": record.id,
                    "sequence": record.sequence,
                    "tensor": {
                        "file": name,
                        "key": key,
                        "dtype": dtype_name,
                        "shape": list(shape),
                        "sha256": digest,
                    },
                }
            )
        descriptor_name = (
            f"{self._prefix}-records-run-{self._generation}-{self._shard_index:05d}.jsonl"
        )
        self._descriptor_shards.append(
            self._write_descriptor_file(descriptor_name, descriptors, tensor_file=name)
        )
        self._record_count += len(descriptors)
        self._current = {}
        self._pending = []
        self._current_size = 0

    def append(
        self,
        records: Iterable[EmbeddingRecord],
        *,
        publish: bool | None = None,
    ) -> None:
        """Persist records while retaining at most one shard of tensors."""

        for record in records:
            position = self._record_count + len(self._pending)
            tensor = record.load_tensor().detach().cpu().contiguous()  # (...)
            if tensor.dtype not in _DTYPE_NAMES:
                raise TypeError(f"Unsupported tensor dtype {tensor.dtype}.")
            nbytes = tensor.numel() * tensor.element_size()
            if nbytes > self.shard_size:
                raise ValueError(
                    f"Embedding {position} requires {nbytes} bytes and cannot fit in a "
                    f"{self.shard_size}-byte safetensors shard."
                )
            if self._current and (
                self._current_size + nbytes > self.shard_size
                or len(self._pending) == _MAX_RECORDS_PER_DESCRIPTOR_SHARD
            ):
                self._write_shard()
                if self.publish_incremental:
                    self._publish_metadata(complete=False)
                position = self._record_count
            key = f"embedding_{position:08d}"
            self._current[key] = tensor
            self._current_size += nbytes
            self._pending.append(
                (
                    record,
                    key,
                    _DTYPE_NAMES[tensor.dtype],
                    tuple(tensor.shape),
                    tensor_sha256(tensor),
                )
            )
        if publish:
            self.publish(complete=False)

    def _publish_metadata(
        self,
        *,
        complete: bool,
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        """Atomically expose one self-consistent metadata generation."""

        if metadata is not None:
            self.metadata = _persistent_metadata(
                metadata,
                descriptor_index="safetensors-generation-index",
            )
        self.metadata["complete"] = complete
        self.metadata["record_count"] = self._record_count
        self._commit_index += 1
        payload = {
            "version": 2,
            "format": "fastplms-embedding-safetensors",
            "metadata": self.metadata,
            "record_count": self._record_count,
            "descriptor_shards": self._descriptor_shards,
        }
        generation_index_name = (
            f"{self._prefix}-index-run-{self._generation}-{self._commit_index:05d}.json"
        )
        generation_index_path = self.index_path.parent / generation_index_name
        temporary_generation_index = generation_index_path.with_name(
            f".{generation_index_path.name}.tmp"
        )
        if temporary_generation_index.exists() or generation_index_path.exists():
            raise FileExistsError(
                f"Refusing to reuse immutable safetensors generation index {generation_index_path}."
            )
        encoded_index = _canonical_json_bytes(payload)
        temporary_generation_index.write_bytes(encoded_index)
        temporary_generation_index.replace(generation_index_path)

        index_sha256 = hashlib.sha256(encoded_index).hexdigest()
        index_reference = {
            "file": generation_index_name,
            "sha256": index_sha256,
        }
        run_manifest = {
            "version": 2,
            "format": "fastplms-embedding-run",
            "index": index_reference,
            "record_count": self._record_count,
        }
        pointer_identity = f"{self._generation}-{self._commit_index:05d}"
        temporary_manifest = self.run_manifest_path.with_name(
            f".{self.run_manifest_path.name}.{pointer_identity}.tmp"
        )
        temporary_manifest.write_bytes(_canonical_json_bytes(run_manifest))
        temporary_manifest.replace(self.run_manifest_path)

        # ``index.json`` is a non-authoritative convenience pointer. The run
        # manifest is committed first, so interruption here cannot invalidate
        # the newly committed generation.
        stable_pointer = {
            "version": 2,
            "format": "fastplms-embedding-index-pointer",
            "index": index_reference,
        }
        temporary_index = self.index_path.with_name(
            f".{self.index_path.name}.{pointer_identity}.tmp"
        )
        temporary_index.write_bytes(_canonical_json_bytes(stable_pointer))
        temporary_index.replace(self.index_path)

        return load_safetensors_result(self.index_path)

    def publish(
        self,
        *,
        complete: bool,
        metadata: dict[str, Any] | None = None,
    ) -> EmbeddingResult:
        """Flush the current shard and atomically expose a consistent generation."""

        self._write_shard()
        return self._publish_metadata(complete=complete, metadata=metadata)


def save_safetensors_result(
    result: EmbeddingResult,
    path: str | Path,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> EmbeddingResult:
    """Write sharded safetensors without materializing the full result."""

    writer = SafetensorsStreamWriter(
        path,
        result.metadata,
        shard_size=shard_size,
        publish_initial=False,
        publish_incremental=False,
    )
    writer.append(result, publish=False)
    return writer.publish(complete=bool(result.metadata.get("complete", True)))


def load_safetensors_result(path: str | Path) -> EmbeddingResult:
    """Load an indexed safetensors result without loading tensor payloads."""

    payload, index_path, _ = _load_authoritative_index(path)
    if payload.get("version") == 2:
        lazy_records = _SafetensorsRecordSequence(
            index_path.parent, payload.get("descriptor_shards", ())
        )
        if len(lazy_records) != payload.get("record_count"):
            raise ValueError("Safetensors descriptor count does not match its generation index.")
        return EmbeddingResult(lazy_records, payload.get("metadata", {}))

    records: list[EmbeddingRecord] = []
    for item in payload["records"]:
        records.append(_record_from_safetensors_descriptor(index_path.parent, item))
    return EmbeddingResult(records, payload.get("metadata", {}))


def garbage_collect_safetensors_generations(
    path: str | Path,
    *,
    dry_run: bool = True,
    confirm_no_active_readers_or_writers: bool = False,
) -> tuple[Path, ...]:
    """Remove non-authoritative generations after an explicit exclusivity check.

    Safetensors results retain immutable historical generations because an
    already-open :class:`EmbeddingResult` resolves tensors through those exact
    descriptor and shard paths. Destructive collection is therefore safe only
    when the caller guarantees that no reader or writer for ``path`` remains
    active. ``dry_run=True`` is the default and returns the paths that would be
    removed without changing the output directory.
    """

    if not isinstance(dry_run, bool):
        raise TypeError("dry_run must be a bool.")
    if not isinstance(confirm_no_active_readers_or_writers, bool):
        raise TypeError("confirm_no_active_readers_or_writers must be a bool.")
    if not dry_run and not confirm_no_active_readers_or_writers:
        raise ValueError(
            "Destructive safetensors generation collection requires "
            "confirm_no_active_readers_or_writers=True."
        )

    # Validate the full descriptor graph before identifying anything as stale.
    load_safetensors_result(path)
    payload, authoritative_index_path, _ = _load_authoritative_index(path)
    stable_index_path = _index_path(path)
    run_manifest_path = _run_manifest_path(path)
    root = stable_index_path.parent
    prefix = _safetensors_shard_prefix(path)
    protected = {
        stable_index_path.resolve(),
        run_manifest_path.resolve(),
        authoritative_index_path.resolve(),
        *_referenced_shards(stable_index_path, payload),
    }
    for descriptor_shard in payload.get("descriptor_shards", ()):
        relative = descriptor_shard.get("file")
        if isinstance(relative, str):
            protected.add(_resolve_index_child(root, relative, label="index").resolve())

    candidates: set[Path] = set()
    for pattern in (
        f"{prefix}-run-*-*.safetensors",
        f"{prefix}-records-run-*.jsonl",
        f"{prefix}-index-run-*.json",
        f".{prefix}-*.tmp",
    ):
        candidates.update(root.glob(pattern))
    candidates.update(root.glob(f".{stable_index_path.name}.*.tmp"))
    candidates.update(root.glob(f".{run_manifest_path.name}.*.tmp"))

    stale = tuple(
        sorted(
            (candidate for candidate in candidates if candidate.resolve() not in protected),
            key=lambda candidate: candidate.name,
        )
    )
    if not dry_run:
        for candidate in stale:
            candidate.unlink(missing_ok=True)
    return stale


def _ensure_sqlite_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            published_order INTEGER
        );
        CREATE TABLE IF NOT EXISTS tensors (
            run_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            dtype TEXT NOT NULL,
            shape_json TEXT NOT NULL,
            data BLOB NOT NULL,
            sha256 TEXT NOT NULL,
            PRIMARY KEY (run_id, position),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS records (
            run_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            record_id TEXT NOT NULL,
            sequence TEXT NOT NULL,
            PRIMARY KEY (run_id, position),
            FOREIGN KEY (run_id, position) REFERENCES tensors(run_id, position)
                ON DELETE CASCADE
        );
        """
    )
    run_columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(runs)").fetchall()}
    if "published_order" not in run_columns:
        connection.execute("ALTER TABLE runs ADD COLUMN published_order INTEGER")
        # Databases created before staged publication exposed every stored run.
        # Preserve that view for historical runs containing committed records.
        connection.execute(
            "UPDATE runs SET published_order = rowid "
            "WHERE published_order IS NULL AND EXISTS ("
            "SELECT 1 FROM records WHERE records.run_id = runs.run_id)"
        )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS runs_published_order_idx ON runs(published_order)"
    )
    if "published_order" not in run_columns:
        # Schema upgrades run before callers open their data transaction.
        # End the migration transaction explicitly so BEGIN IMMEDIATE below
        # remains valid on existing databases.
        connection.commit()


def save_sqlite_result(result: EmbeddingResult, path: str | Path) -> EmbeddingResult:
    """Transactionally store an ordered result in normalized SQLite tables."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(result.metadata.get("run_fingerprint", ""))
    if not run_id:
        raise ValueError("SQLite results require metadata['run_fingerprint'].")
    metadata_json = json.dumps(
        _persistent_metadata(
            result.metadata,
            descriptor_index="sqlite-records",
            record_count=len(result),
        ),
        sort_keys=True,
    )
    with sqlite3.connect(path, timeout=30) as connection:
        _ensure_sqlite_schema(connection)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        connection.execute(
            "INSERT INTO runs(run_id, metadata_json, published_order) "
            "SELECT ?, ?, COALESCE(MAX(published_order), 0) + 1 FROM runs",
            (run_id, metadata_json),
        )
        for position, record in enumerate(result):
            X = record.load_tensor().detach().cpu().contiguous()  # (...)
            dtype_name, shape_json, data = _encode_tensor(X)
            digest = tensor_sha256(X)
            connection.execute(
                "INSERT INTO tensors VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, position, dtype_name, shape_json, data, digest),
            )
            connection.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?)",
                (run_id, position, record.id, record.sequence),
            )
        connection.commit()
    return load_sqlite_result(path, run_id=run_id)


def initialize_sqlite_run(
    path: str | Path,
    metadata: dict[str, Any],
    *,
    resume: bool,
) -> str:
    """Create a resumable SQLite run without buffering tensor results."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(metadata.get("run_fingerprint", ""))
    if not run_id:
        raise ValueError("SQLite runs require metadata['run_fingerprint'].")
    with sqlite3.connect(path, timeout=30) as connection:
        _ensure_sqlite_schema(connection)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE")
        exists = connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if exists and not resume:
            connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            exists = None
        if exists is None:
            initial_metadata = _persistent_metadata(
                metadata,
                descriptor_index="sqlite-records",
                record_count=0,
            )
            connection.execute(
                "INSERT INTO runs(run_id, metadata_json) VALUES (?, ?)",
                (run_id, json.dumps(initial_metadata, sort_keys=True)),
            )
        connection.commit()
    return run_id


def append_sqlite_records(
    path: str | Path,
    run_id: str,
    start_position: int,
    records: list[EmbeddingRecord],
    *,
    replace_metadata: dict[str, Any] | None = None,
) -> None:
    """Commit one ordered embedding batch so an interrupted run can resume."""

    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string.")
    if not isinstance(start_position, int) or isinstance(start_position, bool):
        raise TypeError("start_position must be a non-negative integer.")
    if start_position < 0:
        raise ValueError("start_position must be a non-negative integer.")
    if not isinstance(records, list) or not all(
        isinstance(record, EmbeddingRecord) for record in records
    ):
        raise TypeError("records must be a list of EmbeddingRecord values.")

    with sqlite3.connect(Path(path), timeout=30) as connection:
        _ensure_sqlite_schema(connection)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE")
        if replace_metadata is not None:
            replacement_run_id = str(replace_metadata.get("run_fingerprint", ""))
            if replacement_run_id != run_id:
                raise ValueError("Replacement metadata must match the SQLite run ID.")
            initial_metadata = _persistent_metadata(
                replace_metadata,
                descriptor_index="sqlite-records",
                record_count=0,
            )
            connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            connection.execute(
                "INSERT INTO runs(run_id, metadata_json) VALUES (?, ?)",
                (run_id, json.dumps(initial_metadata, sort_keys=True)),
            )
        if connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
            raise KeyError(f"Missing SQLite embedding run {run_id}.")
        current_count, minimum_position, maximum_position = connection.execute(
            "SELECT COUNT(*), MIN(position), MAX(position) FROM records WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if current_count and (minimum_position != 0 or maximum_position != current_count - 1):
            raise ValueError("SQLite embedding run has a non-contiguous record prefix.")
        if start_position != current_count:
            raise ValueError(
                f"start_position={start_position} does not match the contiguous "
                f"SQLite prefix length {current_count}."
            )
        for offset, record in enumerate(records):
            position = start_position + offset
            X = record.load_tensor().detach().cpu().contiguous()  # (...)
            dtype_name, shape_json, data = _encode_tensor(X)
            digest = tensor_sha256(X)
            connection.execute(
                "INSERT INTO tensors VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, position, dtype_name, shape_json, data, digest),
            )
            connection.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?)",
                (run_id, position, record.id, record.sequence),
            )
        row = connection.execute(
            "SELECT metadata_json FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Missing SQLite embedding run {run_id}.")
        metadata = json.loads(row[0])
        if not isinstance(metadata, dict):
            raise ValueError("SQLite run metadata must contain a JSON object.")
        metadata["record_count"] = start_position + len(records)
        metadata["descriptor_index"] = "sqlite-records"
        connection.execute(
            "UPDATE runs SET metadata_json = ? WHERE run_id = ?",
            (json.dumps(metadata, sort_keys=True), run_id),
        )
        if records:
            connection.execute(
                "UPDATE runs SET published_order = ("
                "SELECT COALESCE(MAX(published_order), 0) + 1 FROM runs"
                ") WHERE run_id = ? AND published_order IS NULL",
                (run_id,),
            )
        connection.commit()


def update_sqlite_run_metadata(path: str | Path, run_id: str, metadata: dict[str, Any]) -> None:
    """Finalize reproducibility metadata after the last streamed batch."""

    with sqlite3.connect(Path(path), timeout=30) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM records WHERE run_id = ?", (run_id,)
        ).fetchone()
        record_count = int(row[0]) if row is not None else 0
        cleaned_metadata = _persistent_metadata(
            metadata,
            descriptor_index="sqlite-records",
            record_count=record_count,
        )
        updated = connection.execute(
            "UPDATE runs SET metadata_json = ? WHERE run_id = ?",
            (json.dumps(cleaned_metadata, sort_keys=True), run_id),
        ).rowcount
        if updated != 1:
            raise KeyError(f"Missing SQLite embedding run {run_id}.")
        connection.commit()


def _connect_sqlite_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=30)


def _validate_sqlite_result_schema(connection: sqlite3.Connection, path: Path) -> None:
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    required = {"runs", "records", "tensors"}
    if not required.issubset(tables):
        raise ValueError(
            f"Not a FastPLMs embedding SQLite database: {path}. "
            "Use convert_legacy_sqlite() for a legacy embeddings table."
        )


def _load_sqlite_tensor(path: Path, run_id: str, position: int) -> Tensor:
    with _connect_sqlite_read_only(path) as connection:
        row = connection.execute(
            "SELECT dtype, shape_json, data FROM tensors WHERE run_id = ? AND position = ?",
            (run_id, position),
        ).fetchone()
    if row is None:
        raise KeyError(f"Missing SQLite tensor {run_id}:{position}.")
    return _decode_tensor(*row)


def _validate_sqlite_descriptor_row(
    row: Sequence[Any],
) -> tuple[int, str, str, str, str, str]:
    if len(row) != 6:
        raise ValueError("SQLite embedding descriptor has an invalid column count.")
    position, record_id, sequence, dtype_name, shape_json, digest = row
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        raise ValueError("SQLite embedding position is invalid.")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("SQLite embedding record ID is invalid.")
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("SQLite embedding sequence is invalid.")
    if not isinstance(shape_json, str):
        raise ValueError("SQLite embedding tensor shape is invalid.")
    try:
        shape = json.loads(shape_json)
    except json.JSONDecodeError as error:
        raise ValueError("SQLite embedding tensor shape is invalid.") from error
    _validate_tensor_descriptor(
        {
            "key": f"embedding_{position}",
            "dtype": dtype_name,
            "shape": shape,
            "sha256": digest,
        }
    )
    return position, record_id, sequence, dtype_name, shape_json, digest


def _sqlite_record_from_row(path: Path, run_id: str, row: Sequence[Any]) -> EmbeddingRecord:
    position, record_id, sequence, dtype_name, shape_json, digest = _validate_sqlite_descriptor_row(
        row
    )

    def load_tensor() -> Tensor:
        return _load_sqlite_tensor(path, run_id, position)

    reference = LazyTensorReference(
        source=str(path),
        key=f"{run_id}:{position}",
        dtype=dtype_name,
        shape=tuple(json.loads(shape_json)),
        sha256=digest,
        _loader=load_tensor,
    )
    return EmbeddingRecord(record_id, sequence, reference)


class _SQLiteRecordSequence(Sequence[EmbeddingRecord]):
    """Lazy immutable descriptor view over one SQLite embedding run."""

    _fastplms_immutable_sequence = True

    def __init__(self, path: Path, run_id: str, count: int) -> None:
        self.path = path
        self.run_id = run_id
        self._count = count

    @staticmethod
    def _row_query() -> str:
        return (
            "SELECT r.position, r.record_id, r.sequence, t.dtype, t.shape_json, t.sha256 "
            "FROM records r JOIN tensors t USING (run_id, position) "
            "WHERE r.run_id = ?"
        )

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[EmbeddingRecord]:
        with _connect_sqlite_read_only(self.path) as connection:
            cursor = connection.execute(f"{self._row_query()} ORDER BY r.position", (self.run_id,))
            while rows := cursor.fetchmany(1_024):
                for row in rows:
                    yield _sqlite_record_from_row(self.path, self.run_id, row)

    @overload
    def __getitem__(self, index: int, /) -> EmbeddingRecord: ...

    @overload
    def __getitem__(self, index: slice, /) -> Sequence[EmbeddingRecord]: ...

    def __getitem__(self, index: int | slice) -> EmbeddingRecord | Sequence[EmbeddingRecord]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return [self[position] for position in range(start, stop, step)]
        position = index + self._count if index < 0 else index
        if position < 0 or position >= self._count:
            raise IndexError(index)
        with _connect_sqlite_read_only(self.path) as connection:
            row = connection.execute(
                f"{self._row_query()} AND r.position = ?",
                (self.run_id, position),
            ).fetchone()
        if row is None:
            raise IndexError(index)
        return _sqlite_record_from_row(self.path, self.run_id, row)


def load_sqlite_result(
    path: str | Path,
    *,
    run_id: str | None = None,
    positions: Iterable[int] | None = None,
    record_ids: Iterable[str] | None = None,
    sequences: Iterable[str] | None = None,
) -> EmbeddingResult:
    """Load one SQLite run read-only, optionally in explicit selector order.

    Exactly one selector may be supplied. Repeated selectors are retained. An
    ID or sequence selector that matches multiple stored rows returns those
    rows in their original order for every occurrence of that selector.
    """

    path = Path(path).resolve()
    supplied_selectors = sum(
        selector is not None for selector in (positions, record_ids, sequences)
    )
    if supplied_selectors > 1:
        raise ValueError("Choose at most one of positions, record_ids, or sequences.")
    normalized_positions = tuple(positions) if positions is not None else None
    normalized_ids = tuple(record_ids) if record_ids is not None else None
    normalized_sequences = tuple(sequences) if sequences is not None else None
    if normalized_positions is not None and not all(
        isinstance(position, int) and not isinstance(position, bool) and position >= 0
        for position in normalized_positions
    ):
        raise ValueError("positions must contain non-negative integers.")
    for name, values in (
        ("record_ids", normalized_ids),
        ("sequences", normalized_sequences),
    ):
        if values is not None and not all(isinstance(value, str) for value in values):
            raise TypeError(f"{name} must contain strings.")

    with _connect_sqlite_read_only(path) as connection:
        _validate_sqlite_result_schema(connection, path)
        if run_id is None:
            run_columns = {
                str(info[1]) for info in connection.execute("PRAGMA table_info(runs)").fetchall()
            }
            if "published_order" in run_columns:
                row = connection.execute(
                    "SELECT run_id, metadata_json FROM runs "
                    "WHERE published_order IS NOT NULL "
                    "ORDER BY published_order DESC, rowid DESC LIMIT 1"
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT run_id, metadata_json FROM runs "
                    "ORDER BY created_at DESC, rowid DESC LIMIT 1"
                ).fetchone()
        else:
            row = connection.execute(
                "SELECT run_id, metadata_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"No embedding run found in {path}.")
        selected_run, metadata_json = row
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError("SQLite run metadata must contain a JSON object.")
        row_prefix = (
            "SELECT r.position, r.record_id, r.sequence, t.dtype, t.shape_json, t.sha256 "
            "FROM records r JOIN tensors t USING (run_id, position) "
            "WHERE r.run_id = ?"
        )
        record_count, minimum_position, maximum_position = connection.execute(
            "SELECT COUNT(*), MIN(position), MAX(position) FROM records WHERE run_id = ?",
            (selected_run,),
        ).fetchone()
        (tensor_count,) = connection.execute(
            "SELECT COUNT(*) FROM tensors WHERE run_id = ?", (selected_run,)
        ).fetchone()
        (joined_count,) = connection.execute(
            "SELECT COUNT(*) FROM records r JOIN tensors t USING (run_id, position) "
            "WHERE r.run_id = ?",
            (selected_run,),
        ).fetchone()
        if (
            tensor_count != record_count
            or joined_count != record_count
            or (record_count and (minimum_position != 0 or maximum_position != record_count - 1))
        ):
            raise ValueError("SQLite embedding run has inconsistent or non-contiguous records.")
        metadata_count = metadata.get("record_count")
        if (
            not isinstance(metadata_count, int)
            or isinstance(metadata_count, bool)
            or metadata_count != record_count
        ):
            raise ValueError("SQLite metadata record count does not match stored records.")
        descriptor_cursor = connection.execute(f"{row_prefix} ORDER BY r.position", (selected_run,))
        while descriptor_rows := descriptor_cursor.fetchmany(1_024):
            for descriptor_row in descriptor_rows:
                _validate_sqlite_descriptor_row(descriptor_row)
        if supplied_selectors == 0:
            rows: list[tuple[Any, ...]] | None = None
        else:
            selector_values: tuple[Any, ...]
            selector_column: str
            if normalized_positions is not None:
                selector_values = normalized_positions
                selector_column = "r.position"
            elif normalized_ids is not None:
                selector_values = normalized_ids
                selector_column = "r.record_id"
            else:
                if normalized_sequences is None:
                    raise RuntimeError("Filtered SQLite retrieval resolved no selector values.")
                selector_values = normalized_sequences
                selector_column = "r.sequence"
            fetched: list[tuple[Any, ...]] = []
            unique_values = tuple(dict.fromkeys(selector_values))
            for start in range(0, len(unique_values), 900):
                chunk = unique_values[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                fetched.extend(
                    connection.execute(
                        f"{row_prefix} AND {selector_column} IN ({placeholders}) "
                        "ORDER BY r.position",
                        (selected_run, *chunk),
                    ).fetchall()
                )
            value_index = (
                0 if normalized_positions is not None else (1 if normalized_ids is not None else 2)
            )
            matched: dict[Any, list[tuple[Any, ...]]] = {}
            for fetched_row in sorted(fetched, key=lambda item: int(item[0])):
                matched.setdefault(fetched_row[value_index], []).append(fetched_row)
            missing = [value for value in selector_values if value not in matched]
            if missing:
                raise KeyError(f"SQLite embedding selectors were not found: {missing!r}.")
            rows = [
                fetched_row for value in selector_values for fetched_row in matched.get(value, ())
            ]

    if rows is None:
        return EmbeddingResult(
            _SQLiteRecordSequence(path, selected_run, int(record_count)),
            metadata,
        )
    records = [_sqlite_record_from_row(path, selected_run, selected_row) for selected_row in rows]
    if supplied_selectors:
        metadata = dict(metadata)
        metadata["selection"] = {
            "kind": (
                "positions"
                if normalized_positions is not None
                else "record_ids"
                if normalized_ids is not None
                else "sequences"
            ),
            "count": len(rows),
            "duplicate_policy": "preserve-request-order",
        }
    return EmbeddingResult(records, metadata)


def load_legacy_pth(path: str | Path, *, allow_unsafe_pickle: bool = False) -> EmbeddingResult:
    """Import a legacy mapping-only ``.pth`` file after explicit opt-in."""

    if not allow_unsafe_pickle:
        raise ValueError(
            "Legacy .pth loading can execute pickle payloads. Pass "
            "allow_unsafe_pickle=True only for a trusted file."
        )
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("A legacy .pth embedding file must contain a mapping.")
    records: list[EmbeddingRecord] = []
    for position, (sequence, X) in enumerate(payload.items()):
        # X: (...)
        if not isinstance(sequence, str) or not isinstance(X, Tensor):
            raise ValueError("Legacy embedding mappings must use str keys and Tensor values.")
        records.append(EmbeddingRecord(str(position), sequence, X.detach().cpu()))
    return EmbeddingResult(records, {"format": "legacy-pth", "unsafe_pickle": True})


_LEGACY_COMPACT_VERSION = 0x01
_LEGACY_CODE_DTYPES: dict[int, tuple[np.dtype[Any], torch.dtype]] = {
    0: (np.dtype(np.float16), torch.float16),
    # Legacy BF16 blobs stored FP16 payload bytes and converted back to BF16.
    1: (np.dtype(np.float16), torch.bfloat16),
    2: (np.dtype(np.float32), torch.float32),
}


def _decode_legacy_sqlite_blob(
    data: bytes,
    *,
    fallback_shape: tuple[int, ...] | None,
    allow_unsafe_pickle: bool,
) -> Tensor:
    if len(data) >= 6 and data[0] == _LEGACY_COMPACT_VERSION:
        dtype_code = int(data[1])
        if dtype_code not in _LEGACY_CODE_DTYPES:
            raise ValueError(f"Unsupported legacy compact dtype code {dtype_code}.")
        (ndim,) = struct.unpack_from("<i", data, 2)
        if ndim < 0 or ndim > 16 or len(data) < 6 + 4 * ndim:
            raise ValueError("Malformed legacy compact embedding header.")
        shape = tuple(int(value) for value in struct.unpack_from(f"<{ndim}i", data, 6))
        if any(size < 0 for size in shape):
            raise ValueError("Malformed negative legacy embedding dimension.")
        numpy_dtype, target_dtype = _LEGACY_CODE_DTYPES[dtype_code]
        offset = 6 + 4 * ndim
        expected = int(np.prod(shape, dtype=np.int64)) * numpy_dtype.itemsize
        if len(data) - offset != expected:
            raise ValueError("Legacy compact embedding payload length does not match shape.")
        array = (  # shape
            np.frombuffer(data, dtype=numpy_dtype, offset=offset).copy().reshape(shape)
        )
        return torch.from_numpy(array).to(dtype=target_dtype)  # shape

    try:
        loaded = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    except Exception as safe_error:
        if allow_unsafe_pickle:
            loaded = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        elif fallback_shape is None:
            raise ValueError(
                "Legacy embedding blob is neither compact nor safely loadable. "
                "Provide fallback_shape for raw FP32 bytes, or set "
                "allow_unsafe_pickle=True only for a trusted database."
            ) from safe_error
        else:
            expected = int(np.prod(fallback_shape, dtype=np.int64)) * 4
            if len(data) != expected:
                raise ValueError(
                    "Legacy raw FP32 payload length does not match fallback_shape."
                ) from safe_error
            array = np.frombuffer(data, dtype=np.float32).copy().reshape(  # fallback_shape
                fallback_shape
            )
            return torch.from_numpy(array)  # fallback_shape
    if not isinstance(loaded, Tensor):
        raise ValueError("Legacy serialized embedding payload must contain one tensor.")
    return loaded.detach().cpu()  # (...)


def convert_legacy_sqlite(
    source: str | Path,
    output: str | Path,
    *,
    fallback_shape: tuple[int, ...] | None = None,
    allow_unsafe_pickle: bool = False,
    metadata: dict[str, Any] | None = None,
) -> EmbeddingResult:
    """Convert the v0 ``embeddings(sequence, embedding)`` database safely.

    The source is opened read-only. Compact blobs and ``weights_only`` Torch
    tensors are accepted by default. Unsafe general pickle deserialization
    remains an explicit opt-in.
    """

    source_path = Path(source)
    output_path = Path(output)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Legacy SQLite conversion requires a different output path.")
    if fallback_shape is not None and (
        not fallback_shape or any(not isinstance(size, int) or size < 0 for size in fallback_shape)
    ):
        raise ValueError("fallback_shape must contain non-negative integer dimensions.")
    with _connect_sqlite_read_only(source_path) as connection:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        if not {"sequence", "embedding"}.issubset(columns):
            raise ValueError("Legacy SQLite database must contain embeddings(sequence, embedding).")
        rows = connection.execute(
            "SELECT sequence, embedding FROM embeddings ORDER BY rowid"
        ).fetchall()
    if not rows:
        raise ValueError("Legacy SQLite database contains no embeddings.")

    records: list[EmbeddingRecord] = []
    content_digest = hashlib.sha256()
    for position, (sequence, data) in enumerate(rows):
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("Legacy embedding sequences must be non-empty strings.")
        if not isinstance(data, bytes):
            data = bytes(data)
        tensor = _decode_legacy_sqlite_blob(
            data,
            fallback_shape=fallback_shape,
            allow_unsafe_pickle=allow_unsafe_pickle,
        )
        tensor_digest = tensor_sha256(tensor)
        for value in (sequence.encode("utf-8"), tensor_digest.encode("ascii")):
            content_digest.update(len(value).to_bytes(8, "big"))
            content_digest.update(value)
        records.append(EmbeddingRecord(str(position), sequence, tensor))

    content_sha256 = content_digest.hexdigest()
    run_fingerprint = hashlib.sha256(
        f"fastplms-legacy-sqlite-v1:{content_sha256}".encode("ascii")
    ).hexdigest()
    converted_metadata: dict[str, Any] = {
        "format_version": 1,
        "run_fingerprint": run_fingerprint,
        "source_format": "legacy-fastplms-sqlite-v0",
        "source_content_sha256": content_sha256,
        "unsafe_pickle": allow_unsafe_pickle,
        "complete": True,
    }
    if metadata:
        converted_metadata["conversion_metadata"] = _jsonable(metadata)
    return save_sqlite_result(
        EmbeddingResult(records, converted_metadata),
        output_path,
    )


def save_result(
    result: EmbeddingResult,
    path: str | Path,
    *,
    format: str = "safetensors",
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> EmbeddingResult:
    if format == "safetensors":
        return save_safetensors_result(result, path, shard_size=shard_size)
    if format == "sqlite":
        return save_sqlite_result(result, path)
    if format == "pth":
        raise ValueError("Writing pickle-based .pth embeddings is not supported.")
    raise ValueError("format must be 'safetensors' or 'sqlite'.")


def load_result(path: str | Path, *, format: str = "safetensors") -> EmbeddingResult:
    if format == "safetensors":
        return load_safetensors_result(path)
    if format == "sqlite":
        return load_sqlite_result(path)
    raise ValueError("format must be 'safetensors' or 'sqlite'.")


__all__ = [
    "DEFAULT_SHARD_SIZE",
    "SafetensorsStreamWriter",
    "append_sqlite_records",
    "convert_legacy_sqlite",
    "garbage_collect_safetensors_generations",
    "initialize_sqlite_run",
    "load_legacy_pth",
    "load_result",
    "load_safetensors_result",
    "load_sqlite_result",
    "safetensors_result_exists",
    "save_result",
    "save_safetensors_result",
    "save_sqlite_result",
    "tensor_sha256",
    "update_sqlite_run_metadata",
]
