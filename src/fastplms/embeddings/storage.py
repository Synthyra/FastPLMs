"""Lossless, reproducible storage for :mod:`fastplms.embeddings`."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch
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


def _tensor_bytes(X: Tensor) -> bytes:
    """Return the exact contiguous byte representation of X."""

    X = X.detach().cpu().contiguous()
    return X.view(torch.uint8).numpy().tobytes()


def tensor_sha256(X: Tensor) -> str:
    """Hash dtype, shape, and exact tensor bytes."""

    if X.dtype not in _DTYPE_NAMES:
        raise TypeError(f"Unsupported tensor dtype {X.dtype}.")
    digest = hashlib.sha256()
    digest.update(_DTYPE_NAMES[X.dtype].encode())
    digest.update(json.dumps(tuple(X.shape)).encode())
    digest.update(_tensor_bytes(X))
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
    byte_array = np.frombuffer(data, dtype=np.uint8).copy()
    X = torch.from_numpy(byte_array).view(dtype)
    return X.reshape(shape).clone()


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


def safetensors_result_exists(path: str | Path) -> bool:
    """Return whether both transactional safetensors metadata files exist."""

    return _index_path(path).is_file() and _run_manifest_path(path).is_file()


def _load_safetensor(path: Path, key: str) -> Tensor:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ImportError("Loading embeddings requires the 'safetensors' package.") from error
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def save_safetensors_result(
    result: EmbeddingResult,
    path: str | Path,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> EmbeddingResult:
    """Write sharded safetensors, an index, and a validated run manifest."""

    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise ImportError("Saving embeddings requires the 'safetensors' package.") from error
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")

    index_path = _index_path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [record.load_tensor() for record in result]
    shards: list[dict[str, Tensor]] = []
    current: dict[str, Tensor] = {}
    current_size = 0
    locations: list[tuple[int, str, str, tuple[int, ...], str]] = []

    for position, X in enumerate(materialized):
        X = X.detach().cpu().contiguous()
        key = f"embedding_{position:08d}"
        nbytes = X.numel() * X.element_size()
        if nbytes > shard_size:
            raise ValueError(
                f"Embedding {position} requires {nbytes} bytes and cannot fit in a "
                f"{shard_size}-byte safetensors shard."
            )
        if current and current_size + nbytes > shard_size:
            shards.append(current)
            current = {}
            current_size = 0
        shard_index = len(shards)
        current[key] = X
        current_size += nbytes
        locations.append(
            (shard_index, key, _DTYPE_NAMES[X.dtype], tuple(X.shape), tensor_sha256(X))
        )
    if current:
        shards.append(current)

    shard_names = [
        f"embeddings-{i + 1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))
    ]
    for name, tensors in zip(shard_names, shards, strict=True):
        temporary = index_path.parent / f".{name}.tmp"
        save_file(tensors, temporary)
        temporary.replace(index_path.parent / name)

    records_json = []
    for record, location in zip(result, locations, strict=True):
        shard_index, key, dtype_name, shape, digest = location
        records_json.append(
            {
                "id": record.id,
                "sequence": record.sequence,
                "tensor": {
                    "file": shard_names[shard_index],
                    "key": key,
                    "dtype": dtype_name,
                    "shape": list(shape),
                    "sha256": digest,
                },
            }
        )
    metadata = _jsonable(result.metadata)
    payload = {
        "version": 1,
        "format": "fastplms-embedding-safetensors",
        "metadata": metadata,
        "records": records_json,
    }
    temporary_index = index_path.with_name(f".{index_path.name}.tmp")
    temporary_index.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_index.replace(index_path)

    index_sha256 = hashlib.sha256(index_path.read_bytes()).hexdigest()
    run_manifest_path = _run_manifest_path(path)
    run_manifest = {
        "version": 1,
        "format": "fastplms-embedding-run",
        "index": {"file": index_path.name, "sha256": index_sha256},
        "metadata": metadata,
        "record_count": len(records_json),
    }
    temporary_manifest = run_manifest_path.with_name(f".{run_manifest_path.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(run_manifest_path)
    return load_safetensors_result(index_path)


def load_safetensors_result(path: str | Path) -> EmbeddingResult:
    """Load an indexed safetensors result without loading tensor payloads."""

    index_path = _index_path(path)
    run_manifest_path = _run_manifest_path(path)
    if not run_manifest_path.is_file():
        raise ValueError(f"Missing safetensors run manifest: {run_manifest_path}.")
    index_bytes = index_path.read_bytes()
    payload = json.loads(index_bytes.decode("utf-8"))
    if payload.get("format") != "fastplms-embedding-safetensors":
        raise ValueError(f"Not a FastPLMs embedding index: {index_path}.")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected_index = {
        "file": index_path.name,
        "sha256": hashlib.sha256(index_bytes).hexdigest(),
    }
    if run_manifest.get("format") != "fastplms-embedding-run":
        raise ValueError(f"Not a FastPLMs embedding run manifest: {run_manifest_path}.")
    if run_manifest.get("version") != 1 or run_manifest.get("index") != expected_index:
        raise ValueError("Safetensors run manifest does not match its index.")
    if run_manifest.get("metadata") != payload.get("metadata"):
        raise ValueError("Safetensors run manifest metadata does not match its index.")
    if run_manifest.get("record_count") != len(payload.get("records", ())):
        raise ValueError("Safetensors run manifest record count does not match its index.")
    records: list[EmbeddingRecord] = []
    for item in payload["records"]:
        tensor = item["tensor"]
        tensor_path = index_path.parent / tensor["file"]
        reference = LazyTensorReference(
            source=str(tensor_path),
            key=tensor["key"],
            dtype=tensor["dtype"],
            shape=tuple(tensor["shape"]),
            sha256=tensor["sha256"],
            _loader=lambda p=tensor_path, k=tensor["key"]: _load_safetensor(p, k),
        )
        records.append(EmbeddingRecord(item["id"], item["sequence"], reference))
    return EmbeddingResult(records, payload.get("metadata", {}))


def _ensure_sqlite_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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


def save_sqlite_result(result: EmbeddingResult, path: str | Path) -> EmbeddingResult:
    """Transactionally store an ordered result in normalized SQLite tables."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(result.metadata.get("run_fingerprint", ""))
    if not run_id:
        raise ValueError("SQLite results require metadata['run_fingerprint'].")
    metadata_json = json.dumps(_jsonable(result.metadata), sort_keys=True)
    with sqlite3.connect(path, timeout=30) as connection:
        _ensure_sqlite_schema(connection)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        connection.execute(
            "INSERT INTO runs(run_id, metadata_json) VALUES (?, ?)",
            (run_id, metadata_json),
        )
        for position, record in enumerate(result):
            X = record.load_tensor().detach().cpu().contiguous()
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
            connection.execute(
                "INSERT INTO runs(run_id, metadata_json) VALUES (?, ?)",
                (run_id, json.dumps(_jsonable(metadata), sort_keys=True)),
            )
        connection.commit()
    return run_id


def append_sqlite_records(
    path: str | Path,
    run_id: str,
    start_position: int,
    records: list[EmbeddingRecord],
) -> None:
    """Commit one ordered embedding batch so an interrupted run can resume."""

    with sqlite3.connect(Path(path), timeout=30) as connection:
        _ensure_sqlite_schema(connection)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE")
        for offset, record in enumerate(records):
            position = start_position + offset
            X = record.load_tensor().detach().cpu().contiguous()
            dtype_name, shape_json, data = _encode_tensor(X)
            digest = tensor_sha256(X)
            connection.execute(
                "DELETE FROM tensors WHERE run_id = ? AND position = ?",
                (run_id, position),
            )
            connection.execute(
                "INSERT INTO tensors VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, position, dtype_name, shape_json, data, digest),
            )
            connection.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?)",
                (run_id, position, record.id, record.sequence),
            )
        connection.commit()


def update_sqlite_run_metadata(path: str | Path, run_id: str, metadata: dict[str, Any]) -> None:
    """Finalize reproducibility metadata after the last streamed batch."""

    with sqlite3.connect(Path(path), timeout=30) as connection:
        updated = connection.execute(
            "UPDATE runs SET metadata_json = ? WHERE run_id = ?",
            (json.dumps(_jsonable(metadata), sort_keys=True), run_id),
        ).rowcount
        if updated != 1:
            raise KeyError(f"Missing SQLite embedding run {run_id}.")
        connection.commit()


def _load_sqlite_tensor(path: Path, run_id: str, position: int) -> Tensor:
    with sqlite3.connect(path, timeout=30) as connection:
        row = connection.execute(
            "SELECT dtype, shape_json, data FROM tensors WHERE run_id = ? AND position = ?",
            (run_id, position),
        ).fetchone()
    if row is None:
        raise KeyError(f"Missing SQLite tensor {run_id}:{position}.")
    return _decode_tensor(*row)


def load_sqlite_result(path: str | Path, *, run_id: str | None = None) -> EmbeddingResult:
    """Load one SQLite run with lazy, lossless tensor references."""

    path = Path(path)
    with sqlite3.connect(path, timeout=30) as connection:
        _ensure_sqlite_schema(connection)
        if run_id is None:
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
        rows = connection.execute(
            "SELECT r.position, r.record_id, r.sequence, t.dtype, t.shape_json, t.sha256 "
            "FROM records r JOIN tensors t USING (run_id, position) "
            "WHERE r.run_id = ? ORDER BY r.position",
            (selected_run,),
        ).fetchall()

    records: list[EmbeddingRecord] = []
    for position, record_id, sequence, dtype_name, shape_json, digest in rows:
        reference = LazyTensorReference(
            source=str(path),
            key=f"{selected_run}:{position}",
            dtype=dtype_name,
            shape=tuple(json.loads(shape_json)),
            sha256=digest,
            _loader=lambda p=path, r=selected_run, i=position: _load_sqlite_tensor(p, r, i),
        )
        records.append(EmbeddingRecord(record_id, sequence, reference))
    return EmbeddingResult(records, json.loads(metadata_json))


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
        if not isinstance(sequence, str) or not isinstance(X, Tensor):
            raise ValueError("Legacy embedding mappings must use str keys and Tensor values.")
        records.append(EmbeddingRecord(str(position), sequence, X.detach().cpu()))
    return EmbeddingResult(records, {"format": "legacy-pth", "unsafe_pickle": True})


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
    "append_sqlite_records",
    "initialize_sqlite_run",
    "load_legacy_pth",
    "load_result",
    "load_safetensors_result",
    "load_sqlite_result",
    "save_result",
    "save_safetensors_result",
    "save_sqlite_result",
    "safetensors_result_exists",
    "tensor_sha256",
    "update_sqlite_run_metadata",
]
