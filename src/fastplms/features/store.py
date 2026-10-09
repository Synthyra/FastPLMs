"""The FastPLMs feature store: one directory per feature key, read by sequence.

A store answers one question: what is this feature of this sequence, and does it already exist?
That makes a run embed only what is missing, and lets one embedding pass serve every head trained
afterwards.

Layout on disk, under one root that holds many features::

    <root>/<key>/
        feature.json                       the key's descriptor, layout, width, and dtype
        index.sqlite                       sequence sha256 -> segment, part, row, residue count
        segments/<fingerprint>/
            part-00000.safetensors         immutable, memory-mappable, one per append
            part-00001.safetensors
            run.json                       the commit marker, written last

**A segment is immutable and committed once.** Parts appear as a run streams, and nothing reads
them until ``run.json`` names them; a run that dies leaves a directory the index ignores and
``sweep`` removes. Nothing is ever rewritten, so a reader never sees a half-written feature and two
runs never race over one file.

**The index is a cache of the commit markers, not the record.** Every indexed row can be rebuilt
from the committed segments, which ``reindex`` does, so a lost or corrupt ``index.sqlite`` costs a
scan rather than the features.

**A sequence is identified by the SHA-256 of its exact UTF-8 bytes**, so two callers agree without
coordinating, and a sequence that differs by one residue is a different row. The store keeps the
digest and the residue count, never the sequence text: a caller that has the sequences can always
recompute the digest, and storing millions of them again would cost more than the features.

The key belongs to ``foundry.embedding.feature_key``, which composes the model, its revision, the
sparse autoencoder, the layer, the pooling, the dtype, and the residue limit into one filename-safe
name. This module never invents a key; it stores the name and the descriptor it is given and
refuses a second, different descriptor under the same name. FastPLMs ships without foundry, so the
store takes the name as a string rather than importing the key.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
import threading
import torch

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, cast
from torch import Tensor

from .digests import file_sha256, json_sha256
from .json_files import indented_json
from .layouts import (
    CSR,
    DENSE,
    LAYOUT_NAMES,
    RAGGED,
    RAGGED_TOPK,
    SparseRow,
    TopKRow,
    dtype_name,
    encode_csr,
    encode_dense,
    encode_ragged,
    encode_topk_rows,
    row_count,
    row_tensor_bytes,
    tensor_names,
    validate_topk,
    value_dtype,
)
from .transactions import file_lock, flush_and_evict, publish_file, sync_directory


FORMAT = "fastplms-feature-store-v1"
FEATURE_FILE = "feature.json"
INDEX_FILE = "index.sqlite"
SEGMENTS_DIRECTORY = "segments"
COMMIT_FILE = "run.json"
PART_TEMPLATE = "part-{:05d}.safetensors"

# Descriptor schemas that carry a complete scientific contract: v1 keeps residues only, v2 keeps CLS and EOS too,
# and v3 keeps v2's rows computed under a pinned embedding profile.
COMPLETE_SCHEMAS = frozenset({"feature_spec_v1", "feature_spec_v2", "feature_spec_v3"})

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_PART = re.compile(r"part-(\d{5})\.safetensors")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_WRITER_LEASE = object()


def sequence_digest(sequence: str) -> str:
    """The SHA-256 of a sequence's exact UTF-8 bytes, which is its row identity."""

    if not isinstance(sequence, str) or not sequence:
        raise ValueError("A sequence must be a non-empty string.")
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def partition_sequences(sequences: Iterable[str], *, shard: int, shards: int) -> tuple[str, ...]:
    """Assign exact unique sequences by SHA-256 modulo shard count, preserving input order.

    Every worker must receive the same input inventory and shard count. Assignment does not
    depend on Python's randomized hash or process/device identity.
    """
    if type(shards) is not int or shards < 1 or type(shard) is not int or not 0 <= shard < shards:
        raise ValueError("Require a positive shard count and 0 <= shard < shards.")
    unique = dict.fromkeys(sequences)
    return tuple(
        sequence for sequence in unique if int(sequence_digest(sequence), 16) % shards == shard
    )


@dataclass(frozen=True, slots=True)
class StoredFeature:
    """What one feature is: its key, how its rows are laid out, and what they hold.

    ``width`` is the vector width for ``dense``, the codebook size for ``csr``, and the hidden
    width for ``ragged``. For ``ragged_topk``, ``width`` is the codebook size and ``sparse_count``
    is the number of retained codes per residue. ``positions`` says whether ``csr`` rows carry
    each entry's argmax residue. ``descriptor`` is the plain data the key was computed from,
    so a store can say what it holds without the caller that made it.
    """

    key: str
    layout: str
    width: int
    dtype: torch.dtype
    positions: bool = False
    descriptor: Mapping[str, Any] = field(default_factory=dict)
    sparse_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not _NAME.fullmatch(self.key):
            raise ValueError(
                "A feature key must be a filename-safe name, as "
                "foundry.embedding.feature_key(...).name returns; received "
                f"{self.key!r}."
            )
        if self.layout not in LAYOUT_NAMES:
            raise ValueError(
                f"layout must be one of {list(LAYOUT_NAMES)}; received {self.layout!r}."
            )
        if not isinstance(self.width, int) or isinstance(self.width, bool) or self.width <= 0:
            raise ValueError(f"width must be a positive integer; received {self.width!r}.")
        dtype_name(self.dtype)
        if self.positions and self.layout != CSR:
            raise ValueError("Only a csr feature stores argmax positions.")
        if self.layout == RAGGED_TOPK:
            if type(self.sparse_count) is not int or not 1 <= self.sparse_count <= self.width:
                raise ValueError(
                    "A ragged top-k feature requires integer sparse_count in 1..width."
                )
            if self.width > torch.iinfo(torch.int32).max + 1:
                raise ValueError("Top-k codebook exceeds the stored int32 index range.")
        elif self.sparse_count is not None:
            raise ValueError("Only a ragged top-k feature declares sparse_count.")
        if not isinstance(self.descriptor, Mapping):
            raise TypeError("descriptor must be a mapping of plain data.")
        object.__setattr__(self, "descriptor", json.loads(json.dumps(dict(self.descriptor))))

    def payload(self) -> dict[str, Any]:
        """``feature.json``'s content."""

        payload = {
            "format": FORMAT,
            "key": self.key,
            "layout": self.layout,
            "width": self.width,
            "dtype": dtype_name(self.dtype),
            "positions": self.positions,
            "descriptor": dict(self.descriptor),
        }
        if self.sparse_count is not None:
            payload["sparse_count"] = self.sparse_count
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StoredFeature:
        """The spec a ``feature.json`` describes."""

        if payload.get("format") != FORMAT:
            raise ValueError(
                f"Not a {FORMAT} feature directory; its format is {payload.get('format')!r}."
            )
        return cls(
            key=str(payload["key"]),
            layout=str(payload["layout"]),
            width=int(payload["width"]),
            dtype=value_dtype(str(payload["dtype"])),
            positions=bool(payload.get("positions", False)),
            descriptor=payload.get("descriptor") or {},
            sparse_count=payload.get("sparse_count"),
        )


@dataclass(frozen=True, slots=True)
class RowAddress:
    """Where one sequence's row sits: which segment, which part, which row of it."""

    segment: str
    part: int
    row: int
    residues: int


@dataclass(frozen=True, slots=True)
class SegmentReceipt:
    """What one committed segment holds, as ``run.json`` records it."""

    fingerprint: str
    parts: tuple[int, ...]
    rows: int
    committed_at: str
    metadata: Mapping[str, Any]


class FeatureStore:
    """One feature's directory, opened for reading and appending."""

    def __init__(
        self, directory: Path, spec: StoredFeature, *, read_only: bool = False,
        content_pins: Mapping[str, str] | None = None, deep_verify: bool = True,
        partial: bool = False,
    ) -> None:
        self.directory = directory.resolve()
        self.spec = spec
        self._read_only = read_only
        # Deep verification re-reads every committed part whenever the index is recovered, which opening
        # and every commit do. A store of terabytes opens with deep_verify=False: only segments the index
        # lacks are read and indexed, and each part is still verified when a reader first touches it.
        self._deep_verify = deep_verify
        # A partial copy holds some of each segment's committed parts, as a fetch of a few rows from a
        # published store leaves it. Its index lists the rows of the parts present, so a row of an absent
        # part reads as missing, a part fetched later is indexed on the next open, and it takes no new
        # segment. Absent parts cannot be verified, so it opens without deep verification.
        if partial and deep_verify:
            raise ValueError("A partial copy opens with deep_verify=False; its absent parts cannot be read.")
        self._partial = partial
        self._content_pins = None if content_pins is None else dict(content_pins)
        if self._content_pins is not None:
            if not read_only or FEATURE_FILE not in self._content_pins:
                raise ValueError("Pinned feature handles must be read-only and pin feature.json.")
            for relative, digest in self._content_pins.items():
                components = relative.split("/") if isinstance(relative, str) else []
                valid_path = relative == FEATURE_FILE or (
                    len(components) == 3 and components[0] == SEGMENTS_DIRECTORY
                    and _NAME.fullmatch(components[1])
                    and (components[2] == COMMIT_FILE or re.fullmatch(
                        r"part-\d{5,}\.(safetensors|rows\.json\.gz)", components[2],
                    ))
                )
                if not valid_path or not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                    raise ValueError(
                        "Feature content pins must name immutable store files and SHA-256 digests."
                    )

    # Opening -----------------------------------------------------------------

    @classmethod
    def open(
        cls, root: str | Path, spec: StoredFeature, *, deep_verify: bool = True, partial: bool = False,
    ) -> FeatureStore:
        """Open the store for ``spec`` under ``root``, creating it when it does not exist.

        An existing directory must describe exactly this spec. A mismatch is a caller using one key
        for two different features, which would serve one project another project's numbers, so it
        raises rather than migrating. ``deep_verify=False`` is for stores too large to re-read on
        every open and commit, and ``partial=True`` for a copy holding only some committed parts,
        which must already exist: see the constructor.
        """

        directory = Path(root) / spec.key
        recorded = directory / FEATURE_FILE
        if partial and not recorded.exists():
            raise FileNotFoundError(f"{directory} holds no {FEATURE_FILE}, so it is no copy of a store.")
        store = cls(directory, spec, deep_verify=deep_verify, partial=partial)
        with store._write_lock():
            if recorded.exists():
                existing = StoredFeature.from_payload(
                    json.loads(recorded.read_text(encoding="utf-8")))
                if existing != spec:
                    raise ValueError(
                        f"{directory} already holds a different feature under key {spec.key!r}.\n"
                        f"  stored:    {existing.payload()}\n"
                        f"  requested: {spec.payload()}"
                    )
            else:
                (directory / SEGMENTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
                _write_json_atomically(recorded, spec.payload())
            store._ensure_index_locked()
        return store

    @classmethod
    def read_only(
        cls, directory: str | Path, *, content_pins: Mapping[str, str] | None = None,
    ) -> FeatureStore:
        """Open an existing descriptor without creating or repairing any file.

        Each query opens its own SQLite connection in read-only mode and closes it before
        returning. A missing or corrupt index fails when queried, without being repaired.
        """

        path = Path(directory)
        payload = json.loads((path / FEATURE_FILE).read_text(encoding="utf-8"))
        store = cls(
            path, StoredFeature.from_payload(payload), read_only=True, content_pins=content_pins,
        )
        store._check_pin(FEATURE_FILE)
        return store

    # Membership --------------------------------------------------------------

    def __len__(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT count(*) FROM rows").fetchone()[0])

    def address(self, sequence: str) -> RowAddress | None:
        """Where this sequence's row is, or None when the store lacks it."""

        with self._connect() as connection:
            found = connection.execute(
                "SELECT segment, part, row, residues FROM rows WHERE digest = ?",
                (sequence_digest(sequence),),
            ).fetchone()
        return None if found is None else RowAddress(*found)

    def missing(self, sequences: Iterable[str]) -> tuple[str, ...]:
        """The sequences this store lacks, in the order given, without repeats.

        This is the call that makes a run embed only what is new.
        """

        wanted: dict[str, str] = {}
        for sequence in sequences:
            wanted.setdefault(sequence_digest(sequence), sequence)
        if not wanted:
            return ()
        present = self.present_digests(list(wanted))
        return tuple(sequence for digest, sequence in wanted.items() if digest not in present)

    def present_digests(self, digests: Sequence[str]) -> set[str]:
        """The digests among ``digests`` that this store already holds, from one index connection."""

        present: set[str] = set()
        with self._connect() as connection:
            for chunk in _chunks(list(digests), 512):
                marks = ",".join("?" * len(chunk))
                present.update(
                    row[0]
                    for row in connection.execute(
                        f"SELECT digest FROM rows WHERE digest IN ({marks})", chunk
                    )
                )
        return present

    def segments(self) -> tuple[SegmentReceipt, ...]:
        """Every committed segment, oldest commit first, then by fingerprint.

        Commit times have one-second resolution, so the fingerprint breaks the tie and the order is
        total.
        """

        receipts = [_receipt(payload) for payload in self._verified_segments()]
        return tuple(sorted(receipts, key=lambda receipt: (receipt.committed_at, receipt.fingerprint)))

    # Writing -----------------------------------------------------------------

    @contextmanager
    def segment(
        self, fingerprint: str, metadata: Mapping[str, Any] | None = None,
        *, before_commit: Callable[[], None] | None = None, verify_staged: bool = True,
    ) -> Iterator[SegmentWriter]:
        """Open a new segment for one embedding run, committed on a clean exit.

        ``verify_staged`` makes the commit re-read and re-hash every staged file. A caller that wrote
        large parts through ``append_packed`` and hashed them as it wrote may pass False: the commit
        then compares each file's size and modification time with what the write recorded, which
        catches an appended or rewritten file without reading the payload a second time.

        ``fingerprint`` identifies the run that produced these rows, and is FastPLMs' own run
        fingerprint in a real run. A committed segment of that name already existing is an error:
        the same run producing the same rows twice means one of the two is not what it claims.

        A body that writes nothing leaves nothing behind, and a body that raises leaves the segment
        uncommitted for ``sweep``.
        """

        self._require_writable()
        if self._partial:
            raise PermissionError("A partial copy takes no new segment; embed into a store of its own.")
        if not isinstance(fingerprint, str) or not _NAME.fullmatch(fingerprint):
            raise ValueError(f"A segment fingerprint must be a filename-safe name; received {fingerprint!r}.")
        if self.spec.descriptor.get("schema") in COMPLETE_SCHEMAS and before_commit is None:
            raise ValueError(
                "A complete feature contract requires a pre-commit validation callback."
            )
        path = self.directory / SEGMENTS_DIRECTORY / fingerprint
        with file_lock(self._segment_lock_path(fingerprint), wait=False):
            with self._write_lock():
                if (path / COMMIT_FILE).exists():
                    raise FileExistsError(
                        f"Segment {fingerprint!r} is already committed in {self.directory}."
                    )
                # Only the owner of this segment lock can discard a crashed attempt.
                if path.exists():
                    _discard_segment(path)
                path.mkdir(parents=True)
                sync_directory(path.parent)
            writer = SegmentWriter(
                self, path, fingerprint, dict(metadata or {}), before_commit, lease=_WRITER_LEASE,
                verify_staged=verify_staged,
            )
            try:
                yield writer
                if not writer.committed and not writer.closed:
                    if writer.parts:
                        writer.commit()
                    else:
                        writer.abandon()
            finally:
                writer.closed = True

    def sweep(self) -> tuple[str, ...]:
        """Delete every uncommitted segment directory, and name what was deleted.

        An uncommitted segment is the remains of a run that died. Nothing reads it.
        """

        self._require_writable()
        removed: list[str] = []
        for path in sorted((self.directory / SEGMENTS_DIRECTORY).iterdir()):
            if not path.is_dir() or not _NAME.fullmatch(path.name):
                continue
            with ExitStack() as stack:
                try:
                    stack.enter_context(file_lock(self._segment_lock_path(path.name), wait=False))
                except BlockingIOError:
                    continue
                with self._write_lock():
                    if path.exists() and not (path / COMMIT_FILE).exists():
                        _discard_segment(path)
                        removed.append(path.name)
        return tuple(removed)

    def reindex(self) -> int:
        """Rebuild the index from the committed segments, and return the row count.

        The index is a cache, so this is the repair when it is lost or doubted. It reads each
        part's row count from its tensors and each part's digests from the commit marker.
        """

        self._require_writable()
        with self._write_lock():
            self._ensure_index_locked(repair=True)
        return len(self)

    # Reading -----------------------------------------------------------------

    def content_pins(self, sequences: Sequence[str]) -> dict[str, str]:
        """Pin selected immutable files, excluding the rebuildable index.

        Adding an unrelated committed segment does not invalidate this selection. A pinned
        reader checks the requested rows' actual index associations, markers and part bytes.
        """
        payload = json.loads((self.directory / FEATURE_FILE).read_text(encoding="utf-8"))
        recorded = StoredFeature.from_payload(payload)
        if recorded != self.spec:
            raise ValueError("Feature descriptor changed while pinning a selection.")
        pins = {FEATURE_FILE: file_sha256(self.directory / FEATURE_FILE)}
        markers = {}
        for _, address, _ in self._located(sequences, load=False):
            prefix = f"{SEGMENTS_DIRECTORY}/{address.segment}"
            if address.segment not in markers:
                markers[address.segment] = self._segment_payload(address.segment)
                marker = self.directory / prefix / COMMIT_FILE
                pins[f"{prefix}/{COMMIT_FILE}"] = file_sha256(marker)
            part = markers[address.segment]["parts"][address.part]
            pins[f"{prefix}/{PART_TEMPLATE.format(address.part)}"] = part["sha256"]
            if part.get("row_metadata") is not None:
                pins[f"{prefix}/{part['row_metadata']['file']}"] = part["row_metadata"]["sha256"]
        for relative, expected in pins.items():
            actual = file_sha256(self.directory / relative)
            if actual != expected:
                raise ValueError("Feature bytes changed while pinning a selection.")
            self._check_pin(relative, actual)
        return pins

    def read(self, sequences: Sequence[str]) -> list[Tensor]:
        """Each sequence's feature as a tensor, in the order asked.

        A dense feature gives ``(w,)``, a ragged one ``(r_i, d)``, and a csr one a densified
        ``(w,)`` row; ``read_sparse`` keeps a csr row compressed. ``ragged_topk`` requires
        ``read_topk`` to keep residue codes sparse. A sequence the store lacks raises,
        because a silent zero row is indistinguishable from a real one.
        """

        if self.spec.layout == CSR:
            return [row.to_dense(self.spec.width) for row in self.read_sparse(sequences)]  # (w,) each
        if self.spec.layout == RAGGED_TOPK:
            raise ValueError(
                "Use read_topk for sparse residue codes; implicit densification is refused."
            )
        rows: list[Tensor] = []
        for sequence, address, tensors in self._located(sequences):
            # tensors["values"]: (n, w) dense or (sum r_i, d) ragged, for the n rows of one part
            if self.spec.layout == DENSE:
                rows.append(tensors["values"][address.row].clone())  # (w,)
            else:
                offsets = tensors["offsets"]  # (n + 1,)
                start, stop = int(offsets[address.row]), int(offsets[address.row + 1])
                rows.append(tensors["values"][start:stop].clone())  # (r_i, d)
            del sequence
        return rows  # (w,) or (r_i, d) per sequence

    def read_sparse(self, sequences: Sequence[str]) -> list[SparseRow]:
        """Each sequence's compressed row, in the order asked. Only for a csr feature."""

        if self.spec.layout != CSR:
            raise ValueError(f"read_sparse needs a csr feature; this one is {self.spec.layout}.")
        rows: list[SparseRow] = []
        for _, address, tensors in self._located(sequences):
            indptr = tensors["indptr"]  # (n + 1,)
            start, stop = int(indptr[address.row]), int(indptr[address.row + 1])
            positions = tensors.get("positions")  # (nnz,) or None, for the part's nnz entries
            rows.append(
                SparseRow(
                    indices=tensors["indices"][start:stop].clone(),  # (nnz_i,)
                    values=tensors["values"][start:stop].clone(),  # (nnz_i,)
                    positions=None if positions is None else positions[start:stop].clone(),  # (nnz_i,)
                )
            )
        return rows

    def read_topk(self, sequences: Sequence[str]) -> list[TopKRow]:
        """Return sparse ``(residues,k)`` codes in request order, including duplicate requests."""
        if self.spec.layout != RAGGED_TOPK:
            raise ValueError(
                f"read_topk needs a ragged_topk feature; this one is {self.spec.layout}."
            )
        rows = []
        for _, address, tensors in self._located(sequences):
            offsets = tensors["offsets"]  # (n + 1,)
            start, stop = int(offsets[address.row]), int(offsets[address.row + 1])
            rows.append(TopKRow(
                tensors["indices"][start:stop].clone(),  # (r_i, k)
                tensors["values"][start:stop].clone(),  # (r_i, k)
            ))
        return rows

    def residue_counts(self, sequences: Sequence[str]) -> list[int]:
        """Each sequence's stored residue count, which a length-bucketed reader batches by."""

        return [address.residues for _, address, _ in self._located(sequences, load=False)]

    def row_metadata(
        self, sequences: Sequence[str], *, verify_data: bool = True,
    ) -> list[dict[str, Any]]:
        """Read opaque row identities in request order and verify committed file digests.

        The contract provider owns the scientific schema. This layer verifies the physical
        association between sequence, index address, commit marker, data and metadata file.
        Legacy rows without identities raise rather than becoming canonical cache hits.
        """
        markers: dict[str, dict[str, Any]] = {}
        parts: dict[tuple[str, int], list[dict[str, Any]]] = {}
        identities = []
        for sequence, address, _ in self._located(sequences, load=False):
            if not _NAME.fullmatch(address.segment) or address.part < 0:
                raise ValueError("Invalid committed feature address.")
            if address.segment not in markers:
                marker = self.directory / SEGMENTS_DIRECTORY / address.segment / COMMIT_FILE
                payload = json.loads(marker.read_text(encoding="utf-8"))
                if payload["key"] != self.spec.key or payload["fingerprint"] != address.segment:
                    raise ValueError("Committed feature identity does not match the index.")
                markers[address.segment] = payload
            payload = markers[address.segment]
            matching = [part for part in payload["parts"] if part["part"] == address.part]
            if len(matching) != 1:
                raise ValueError("Index part does not name one committed feature part.")
            part = matching[0]
            if (not 0 <= address.row < len(part["digests"])
                    or part["digests"][address.row] != sequence_digest(sequence)
                    or part["residues"][address.row] != address.residues):
                raise ValueError("Feature index and committed sequence row disagree.")
            where = (address.segment, address.part)
            if where not in parts:
                identity = part.get("row_metadata")
                if not isinstance(identity, dict):
                    raise ValueError(
                        "Stored row identities are unavailable; recompute this legacy feature."
                    )
                filename = f"part-{address.part:05d}.rows.json.gz"
                if identity.get("file") != filename:
                    raise ValueError("Invalid row metadata filename.")
                path = self.directory / SEGMENTS_DIRECTORY / address.segment / filename
                if file_sha256(path) != identity.get("sha256"):
                    raise ValueError("Stored row metadata digest does not match its commit marker.")
                if verify_data and file_sha256(self._part_path(*where)) != part.get("sha256"):
                    raise ValueError("Stored feature data digest does not match its commit marker.")
                rows = json.loads(gzip.decompress(path.read_bytes()))
                if (not isinstance(rows, list) or len(rows) != len(part["digests"])
                        or not all(isinstance(row, dict) for row in rows)):
                    raise ValueError("Stored row metadata does not match the committed part rows.")
                parts[where] = rows
            identities.append(parts[where][address.row])
        return identities

    # Internals ---------------------------------------------------------------

    def _located(
        self, sequences: Sequence[str], *, load: bool = True
    ) -> list[tuple[str, RowAddress, dict[str, Tensor]]]:
        """Resolve each sequence to its address, reading each part file at most once."""

        self._check_pin(FEATURE_FILE)
        addresses = list(zip(sequences, self._addresses(sequences), strict=True))
        cache: dict[tuple[str, int], dict[str, Tensor]] = {}
        markers: dict[str, dict[str, Any]] = {}
        located: list[tuple[str, RowAddress, dict[str, Tensor]]] = []
        for sequence, address in addresses:
            if address.segment not in markers:
                markers[address.segment] = self._segment_payload(address.segment)
            payload = markers[address.segment]
            if not 0 <= address.part < len(payload["parts"]):
                raise ValueError("Index part does not name one committed feature part.")
            part = payload["parts"][address.part]
            if (not 0 <= address.row < len(part["digests"])
                    or part["digests"][address.row] != sequence_digest(sequence)
                    or part["residues"][address.row] != address.residues):
                raise ValueError("Feature index and committed sequence row disagree.")
            where = (address.segment, address.part)
            if where not in cache:
                tensors = self._verified_part(address.segment, part)
                cache[where] = tensors if load else {}
            tensors = cache[where]
            located.append((sequence, address, tensors))
        return located  # (sequence, address, part tensors) per sequence; the tensors are those of _verified_part

    def _addresses(self, sequences: Sequence[str]) -> list[RowAddress]:
        """Each sequence's address in the order given, from one index connection.

        Opening the index once per sequence cost minutes for a run of a hundred thousand sequences on a
        network volume. The first sequence the store lacks raises, as ``read`` always has.
        """

        digests = [sequence_digest(sequence) for sequence in sequences]
        found: dict[str, RowAddress] = {}
        with self._connect() as connection:
            for chunk in _chunks(list(dict.fromkeys(digests)), 512):
                marks = ",".join("?" * len(chunk))
                for digest, *where in connection.execute(
                    f"SELECT digest, segment, part, row, residues FROM rows WHERE digest IN ({marks})",
                    chunk,
                ):
                    found[digest] = RowAddress(*where)
        addresses: list[RowAddress] = []
        for sequence, digest in zip(sequences, digests, strict=True):
            address = found.get(digest)
            if address is None:
                raise KeyError(
                    f"Feature {self.spec.key!r} has no row for a sequence of "
                    f"{len(sequence)} residues (sha256 {digest[:12]}). "
                    "Embed it first, or call missing() before reading."
                )
            addresses.append(address)
        return addresses

    def _segment_payload(self, fingerprint: str) -> dict[str, Any]:
        """Validate a commit marker before using any filename or row it supplies."""
        if not isinstance(fingerprint, str) or not _NAME.fullmatch(fingerprint):
            raise ValueError("Invalid committed segment fingerprint.")
        marker = self.directory / SEGMENTS_DIRECTORY / fingerprint / COMMIT_FILE
        encoded = marker.read_bytes()
        self._check_pin(
            f"{SEGMENTS_DIRECTORY}/{fingerprint}/{COMMIT_FILE}",
            hashlib.sha256(encoded).hexdigest(),
        )
        payload = json.loads(encoded.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("A feature commit marker must be an object.")
        if (payload.get("format") != FORMAT or payload.get("key") != self.spec.key
                or payload.get("fingerprint") != fingerprint):
            raise ValueError("Committed feature identity does not match its directory.")
        if any(key in payload for key in (
            "transaction_schema", "manifest_sha256", "descriptor_sha256",
        )):
            if (type(payload.get("transaction_schema")) is not int
                    or payload["transaction_schema"] != 2):
                raise ValueError("Unsupported feature transaction schema.")
            unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
            if payload.get("manifest_sha256") != json_sha256(unsigned, allow_nan=False):
                raise ValueError("Feature commit marker checksum mismatch.")
            if payload.get("descriptor_sha256") != json_sha256(self.spec.payload(), allow_nan=False):
                raise ValueError("Feature descriptor does not match its committed digest.")
        parts = payload.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("A committed segment must contain parts.")
        seen: set[str] = set()
        for number, part in enumerate(parts):
            if (not isinstance(part, dict) or type(part.get("part")) is not int
                    or part["part"] != number):
                raise ValueError("Committed part numbers must be consecutive and unique.")
            digests, residues = part.get("digests"), part.get("residues")
            if (not isinstance(digests, list) or not digests or not isinstance(residues, list)
                    or len(digests) != len(residues)):
                raise ValueError("Committed sequence and residue counts disagree.")
            if any(not isinstance(value, str) or not _SHA256.fullmatch(value) for value in digests):
                raise ValueError("Invalid committed sequence digest.")
            if len(set(digests)) != len(digests) or seen.intersection(digests):
                raise ValueError("A committed segment repeats sequence rows.")
            seen.update(digests)
            if any(type(value) is not int or value < 0 for value in residues):
                raise ValueError("Invalid committed residue count.")
            if not isinstance(part.get("sha256"), str) or not _SHA256.fullmatch(part["sha256"]):
                raise ValueError("Committed data checksum missing; recompute this legacy segment.")
        if type(payload.get("rows")) is not int or payload["rows"] != len(seen):
            raise ValueError("Committed segment row count disagrees with its parts.")
        if (not isinstance(payload.get("committed_at"), str)
                or not isinstance(payload.get("metadata"), dict)):
            raise ValueError("Invalid committed segment metadata.")
        return payload

    def _verified_part(self, fingerprint: str, part: Mapping[str, Any]) -> dict[str, Tensor]:
        """Check immutable bytes, physical layout, and optional row sidecars."""
        number = part["part"]
        data_digest = file_sha256(self._part_path(fingerprint, number))
        if data_digest != part["sha256"]:
            raise ValueError("Stored feature data digest does not match its commit marker.")
        self._check_pin(
            f"{SEGMENTS_DIRECTORY}/{fingerprint}/{PART_TEMPLATE.format(number)}", data_digest,
        )
        tensors = self._load_part(fingerprint, number)
        # The part holds count rows. Dense: values (count, w). Otherwise offsets or indptr (count + 1,), with
        # values (sum r_i, d) ragged, (sum r_i, k) top-k plus indices of the same shape, or (nnz,) csr.
        spec, count = self.spec, len(part["digests"])
        if set(tensors) != set(tensor_names(spec.layout, positions=spec.positions)):
            raise ValueError("Committed tensor names do not match the feature layout.")
        values = tensors["values"]  # (count, w) dense, (sum r_i, d) ragged, (sum r_i, k) top-k, (nnz,) csr
        if values.dtype != spec.dtype:
            raise ValueError("Committed tensor dtype does not match the feature.")
        if spec.layout == DENSE:
            if values.shape != (count, spec.width) or any(part["residues"]):
                raise ValueError("Committed dense shape or residue counts disagree.")
        else:
            offsets = tensors["indptr" if spec.layout == CSR else "offsets"]  # (count + 1,)
            if (offsets.dtype != torch.int64 or offsets.shape != (count + 1,)
                    or int(offsets[0]) != 0 or int(offsets[-1]) != len(values)
                    or bool((offsets[1:] < offsets[:-1]).any())):
                raise ValueError("Committed row offsets are invalid.")
            if spec.layout == RAGGED:
                if (values.ndim != 2 or values.shape[1] != spec.width
                        or (offsets[1:] - offsets[:-1]).tolist() != part["residues"]):
                    raise ValueError("Committed ragged shape or residue counts disagree.")
            elif spec.layout == RAGGED_TOPK:
                indices = tensors["indices"]  # (sum r_i, k)
                if (indices.dtype != torch.int32
                        or (offsets[1:] - offsets[:-1]).tolist() != part["residues"]):
                    raise ValueError("Committed top-k index dtype or residue counts disagree.")
                validate_topk(indices, values, spec.width, cast(int, spec.sparse_count))
            else:
                indices = tensors["indices"]  # (nnz,)
                if (values.ndim != 1 or indices.dtype != torch.int32
                        or indices.shape != values.shape or any(part["residues"])
                        or bool(((indices < 0) | (indices >= spec.width)).any())):
                    raise ValueError("Committed sparse shape or indices are invalid.")
                for start, stop in pairwise(offsets):
                    codes = indices[int(start):int(stop)]  # (nnz_i,)
                    if len(torch.unique(codes)) != len(codes):
                        raise ValueError("Committed sparse row repeats indices.")
                if spec.positions:
                    positions = tensors["positions"]  # (nnz,)
                    if (positions.dtype != torch.int16 or positions.shape != values.shape
                            or bool((positions < 0).any())):
                        raise ValueError("Committed sparse positions are invalid.")
        if "tensor_bytes" in part and part["tensor_bytes"] != sum(
            value.numel() * value.element_size() for value in tensors.values()
        ):
            raise ValueError("Committed tensor byte count disagrees with its payload.")
        identity = part.get("row_metadata")
        if identity is None and spec.descriptor.get("schema") in COMPLETE_SCHEMAS:
            raise ValueError("Complete feature contracts require committed row identities.")
        if identity is not None:
            filename = f"part-{number:05d}.rows.json.gz"
            if not isinstance(identity, dict) or identity.get("file") != filename:
                raise ValueError("Invalid row metadata filename.")
            path = self.directory / SEGMENTS_DIRECTORY / fingerprint / filename
            metadata_digest = file_sha256(path)
            if metadata_digest != identity.get("sha256"):
                raise ValueError("Stored row metadata digest does not match its commit marker.")
            self._check_pin(f"{SEGMENTS_DIRECTORY}/{fingerprint}/{filename}", metadata_digest)
            rows = json.loads(gzip.decompress(path.read_bytes()))
            if (not isinstance(rows, list) or len(rows) != count
                    or not all(isinstance(row, dict) for row in rows)):
                raise ValueError("Stored row metadata does not match the committed part rows.")
        return tensors  # (count, w) values for dense; offsets or indptr (count + 1,) with values as above otherwise

    def _check_pin(self, relative: str, actual: str | None = None) -> None:
        """Check independent selection pins, in addition to a marker's internal checksums."""
        if self._content_pins is not None:
            if relative not in self._content_pins:
                raise ValueError(
                    f"Requested feature file is outside the pinned selection: {relative}."
                )
            actual = file_sha256(self.directory / relative) if actual is None else actual
            if actual != self._content_pins[relative]:
                raise ValueError(
                    f"Feature file differs from its independent content pin: {relative}."
                )

    def _verified_segments(self) -> list[dict[str, Any]]:
        payloads, seen = [], set()
        for marker in sorted((self.directory / SEGMENTS_DIRECTORY).glob(f"*/{COMMIT_FILE}")):
            payload = self._segment_payload(marker.parent.name)
            for part in payload["parts"]:
                if seen.intersection(part["digests"]):
                    raise ValueError("Committed segments contain conflicting sequence rows.")
                seen.update(part["digests"])
                self._verified_part(payload["fingerprint"], part)
            payloads.append(payload)
        return payloads

    def _write_lock(self) -> AbstractContextManager[None]:
        self._require_writable()
        return file_lock(self.directory / ".locks" / "store.lock")

    def _segment_lock_path(self, fingerprint: str) -> Path:
        # 128 bits keep distinct segments on distinct locks, and a short name keeps the path under
        # Windows' 260-character limit, which a store in a nested directory would otherwise exceed.
        name = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]
        return self.directory / ".locks" / f"segment-{name}.lock"

    def _load_part(self, segment: str, part: int) -> dict[str, Tensor]:
        path = self._part_path(segment, part)
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise ImportError("Reading a feature store requires the 'safetensors' package.") from error
        with safe_open(path, framework="pt", device="cpu") as handle:
            # `safe_open` is a handle with keys(), not a mapping: iterating it directly does not work.
            return {name: cast(Tensor, handle.get_tensor(name)) for name in handle.keys()}  # noqa: dict-idiom  # (...) each, as stored

    def _part_path(self, segment: str, part: int) -> Path:
        return self.directory / SEGMENTS_DIRECTORY / segment / PART_TEMPLATE.format(part)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection that is always closed.

        `sqlite3.Connection` as a context manager ends the transaction but leaves the handle open,
        which on Windows keeps the index file locked against the next writer.
        """

        database = self.directory / INDEX_FILE
        connection = (
            sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
            if self._read_only else sqlite3.connect(database)
        )
        try:
            yield connection
        finally:
            connection.close()

    def _require_writable(self) -> None:
        if self._read_only:
            raise PermissionError("This feature store handle is read-only.")

    def _ensure_index(self) -> None:
        self._require_writable()
        with self._write_lock():
            self._ensure_index_locked()

    def _ensure_index_locked(self, *, repair: bool = False) -> None:
        if not self._deep_verify and not repair:
            self._index_new_segments_locked()
            return
        payloads = self._verified_segments()
        try:
            with self._connect() as connection:
                _rebuild_rows(connection, payloads, repair=repair)
        except sqlite3.DatabaseError as error:
            if getattr(error, "sqlite_errorcode", None) not in (
                sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB,
            ):
                raise
            # A derived corrupt index can be replaced only after every source segment verifies.
            temporary = self.directory / (INDEX_FILE + ".writing")
            temporary.unlink(missing_ok=True)
            connection = sqlite3.connect(temporary)
            try:
                _rebuild_rows(connection, payloads)
            finally:
                connection.close()
            publish_file(temporary, self.directory / INDEX_FILE)

    def _index_uncounted_segments(self) -> None:
        """Index any committed segment the index does not hold, which a crash can leave behind."""

        self._ensure_index()

    def _index_new_segments_locked(self) -> None:
        """Index only the committed segments the index lacks, reading their markers and nothing else.

        This is the recovery of a store opened with ``deep_verify=False``. A segment already in the
        index is trusted until a reader touches its parts, so the cost of a commit grows with the new
        segment and the count of segments, never with the bytes already stored. A corrupt index is
        rebuilt from the markers alone.
        """

        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                _create_index_tables(connection)
                self._index_missing_segments(connection)
                connection.commit()
        except sqlite3.DatabaseError as error:
            if getattr(error, "sqlite_errorcode", None) not in (
                sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB,
            ):
                raise
            temporary = self.directory / (INDEX_FILE + ".writing")
            temporary.unlink(missing_ok=True)
            connection = sqlite3.connect(temporary)
            try:
                connection.execute("BEGIN IMMEDIATE")
                _create_index_tables(connection)
                self._index_missing_segments(connection)
                connection.commit()
            finally:
                connection.close()
            publish_file(temporary, self.directory / INDEX_FILE)

    def _index_missing_segments(self, connection: sqlite3.Connection) -> None:
        known = {row[0] for row in connection.execute("SELECT segment FROM segments")}
        # A partial copy's index is always this code's, and lists a segment once all its parts are in.
        if not known and not self._partial:
            # An index written before the segment table existed lists its segments only through its rows.
            connection.execute("INSERT OR IGNORE INTO segments (segment) SELECT DISTINCT segment FROM rows")
            known = {row[0] for row in connection.execute("SELECT segment FROM segments")}
        for marker in sorted((self.directory / SEGMENTS_DIRECTORY).glob(f"*/{COMMIT_FILE}")):
            name = marker.parent.name
            if name in known:
                continue
            payload = self._segment_payload(name)
            complete = True
            for part in payload["parts"]:
                number = int(part["part"])
                if not self._part_path(name, number).is_file():
                    if not self._partial:
                        raise ValueError(f"Committed part {part['part']} of segment {name!r} is missing.")
                    complete = False  # not fetched: its rows read as missing until a fetch brings it
                    continue
                if self._partial and connection.execute(
                    "SELECT 1 FROM rows WHERE segment = ? AND part = ? LIMIT 1", (name, number),
                ).fetchone():
                    continue  # an earlier open of this copy indexed it
                _insert_rows(connection, name, number, part)
            if complete:
                connection.execute("INSERT OR IGNORE INTO segments (segment) VALUES (?)", (name,))


class SegmentWriter:
    """One run's segment: parts as it streams, then one commit marker and the index rows."""

    def __init__(
        self, store: FeatureStore, path: Path, fingerprint: str, metadata: dict[str, Any],
        before_commit: Callable[[], None] | None = None,
        *, lease: object | None = None, verify_staged: bool = True,
    ) -> None:
        store._require_writable()
        if lease is not _WRITER_LEASE:
            raise RuntimeError("Open a segment writer through FeatureStore.segment().")
        self.store = store
        self.path = path
        self.fingerprint = fingerprint
        self.metadata = json.loads(json.dumps(metadata, allow_nan=False))
        self.before_commit = before_commit
        self.committed = False
        self.closed = False
        self.verify_staged = verify_staged
        self._owner_pid = os.getpid()
        self._failed = False
        self._seen_digests: set[str] = set()
        self._parts: list[dict[str, Any]] = []
        # Parts are numbered when reserved and recorded when written, which may be on another thread.
        self._lock = threading.Lock()
        self._reserved = 0
        self._staged_stats: dict[str, tuple[int, int]] = {}

    def __getstate__(self) -> dict[str, Any]:
        # A lock does not pickle. A copy sent to another process gets a fresh one and still refuses every write,
        # commit and abandon, because its owner pid is not that process's.
        state = dict(self.__dict__)
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def _require_open(self) -> None:
        if self._owner_pid != os.getpid():
            raise RuntimeError("A segment writer belongs to the process that opened its context.")
        if self.committed or (self.path / COMMIT_FILE).exists():
            raise RuntimeError(f"Segment {self.fingerprint!r} is already committed.")
        if self.closed or self._failed:
            raise RuntimeError("This segment writer is closed or failed; start a fresh attempt.")

    @property
    def parts(self) -> tuple[int, ...]:
        """The part numbers written so far, which are visible only once committed."""

        return tuple(int(part["part"]) for part in self._parts)

    def append_bounded(
        self, sequences: Sequence[str],
        rows: Sequence[Tensor] | Sequence[SparseRow] | Sequence[TopKRow],
        *, max_tensor_bytes: int, row_metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[int, ...]:
        """Split a bounded window into lossless parts capped by their encoded tensor payload.

        Metadata sidecars and safetensors headers are separate. A row cannot span parts; reject
        an oversized row instead of silently writing an oversized part or changing its values.
        """
        # rows: (w,) dense or (r_i, d) ragged tensors; a SparseRow holds (nnz_i,) and a TopKRow (r_i, k) tensors
        if type(max_tensor_bytes) is not int or max_tensor_bytes <= 0:
            raise ValueError("max_tensor_bytes must be a positive integer.")
        if not sequences or len(sequences) != len(rows):
            raise ValueError("append_bounded needs one row per sequence and at least one row.")
        if len(set(sequences)) != len(sequences):
            raise ValueError("This batch repeats sequences; each feature has one row.")
        if row_metadata is not None and len(row_metadata) != len(sequences):
            raise ValueError("Row metadata must contain one identity per sequence.")
        spec = self.store.spec
        initial = 0 if spec.layout == DENSE else 8
        sizes = [
            row_tensor_bytes(row, spec.layout, spec.width, spec.dtype, positions=spec.positions)
            for row in rows
        ]
        if any(initial + size > max_tensor_bytes for size in sizes):
            raise ValueError(
                "A feature row exceeds max_tensor_bytes; increase the explicit part budget."
            )
        written, start, used = [], 0, initial
        for stop in range(len(rows) + 1):
            size = sizes[stop] if stop < len(rows) else 0
            if stop == len(rows) or used + size > max_tensor_bytes:
                part = self.append(
                    sequences[start:stop], rows[start:stop],
                    row_metadata=None if row_metadata is None else row_metadata[start:stop],
                )
                if self._parts[part]["tensor_bytes"] > max_tensor_bytes:
                    raise RuntimeError("Encoded tensor payload exceeded the planned part budget.")
                written.append(part)
                start, used = stop, initial
            used += size
        return tuple(written)

    def append(
        self, sequences: Sequence[str],
        rows: Sequence[Tensor] | Sequence[SparseRow] | Sequence[TopKRow],
        *, row_metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        """Write one part holding these rows, and return the part number.

        Sequences the store already holds, or that repeat inside this batch, are an error: a
        feature has one row, and writing it twice makes two answers to one question. Call
        ``missing`` first.
        """

        # rows: (w,) dense or (r_i, d) ragged tensors; a SparseRow holds (nnz_i,) and a TopKRow (r_i, k) tensors
        self._require_open()
        if len(sequences) != len(rows):
            raise ValueError(
                f"append needs one row per sequence; received {len(sequences)} sequences and "
                f"{len(rows)} rows."
            )
        if not sequences:
            raise ValueError("append needs at least one sequence.")
        digests = [sequence_digest(sequence) for sequence in sequences]
        repeated = sorted({digest for digest in digests if digests.count(digest) > 1})
        if repeated or self._seen_digests.intersection(digests):
            raise ValueError(f"This batch repeats {len(repeated)} sequences; each feature has one row.")
        already = self.store.missing(sequences)
        if len(already) != len(sequences):
            raise ValueError(
                f"{len(sequences) - len(already)} of these sequences already have a row in "
                f"{self.store.spec.key!r}; call missing() and embed only what it returns."
            )

        spec = self.store.spec
        if spec.descriptor.get("schema") in COMPLETE_SCHEMAS and row_metadata is None:
            raise ValueError(
                "Complete feature contracts require a persisted identity for each row."
            )
        if row_metadata is not None and len(row_metadata) != len(sequences):
            raise ValueError("Row metadata must contain one identity per sequence.")
        encoded_metadata = None if row_metadata is None else gzip.compress(
            json.dumps([dict(row) for row in row_metadata], sort_keys=True, allow_nan=False,
                       separators=(",", ":")).encode("utf-8"), mtime=0,
        )
        if spec.layout == DENSE:
            tensors = encode_dense(cast(Sequence[Tensor], rows), spec.width, spec.dtype)
            residues = [0] * len(sequences)
        elif spec.layout == CSR:
            tensors = encode_csr(cast(Sequence[SparseRow], rows), spec.width, spec.dtype)
            if spec.positions and "positions" not in tensors:
                raise ValueError(
                    f"Feature {spec.key!r} stores argmax positions; these rows carry none."
                )
            if not spec.positions and "positions" in tensors:
                raise ValueError(
                    f"Feature {spec.key!r} stores no argmax positions; these rows carry them."
                )
            residues = [0] * len(sequences)
        else:
            if spec.layout == RAGGED_TOPK:
                tensors = encode_topk_rows(
                    cast(Sequence[TopKRow], rows), spec.width,
                    cast(int, spec.sparse_count), spec.dtype,
                )
            else:
                tensors = encode_ragged(cast(Sequence[Tensor], rows), spec.width, spec.dtype)
            offsets = tensors["offsets"]  # (n + 1,)
            residues = [
                int(offsets[position + 1]) - int(offsets[position])
                for position in range(len(sequences))
            ]
        written = row_count(spec.layout, tensors)
        if written != len(sequences):
            raise ValueError(
                f"Encoded {written} rows for {len(sequences)} sequences; the layout and the rows "
                "disagree."
            )

        part = self.reserve_part()
        try:
            _transaction_event("before_part_write", self.path)
            _save_safetensors_atomically(
                self.path / PART_TEMPLATE.format(part),
                {name: tensors[name]
                 for name in tensor_names(spec.layout, positions=spec.positions)},
            )
            _transaction_event("after_part_write", self.path)
            record = {
                "part": part, "digests": digests, "residues": residues,
                "sha256": file_sha256(self.path / PART_TEMPLATE.format(part)),
                "tensor_bytes": sum(
                    value.numel() * value.element_size() for value in tensors.values()),
            }
            if encoded_metadata is not None:
                target = self.path / f"part-{part:05d}.rows.json.gz"
                temporary = target.with_name(target.name + ".writing")
                temporary.write_bytes(encoded_metadata)
                publish_file(temporary, target)
                record["row_metadata"] = {"file": target.name, "sha256": file_sha256(target)}
                _transaction_event("after_metadata_write", self.path)
        except BaseException:
            self._failed = True
            raise
        self._parts.append(record)
        self._seen_digests.update(digests)
        return part

    def reserve_part(self) -> int:
        """The next part number, taken before a part is written so that writers on several threads never collide."""

        with self._lock:
            number = self._reserved
            self._reserved += 1
        return number

    def seen(self, digests: Sequence[str]) -> bool:
        """Whether any digest was already written by this segment, a cheap guard before a large write."""

        with self._lock:
            return not self._seen_digests.isdisjoint(digests)

    def append_packed(
        self, part: int, digests: Sequence[str], tensors: Mapping[str, Tensor], residues: Sequence[int],
        *, row_metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Write one large part from tensors already packed in the layout, hashing as it writes.

        ``tensors`` holds exactly the layout's tensors, as ``encode_*`` would build them. This skips
        the per-row copies, finite scans and index re-reads of ``append``: the caller proved the values
        finite on the device and packed them in order. Shapes, dtypes, offsets and the residue counts
        are still checked, because they decide whether a reader can address the part. The file and
        its digest come from one pass, and one flush covers the part. Safe to call from several
        threads with parts from ``reserve_part``.

        Shapes, with b rows (sequences) in the part and n = sum(residues) stored rows: dense ``values`` (b, w);
        ragged ``offsets`` (b + 1,) and ``values`` (n, w); ragged top-k adds ``indices`` (n, k); csr ``indptr``
        (b + 1,) with ``indices`` and ``values`` (nnz,). ``residues[i]`` is row i's stored rows, which is l + 2
        for a stream that keeps CLS and EOS.
        """
        # tensors: (b, w) dense; (b + 1,) offsets and (n, w) values ragged; (b + 1,) indptr and (nnz,) csr.
        spec = self.store.spec
        if self._owner_pid != os.getpid():
            raise RuntimeError("A segment writer belongs to the process that opened its context.")
        if self.committed or (self.path / COMMIT_FILE).exists():
            raise RuntimeError(f"Segment {self.fingerprint!r} is already committed.")
        if self.closed or self._failed:
            raise RuntimeError("This segment writer is closed or failed; start a fresh attempt.")
        count = len(digests)
        if not count or len(residues) != count:
            raise ValueError("append_packed needs one residue count and one digest per row, and at least one row.")
        if spec.descriptor.get("schema") in COMPLETE_SCHEMAS and row_metadata is None:
            raise ValueError("Complete feature contracts require a persisted identity for each row.")
        if row_metadata is not None and len(row_metadata) != count:
            raise ValueError("Row metadata must contain one identity per sequence.")
        if set(tensors) != set(tensor_names(spec.layout, positions=spec.positions)):
            raise ValueError("Packed tensors do not match the feature layout.")
        if len(set(digests)) != count or self.seen(digests):
            raise ValueError("A packed part repeats sequences; each feature has one row.")
        _check_packed(spec, tensors, count, residues)

        try:
            _transaction_event("before_part_write", self.path)
            target = self.path / PART_TEMPLATE.format(part)
            digest, size = _write_safetensors_streaming(target, tensors)
            _transaction_event("after_part_write", self.path)
            record: dict[str, Any] = {
                "part": part, "digests": list(digests), "residues": [int(value) for value in residues],
                "sha256": digest,
                "tensor_bytes": sum(value.numel() * value.element_size() for value in tensors.values()),
            }
            stats = {target.name: (size, target.stat().st_mtime_ns)}
            if row_metadata is not None:
                encoded = gzip.compress(
                    json.dumps([dict(row) for row in row_metadata], sort_keys=True, allow_nan=False,
                               separators=(",", ":")).encode("utf-8"), mtime=0,
                )
                sidecar = self.path / f"part-{part:05d}.rows.json.gz"
                temporary = sidecar.with_name(sidecar.name + ".writing")
                temporary.write_bytes(encoded)
                publish_file(temporary, sidecar, sync_parent=False)
                record["row_metadata"] = {"file": sidecar.name, "sha256": hashlib.sha256(encoded).hexdigest()}
                stats[sidecar.name] = (len(encoded), sidecar.stat().st_mtime_ns)
                _transaction_event("after_metadata_write", self.path)
        except BaseException:
            self._failed = True
            raise
        with self._lock:
            self._parts.append(record)
            self._seen_digests.update(digests)
            self._staged_stats.update(stats)
        return record

    def commit(self) -> SegmentReceipt:
        """Write the commit marker, then index the parts. Nothing reads a segment until this."""

        self._require_open()
        with self._lock:
            # Parts written on several threads finish in any order; the marker lists them by number.
            self._parts.sort(key=lambda part: part["part"])
        if not self._parts:
            raise RuntimeError(f"Segment {self.fingerprint!r} holds no parts to commit.")
        payload = {
            "format": FORMAT,
            "key": self.store.spec.key,
            "fingerprint": self.fingerprint,
            "committed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "rows": sum(len(part["digests"]) for part in self._parts),
            "metadata": self.metadata,
            "parts": self._parts,
            "transaction_schema": 2,
            "descriptor_sha256": json_sha256(self.store.spec.payload(), allow_nan=False),
        }
        recorded = json.loads((self.store.directory / FEATURE_FILE).read_text(encoding="utf-8"))
        if recorded != self.store.spec.payload():
            raise ValueError("Feature descriptor changed while the segment was staged.")
        for part in self._parts:
            self._check_staged(self.path / PART_TEMPLATE.format(part["part"]), part["sha256"])
            identity = part.get("row_metadata")
            if identity:
                self._check_staged(self.path / identity["file"], identity["sha256"])
        if self.before_commit is not None:
            self.before_commit()
        # Parts and sidecars were renamed without a directory flush; one flush covers them all.
        sync_directory(self.path)
        payload["manifest_sha256"] = json_sha256(payload, allow_nan=False)
        with self.store._write_lock():
            # Recover prior durable markers before deciding whether any row conflicts.
            self.store._ensure_index_locked()
            with self.store._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                _transaction_event("before_index_update", self.path)
                _create_index_tables(connection)  # idempotent: an index from before the segment list gains it here
                for part in self._parts:
                    _insert_rows(connection, self.fingerprint, int(part["part"]), part)
                connection.execute("INSERT OR IGNORE INTO segments (segment) VALUES (?)", (self.fingerprint,))
                _transaction_event("after_index_update", self.path)
                _transaction_event("before_commit_marker", self.path)
                _write_json_atomically(self.path / COMMIT_FILE, payload)
                self.committed = True
                _transaction_event("after_commit_marker", self.path)
                connection.commit()
                _transaction_event("after_index_commit", self.path)
        return _receipt(payload)

    def _check_staged(self, path: Path, digest: str) -> None:
        """Refuse a staged file that changed since it was written.

        A writer that hashed as it wrote (``verify_staged=False``) compares size and modification
        time, which catches an appended or rewritten file without a second read of the payload.
        """
        recorded = self._staged_stats.get(path.name)
        if self.verify_staged or recorded is None:
            if file_sha256(path) != digest:
                raise ValueError(
                    "Staged feature data changed before commit." if path.suffix == ".safetensors"
                    else "Staged row metadata changed before commit."
                )
            return
        observed = path.stat()
        if (observed.st_size, observed.st_mtime_ns) != recorded:
            raise ValueError(
                "Staged feature data changed before commit." if path.suffix == ".safetensors"
                else "Staged row metadata changed before commit."
            )

    def abandon(self) -> None:
        """Delete this segment's parts, for a run that decides not to keep them."""

        if self._owner_pid != os.getpid():
            raise RuntimeError("A segment writer belongs to the process that opened its context.")
        if self.committed or (self.path / COMMIT_FILE).exists():
            raise RuntimeError(f"Segment {self.fingerprint!r} is committed and immutable.")
        if self.closed:
            raise RuntimeError("This segment writer is closed.")
        _discard_segment(self.path)
        self._parts.clear()
        self._seen_digests.clear()
        self.closed = True


# The documented public entry point (docs/feature_store.md, `features.__all__`), kept under its name.
def open_feature(root: str | Path, spec: StoredFeature) -> FeatureStore:  # noqa: renaming-wrapper
    """Open, or create, the store for one feature under ``root``."""

    return FeatureStore.open(root, spec)


def features_in(root: str | Path) -> tuple[FeatureStore, ...]:
    """Every feature store under ``root``, by key."""

    return tuple(
        FeatureStore.read_only(recorded.parent)
        for recorded in sorted(Path(root).glob(f"*/{FEATURE_FILE}"))
    )


def _insert_rows(
    connection: sqlite3.Connection, fingerprint: str, part: int, payload: Mapping[str, Any]
) -> None:
    try:
        connection.executemany(
            "INSERT INTO rows (digest, segment, part, row, residues) VALUES (?, ?, ?, ?, ?)",
            [(digest, fingerprint, part, row, int(residues))
             for row, (digest, residues) in enumerate(
                 zip(payload["digests"], payload["residues"], strict=True)
             )],
        )
    except sqlite3.IntegrityError as error:
        raise ValueError(
            f"Segment {fingerprint!r} conflicts with an existing sequence row."
        ) from error


def _receipt(payload: Mapping[str, Any]) -> SegmentReceipt:
    return SegmentReceipt(
        fingerprint=str(payload["fingerprint"]),
        parts=tuple(int(part["part"]) for part in payload["parts"]),
        rows=int(payload["rows"]),
        committed_at=str(payload["committed_at"]),
        metadata=payload.get("metadata") or {},
    )


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.writing")
    temporary.write_text(indented_json(payload, allow_nan=False), encoding="utf-8")
    publish_file(temporary, path)


def _save_safetensors_atomically(path: Path, tensors: Mapping[str, Tensor]) -> None:
    # tensors: (n, w), (n + 1,), (nnz,) or (sum r_i, d) by name and layout; each is written as it is
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise ImportError("Writing a feature store requires the 'safetensors' package.") from error
    temporary = path.with_name(f"{path.name}.writing")
    save_file({name: tensor.contiguous() for name, tensor in tensors.items()}, str(temporary))
    publish_file(temporary, path)


_SAFETENSORS_DTYPES = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.int64: "I64", torch.int32: "I32", torch.int16: "I16",
}
_WRITE_BLOCK_BYTES = 64 * 1024**2


def _write_safetensors_streaming(path: Path, tensors: Mapping[str, Tensor]) -> tuple[str, int]:
    """Write ``tensors`` as a safetensors file in one pass, hashing the bytes as they are written.

    The file is the safetensors format that ``safe_open`` reads: an 8-byte little-endian header
    length, a JSON header padded with spaces to a multiple of 8, then the raw tensor bytes. Tensors go
    in decreasing element size, so every tensor starts aligned to its own element size. The tensor
    memory is written from its own buffer in blocks, with no second serialization copy, and the
    SHA-256 of the file comes from those same blocks. One flush makes the part durable. Returns the
    digest and the size in bytes.
    """
    # tensors: (b, w), (n, w), (n, k), (b + 1,) or (nnz,) by layout; any shape, the header records each as is.
    ordered = sorted(tensors.items(), key=lambda item: (-item[1].element_size(), item[0]))
    header: dict[str, Any] = {}
    cursor = 0
    for name, tensor in ordered:
        size = tensor.numel() * tensor.element_size()
        header[name] = {
            "dtype": _SAFETENSORS_DTYPES[tensor.dtype], "shape": list(tensor.shape),
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    prefix = len(encoded).to_bytes(8, "little")
    temporary = path.with_name(f"{path.name}.writing")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for block in (prefix, encoded):
            handle.write(block)
            digest.update(block)
        for _, tensor in ordered:
            raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8)  # (bytes,) over the tensor's own memory
            buffer = memoryview(raw.numpy())
            for start in range(0, len(buffer), _WRITE_BLOCK_BYTES):
                block = buffer[start : start + _WRITE_BLOCK_BYTES]
                handle.write(block)
                digest.update(block)
        handle.flush()
        flush_and_evict(handle)
    temporary.replace(path)
    return digest.hexdigest(), 8 + len(encoded) + cursor


def _check_packed(
    spec: StoredFeature, tensors: Mapping[str, Tensor], count: int, residues: Sequence[int],
) -> None:
    """The structural checks `append_packed` keeps: whatever decides whether a reader can address the part."""
    # tensors: (b, w) dense; (b + 1,) offsets and (n, w) values ragged; (b + 1,) indptr and (nnz,) csr.
    values = tensors["values"]  # (b, w) dense, (n, w) ragged, (nnz,) csr
    if values.dtype != spec.dtype:
        raise ValueError("Packed tensor dtype does not match the feature.")
    if spec.layout == DENSE:
        if tuple(values.shape) != (count, spec.width) or any(residues):
            raise ValueError("Packed dense shape or residue counts disagree.")
        return
    offsets = tensors["indptr" if spec.layout == CSR else "offsets"]  # (b + 1,) int64 row boundaries
    if (offsets.dtype != torch.int64 or tuple(offsets.shape) != (count + 1,)
            or int(offsets[0]) != 0 or int(offsets[-1]) != len(values)
            or bool((offsets[1:] < offsets[:-1]).any())):
        raise ValueError("Packed row offsets are invalid.")
    spans = (offsets[1:] - offsets[:-1]).tolist()  # (b,) stored rows (ragged) or entries (csr) per row
    if spec.layout == RAGGED:
        if values.ndim != 2 or values.shape[1] != spec.width or spans != list(residues):
            raise ValueError("Packed ragged shape or residue counts disagree.")
    elif spec.layout == RAGGED_TOPK:
        indices = tensors["indices"]  # (n, k) int32 codes, the shape of values
        if (indices.dtype != torch.int32 or tuple(indices.shape) != tuple(values.shape)
                or values.ndim != 2 or values.shape[1] != spec.sparse_count or spans != list(residues)):
            raise ValueError("Packed top-k shape, index dtype or residue counts disagree.")
    else:
        indices = tensors["indices"]
        if (values.ndim != 1 or indices.dtype != torch.int32 or indices.shape != values.shape
                or any(residues)):
            raise ValueError("Packed sparse shape or residue counts disagree.")
        if spec.positions and (tensors["positions"].dtype != torch.int16
                               or tensors["positions"].shape != values.shape):
            raise ValueError("Packed sparse positions are invalid.")


def _discard_segment(path: Path) -> None:
    if path.is_symlink() or any(not part.is_file() or part.is_symlink() for part in path.iterdir()):
        raise ValueError("Refusing to discard an uncommitted segment with unexpected entries.")
    for part in path.iterdir():
        part.unlink()
    path.rmdir()
    sync_directory(path.parent)


def _rebuild_rows(
    connection: sqlite3.Connection, payloads: Sequence[Mapping[str, Any]], *, repair: bool = False,
) -> None:
    connection.execute("BEGIN IMMEDIATE")
    _create_index_tables(connection)
    expected = {
        digest: (payload["fingerprint"], part["part"], row, residues)
        for payload in payloads for part in payload["parts"]
        for row, (digest, residues) in enumerate(
            zip(part["digests"], part["residues"], strict=True))
    }
    if repair:
        connection.execute("DELETE FROM rows")
    indexed = {
        row[0]: tuple(row[1:])
        for row in connection.execute("SELECT digest, segment, part, row, residues FROM rows")
    }
    if any(expected.get(digest) != address for digest, address in indexed.items()):
        raise ValueError(
            "Feature index disagrees with committed rows; inspect it and call reindex()."
        )
    # Recover missing rows only. Changed addresses must fail, not silently become cache hits.
    connection.executemany(
        "INSERT INTO rows (digest, segment, part, row, residues) VALUES (?, ?, ?, ?, ?)",
        [(digest, *address) for digest, address in expected.items() if digest not in indexed],
    )
    connection.executemany(
        "INSERT OR IGNORE INTO segments (segment) VALUES (?)", [(payload["fingerprint"],) for payload in payloads],
    )
    connection.commit()


def _create_index_tables(connection: sqlite3.Connection) -> None:
    """The row index, and the list of segments it already holds, so recovery can skip them."""

    connection.execute(
        "CREATE TABLE IF NOT EXISTS rows (digest TEXT PRIMARY KEY, segment TEXT NOT NULL, "
        "part INTEGER NOT NULL, row INTEGER NOT NULL, residues INTEGER NOT NULL)"
    )
    connection.execute("CREATE INDEX IF NOT EXISTS rows_by_segment ON rows (segment, part)")
    connection.execute("CREATE TABLE IF NOT EXISTS segments (segment TEXT PRIMARY KEY)")


def _transaction_event(stage: str, path: Path) -> None:
    """A test observation point at a real I/O boundary; production performs no action."""


def _chunks(values: Sequence[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


__all__ = [
    "COMMIT_FILE",
    "FEATURE_FILE",
    "FORMAT",
    "INDEX_FILE",
    "SEGMENTS_DIRECTORY",
    "FeatureStore",
    "RowAddress",
    "SegmentReceipt",
    "SegmentWriter",
    "StoredFeature",
    "features_in",
    "open_feature",
    "partition_sequences",
    "sequence_digest",
]
