"""Random-access reads of one feature, for loops that ask for a few rows at a time.

``FeatureStore.read`` is the verified one-shot read. Each call opens a fresh index connection,
hashes every part the request touches, and loads each of those parts whole, so a training loop that
reads one batch per step pays for the whole store on every step. ``FeatureReader`` pays the same
verification once per part, then serves each row by a memory-mapped slice of the part file.

A reader gives the same rows as the store, in the same types, and refuses what the store refuses:
a missing sequence raises, changed part bytes raise on first access, and independent content pins
are honored. What it changes is the cost:
one read-only index connection per thread, one open handle per verified part, and one slice per row.

A reader belongs to a single feature directory and may be shared by threads. It pickles as the store
it reads, so a spawned worker reopens and re-verifies only the parts it touches. Call ``verify``
before forking to pay for verification once instead of once per worker.

A reader given a ``receipt`` path remembers its verifications across processes (see ``receipts``):
a part whose marker digests, size and modification time match the receipt is opened without hashing
or loading it, and every part the reader does verify is recorded. A pickled reader keeps its
receipt, so spawned workers skip the verification too. A store opened with content pins never trusts
a receipt.
"""

from __future__ import annotations

import gzip
import json
import os
import sqlite3
import sys
import threading
import torch

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast
from torch import Tensor
from tqdm import tqdm

from .digests import file_sha256
from .layouts import CSR, DENSE, RAGGED_TOPK, SparseRow, TopKRow
from .receipts import PartReceipt
from .store import (
    COMMIT_FILE,
    FEATURE_FILE,
    INDEX_FILE,
    PART_TEMPLATE,
    SEGMENTS_DIRECTORY,
    FeatureStore,
    RowAddress,
    StoredFeature,
    sequence_digest,
)


_LOOKUP_CHUNK = 512


@dataclass(frozen=True, slots=True)
class CsrRows:
    """Rows of a csr feature gathered into one compressed-sparse-row block, in the order asked.

    ``indptr`` is ``(n + 1,)`` int64 and names each row's span of ``indices``, ``values`` and
    ``positions``, which are ``(nnz,)``. Wrap them in whatever sparse type the caller uses: no
    array library is imported here.
    """

    indptr: Tensor
    indices: Tensor
    values: Tensor
    positions: Tensor | None


@dataclass(slots=True)
class _VerifiedPart:
    """A part that passed verification: its open handle, and its row offsets in memory.

    Offsets are ``(n + 1,)`` int64 and small next to the values they index, so they stay resident.
    Values, indices and positions are read by slice from ``handle``.
    """

    handle: Any
    offsets: Tensor | None


class FeatureReader:
    """Verified random access to one feature's rows."""

    def __init__(
        self, store: FeatureStore, *, receipt: str | Path | None = None, trust_receipt: bool = True,
    ) -> None:
        self.store = store
        self._receipt_path = None if receipt is None else Path(receipt)
        self._trust_receipt = trust_receipt
        # Pinned stores check each file against its pin, which a receipt cannot stand in for.
        self._receipt = None
        if self._receipt_path is not None and store._content_pins is None:
            self._receipt = PartReceipt(
                self._receipt_path, store.directory, store.spec.payload(), trust=trust_receipt,
            )
        self._lock = threading.RLock()
        self._connections: dict[tuple[int, int], sqlite3.Connection] = {}
        self._markers: dict[str, dict[str, Any]] = {}
        self._parts: dict[tuple[str, int], _VerifiedPart] = {}
        # Different parts verify concurrently; one part verifies only once.
        self._part_locks: dict[tuple[str, int], threading.Lock] = {}
        self._closed = False

    @classmethod
    def open(
        cls, directory: str | Path, *, content_pins: dict[str, str] | None = None,
    ) -> FeatureReader:
        """A reader over an existing feature directory, which it never creates or repairs."""

        return cls(FeatureStore.read_only(directory, content_pins=content_pins))

    def __reduce__(self) -> tuple[Any, tuple[FeatureStore]]:
        restore = partial(
            FeatureReader, receipt=self._receipt_path, trust_receipt=self._trust_receipt,
        )
        return (restore, (self.store,))

    def __enter__(self) -> FeatureReader:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    @property
    def spec(self) -> StoredFeature:
        return self.store.spec

    def close(self) -> None:
        """Save the receipt, then release every connection and part handle; reads then raise."""

        if self._receipt is not None:
            self._receipt.save()
        with self._lock:
            self._closed = True
            for connection in self._connections.values():
                connection.close()
            self._connections.clear()
            self._parts.clear()

    # Membership --------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._connection().execute("SELECT count(*) FROM rows").fetchone()[0])

    def __contains__(self, sequence: str) -> bool:
        digest = sequence_digest(sequence)
        return digest in self._found([digest])

    def missing(self, sequences: Iterable[str]) -> tuple[str, ...]:
        """The sequences this feature lacks, in the order given, without repeats."""

        wanted: dict[str, str] = {}
        for sequence in sequences:
            wanted.setdefault(sequence_digest(sequence), sequence)
        present = self._found(list(wanted))
        return tuple(sequence for digest, sequence in wanted.items() if digest not in present)

    # Reading -----------------------------------------------------------------

    def verify(
        self, sequences: Sequence[str] | None = None, *, workers: int = 1,
        progress: str | None = None,
    ) -> int:
        """Verify the parts holding these sequences, or every committed part, and count them.

        Reading verifies lazily, so this is only for paying the cost at a moment of the caller's
        choosing, such as before a data loader forks its workers. Verifying a part hashes its
        bytes and loads its tensors, so ``workers`` threads verify that many parts at a time
        (hashing and the loads release the interpreter lock). The first part to fail raises,
        and parts not yet started are not verified. A part the receipt vouches for is opened
        without hashing; the receipt is saved when the pass ends. ``progress`` names a progress
        bar on stderr, one tick per part.
        """

        if workers < 1:
            raise ValueError("verify needs at least one worker.")
        if sequences is not None:
            wanted = {(address.segment, address.part) for _, address in self._resolved(sequences)}
        else:
            wanted = {
                (fingerprint, int(part["part"]))
                for fingerprint in self._segment_names()
                for part in self._marker(fingerprint)["parts"]
            }
        ordered = sorted(wanted)
        bar = tqdm(
            total=len(ordered), desc=progress, unit="part", file=sys.stderr, mininterval=10.0,
            disable=progress is None,
        )

        def verified(segment: str, number: int) -> None:
            self._part(segment, number)
            bar.update()

        try:
            if workers == 1 or len(ordered) < 2:
                for segment, number in ordered:
                    verified(segment, number)
                return len(wanted)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(verified, segment, number) for segment, number in ordered]
                try:
                    for future in futures:
                        future.result()
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
            return len(wanted)
        finally:
            bar.close()
            if self._receipt is not None:
                self._receipt.save()

    def addresses(self, sequences: Sequence[str]) -> list[RowAddress]:
        """Each sequence's address, in the order asked, from the index and its commit markers alone.

        This reads no part, so it costs index lookups, not bytes. It raises ``KeyError`` for a
        sequence the feature lacks and ``ValueError`` when the index and the commit marker disagree
        about a row. Nothing here verifies the part holding the row: ``verify`` or a read does.
        """

        return [address for _, address in self._resolved(sequences)]

    def content_pins(self, sequences: Sequence[str], *, workers: int = 1) -> dict[str, str]:
        """Pin freshly hashed selection bytes after reusing this reader's layout verification.

        Every selected file is hashed again, including when a verification receipt was trusted.
        Cached part handles avoid a second whole-part tensor load. Marker changes and even
        data rewrites preserving size and modification time fail before these pins are returned.
        """
        self.verify(sequences, workers=workers)
        feature = self.store.directory / FEATURE_FILE
        if StoredFeature.from_payload(json.loads(feature.read_text(encoding="utf-8"))) != self.spec:
            raise ValueError("Feature descriptor changed while pinning a selection.")
        pins = {FEATURE_FILE: file_sha256(feature)}
        selected = {(address.segment, address.part) for _, address in self._resolved(sequences)}
        markers: set[str] = set()
        for segment, number in sorted(selected):
            payload = self._marker(segment)
            prefix = f"{SEGMENTS_DIRECTORY}/{segment}"
            if segment not in markers:
                marker = self.store.directory / prefix / COMMIT_FILE
                if json.loads(marker.read_text(encoding="utf-8")) != payload:
                    raise ValueError("Feature commit marker changed while pinning a selection.")
                pins[f"{prefix}/{COMMIT_FILE}"] = file_sha256(marker)
                markers.add(segment)
            part = payload["parts"][number]
            pins[f"{prefix}/{PART_TEMPLATE.format(number)}"] = part["sha256"]
            sidecar = part.get("row_metadata")
            if sidecar is not None:
                pins[f"{prefix}/{sidecar['file']}"] = sidecar["sha256"]

        def check(relative: str) -> None:
            actual = file_sha256(self.store.directory / relative)
            if actual != pins[relative]:
                raise ValueError("Feature bytes changed while pinning a selection.")
            self.store._check_pin(relative, actual)

        if workers == 1:
            for relative in pins:
                check(relative)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(check, pins))
        return pins

    def residue_counts(self, sequences: Sequence[str]) -> list[int]:
        """Each sequence's stored residue count, which a length-bucketed reader batches by."""

        return [address.residues for _, address in self._resolved(sequences, verify=True)]

    def row_metadata(self, sequences: Sequence[str]) -> list[dict[str, Any]]:
        """Return row identities using the same immutable-part verification as feature reads."""
        identities = []
        parts: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for _, address in self._resolved(sequences):
            self._part(address.segment, address.part)
            key = (address.segment, address.part)
            if key not in parts:
                committed = self._marker(address.segment)["parts"][address.part]
                sidecar = committed.get("row_metadata")
                if not isinstance(sidecar, dict):
                    raise ValueError(
                        "Stored row identities are unavailable; recompute this legacy feature."
                    )
                path = self.store.directory / SEGMENTS_DIRECTORY / address.segment / sidecar["file"]
                records = json.loads(gzip.decompress(path.read_bytes()))
                if (not isinstance(records, list) or len(records) != len(committed["digests"])
                        or not all(isinstance(record, dict) for record in records)):
                    raise ValueError("Stored row metadata does not match the committed part rows.")
                parts[key] = records
            identities.append(dict(parts[key][address.row]))
        return identities

    def read(self, sequences: Sequence[str]) -> list[Tensor]:
        """Each sequence's feature as a tensor, in the order asked, as ``FeatureStore.read`` gives.

        A dense feature gives ``(w,)``, a ragged one ``(r_i, d)``, and a csr one a densified
        ``(w,)`` row. ``ragged_topk`` requires ``read_topk``.
        """

        spec = self.spec
        if spec.layout == CSR:
            # Returns b vectors, each (w,).
            return [row.to_dense(spec.width) for row in self.read_sparse(sequences)]  # (w,) per row
        if spec.layout == RAGGED_TOPK:
            raise ValueError(
                "Use read_topk for sparse residue codes; implicit densification is refused."
            )
        rows: list[Tensor] = []
        parts: dict[tuple[str, int], tuple[_VerifiedPart, Tensor]] = {}
        for _, address in self._resolved(sequences):
            key = (address.segment, address.part)
            if key not in parts:
                part = self._part(*key)
                # One mapped tensor per touched part, not one safetensors wrapper per row.
                parts[key] = (part, part.handle.get_tensor("values"))
            part, values = parts[key]  # (n_part, w) dense or (r_part, d) ragged
            if spec.layout == DENSE:
                rows.append(values[address.row].clone())  # (w,)
            else:
                start, stop = self._span(part, address)
                rows.append(values[start:stop].clone())  # (r_i, d)
        return rows  # b tensors, each (w,) for dense or (r_i, d) for ragged

    def read_sparse(self, sequences: Sequence[str]) -> list[SparseRow]:
        """Each sequence's compressed row, in the order asked. Only for a csr feature."""

        if self.spec.layout != CSR:
            raise ValueError(f"read_sparse needs a csr feature; this one is {self.spec.layout}.")
        rows: list[SparseRow] = []
        parts: dict[tuple[str, int], tuple[_VerifiedPart, Tensor, Tensor, Tensor | None]] = {}
        for _, address in self._resolved(sequences):
            key = (address.segment, address.part)
            if key not in parts:
                part = self._part(*key)
                parts[key] = (
                    part, part.handle.get_tensor("indices"), part.handle.get_tensor("values"),
                    part.handle.get_tensor("positions") if self.spec.positions else None,
                )
            part, indices, values, positions = parts[key]  # (nnz_part,) per tensor
            start, stop = self._span(part, address)
            rows.append(SparseRow(
                indices=indices[start:stop].clone(),  # (nnz_i,)
                values=values[start:stop].clone(),  # (nnz_i,)
                positions=(
                    positions[start:stop].clone() if positions is not None else None  # (nnz_i,)
                ),
            ))
        return rows

    def read_csr(self, sequences: Sequence[str]) -> CsrRows:
        """These sequences' rows as one compressed-sparse-row block, in the order asked.

        A sequence asked for twice appears twice. This is the read for a caller that wants a matrix,
        such as a design matrix for a gradient-boosted model, instead of one row at a time.
        """

        spec = self.spec
        rows = self.read_sparse(sequences)
        counts = torch.tensor([row.indices.numel() for row in rows], dtype=torch.int64)  # (n,)
        indptr = torch.zeros(len(rows) + 1, dtype=torch.int64)  # (n + 1,)
        torch.cumsum(counts, dim=0, out=indptr[1:])

        def joined(tensors: list[Tensor], dtype: torch.dtype) -> Tensor:
            # tensors: (nnz_i,) per input vector.
            return torch.cat(tensors) if tensors else torch.empty(0, dtype=dtype)  # (nnz,)

        return CsrRows(
            indptr=indptr,
            indices=joined([row.indices for row in rows], torch.int32),  # (nnz,)
            values=joined([row.values for row in rows], spec.dtype),  # (nnz,)
            positions=(
                joined([cast(Tensor, row.positions) for row in rows], torch.int16)  # (nnz,)
                if spec.positions else None
            ),
        )

    def read_topk(self, sequences: Sequence[str]) -> list[TopKRow]:
        """Each sequence's sparse ``(r_i, k)`` residue codes, in the order asked."""

        if self.spec.layout != RAGGED_TOPK:
            raise ValueError(
                f"read_topk needs a ragged_topk feature; this one is {self.spec.layout}."
            )
        rows: list[TopKRow] = []
        for _, address in self._resolved(sequences):
            part = self._part(address.segment, address.part)
            start, stop = self._span(part, address)
            rows.append(TopKRow(
                cast(Tensor, part.handle.get_slice("indices")[start:stop]),  # (r_i, k)
                cast(Tensor, part.handle.get_slice("values")[start:stop]),  # (r_i, k)
            ))
        return rows

    # Internals ---------------------------------------------------------------

    def _connection(self) -> sqlite3.Connection:
        """This thread's read-only index connection, opened on first use.

        Connections are keyed by process and thread, so a forked worker or a prefetch thread opens
        its own and never touches one that another owner holds.
        """

        owner = (os.getpid(), threading.get_ident())
        with self._lock:
            if self._closed:
                raise ValueError("This feature reader is closed.")
            connection = self._connections.get(owner)
            if connection is None:
                database = (self.store.directory / INDEX_FILE).resolve()
                connection = sqlite3.connect(
                    database.as_uri() + "?mode=ro", uri=True, timeout=30,
                    check_same_thread=False,
                )
                self._connections[owner] = connection
            return connection

    def _found(self, digests: Sequence[str]) -> dict[str, RowAddress]:
        found: dict[str, RowAddress] = {}
        connection = self._connection()
        for start in range(0, len(digests), _LOOKUP_CHUNK):
            chunk = list(digests[start:start + _LOOKUP_CHUNK])
            marks = ",".join("?" * len(chunk))
            # fetchall ends the read transaction so writers can commit.
            for digest, segment, part, row, residues in connection.execute(
                f"SELECT digest, segment, part, row, residues FROM rows WHERE digest IN ({marks})",
                chunk,
            ).fetchall():
                found[digest] = RowAddress(segment, part, row, residues)
        return found

    def _resolved(
        self, sequences: Sequence[str], *, verify: bool = False,
    ) -> list[tuple[str, RowAddress]]:
        """Each sequence with its address, checked against the commit marker that owns it."""

        self.store._check_pin(FEATURE_FILE)
        digests = [sequence_digest(sequence) for sequence in sequences]
        found = self._found(list(dict.fromkeys(digests)))
        resolved: list[tuple[str, RowAddress]] = []
        for sequence, digest in zip(sequences, digests, strict=True):
            address = found.get(digest)
            if address is None:
                raise KeyError(
                    f"Feature {self.spec.key!r} has no row for a sequence of "
                    f"{len(sequence)} residues (sha256 {digest[:12]}). "
                    "Embed it first, or call missing() before reading."
                )
            payload = self._marker(address.segment)
            if not 0 <= address.part < len(payload["parts"]):
                raise ValueError("Index part does not name one committed feature part.")
            part = payload["parts"][address.part]
            if (not 0 <= address.row < len(part["digests"])
                    or part["digests"][address.row] != digest
                    or part["residues"][address.row] != address.residues):
                raise ValueError("Feature index and committed sequence row disagree.")
            if verify:
                self._part(address.segment, address.part)
            resolved.append((sequence, address))
        return resolved

    def _marker(self, segment: str) -> dict[str, Any]:
        """A validated commit marker, read once per reader."""

        with self._lock:
            payload = self._markers.get(segment)
            if payload is None:
                payload = self.store._segment_payload(segment)
                self._markers[segment] = payload
            return payload

    def _segment_names(self) -> list[str]:
        directory = self.store.directory / SEGMENTS_DIRECTORY
        return sorted(marker.parent.name for marker in directory.glob(f"*/{COMMIT_FILE}"))

    def _part(self, segment: str, number: int) -> _VerifiedPart:
        """A part's open handle, verified the first time and kept.

        Verification is the store's own: checksum against the marker, physical layout, and the row
        identity sidecar. The tensors it loads are dropped afterwards except the offsets.
        """

        key = (segment, number)
        with self._lock:
            if self._closed:
                raise ValueError("This feature reader is closed.")
            opened = self._parts.get(key)
            if opened is not None:
                return opened
            part_lock = self._part_locks.setdefault(key, threading.Lock())
        with part_lock:
            # Verify outside the reader lock so other parts can progress.
            with self._lock:
                opened = self._parts.get(key)
            if opened is not None:
                return opened
            from safetensors import safe_open

            committed = self._marker(segment)["parts"][number]
            path = str(self.store._part_path(segment, number))
            offsets_name = "indptr" if self.spec.layout == CSR else "offsets"
            receipt = self._receipt
            described = None if receipt is None else receipt.describe(segment, number, committed)
            if (receipt is not None and described is not None
                    and receipt.trusts(segment, number, described)):
                # An earlier full verification passed these digests on files of this size and time.
                handle = safe_open(path, framework="pt", device="cpu")
                names = set(handle.keys())
                offsets = handle.get_tensor(offsets_name) if offsets_name in names else None
            else:
                tensors = self.store._verified_part(segment, committed)
                offsets = tensors.get(offsets_name)
                del tensors
                handle = safe_open(path, framework="pt", device="cpu")
                if receipt is not None and described is not None:
                    if receipt.describe(segment, number, committed) != described:
                        raise ValueError(
                            f"A feature part changed during verification: {segment}/{number}."
                        )
                    receipt.record(segment, number, described)
            opened = _VerifiedPart(handle=handle, offsets=offsets)
            with self._lock:
                if self._closed:
                    raise ValueError("This feature reader is closed.")
                self._parts[key] = opened
            return opened

    @staticmethod
    def _span(part: _VerifiedPart, address: RowAddress) -> tuple[int, int]:
        """A row's ``[start, stop)`` span in its part's flattened values."""

        assert part.offsets is not None, "Only sparse and ragged layouts carry row offsets."
        return int(part.offsets[address.row]), int(part.offsets[address.row + 1])


__all__ = ["FeatureReader"]
