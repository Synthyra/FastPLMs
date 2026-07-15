"""Multiple-sequence-alignment value objects and lossless encodings."""

from __future__ import annotations

import dataclasses
import string
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from typing import Any

import numpy as np
from Bio import SeqIO
from scipy.spatial.distance import cdist

from .esmfold2_misc import slice_any_object
from .esmfold2_msa_filter_sequences import greedy_select_indices, hhfilter
from .esmfold2_parsing import FastaEntry, read_sequences, write_sequences
from .esmfold2_sequential_dataclass import SequentialDataclass
from .esmfold2_system import PathOrBuffer

_A3M_INSERTION_DELETE_TABLE = str.maketrans(dict.fromkeys(string.ascii_lowercase))
_SERIALIZATION_VERSION = 1
_UINT32_BYTES = 4


def is_a3m_insertion(character: str) -> bool:
    """Return whether a character is an A3M insertion marker."""

    return character == "." or character.islower()


def remove_insertions_from_sequence(sequence: str) -> str:
    """Remove lowercase A3M insertion residues from a sequence."""

    return sequence.translate(_A3M_INSERTION_DELETE_TABLE)


def a3m_deletion_counts(sequence: str) -> np.ndarray:
    """Count insertions preceding each A3M match column."""

    codes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    lowercase = (codes >= ord("a")) & (codes <= ord("z"))
    insertion_mask = lowercase | (codes == ord("."))
    prefix_counts = np.concatenate(([0], np.cumsum(insertion_mask)))
    match_positions = np.flatnonzero(~insertion_mask)
    return np.diff(prefix_counts[match_positions], prepend=0)


def _parse_full_payload(data: bytes) -> tuple[np.ndarray, list[str]]:
    version = int.from_bytes(data[:1], "little")
    if version != _SERIALIZATION_VERSION:
        raise ValueError(f"Unsupported version: {version}")
    seqlen = int.from_bytes(data[1:5], "little")
    depth = int.from_bytes(data[5:9], "little")
    body = data[9:]
    split = seqlen * depth
    array = np.frombuffer(body[:split], dtype="|S1").reshape(depth, seqlen)
    headers = [header for header in body[split:].decode().split("\n") if header]
    if not headers and depth > 0:
        headers = [""] * depth
    return array, headers


def _parse_sequence_payload(data: bytes) -> np.ndarray:
    seqlen = int.from_bytes(data[:_UINT32_BYTES], "little")
    return np.frombuffer(data[_UINT32_BYTES:], dtype="|S1").reshape(-1, seqlen)


def _full_payload(array: np.ndarray, headers: Sequence[str]) -> bytes:
    depth, seqlen = array.shape
    prefix = b"".join(
        (
            _SERIALIZATION_VERSION.to_bytes(1, "little"),
            seqlen.to_bytes(_UINT32_BYTES, "little"),
            depth.to_bytes(_UINT32_BYTES, "little"),
        )
    )
    return prefix + array.tobytes() + "\n".join(headers).encode()


def _sequence_payload(array: np.ndarray) -> bytes:
    return array.shape[1].to_bytes(_UINT32_BYTES, "little") + array.tobytes()


def _random_row_indices(depth: int, count: int) -> np.ndarray:
    sampled = np.random.choice(depth - 1, count - 1, replace=False) + 1
    return np.sort(np.append(0, sampled))


@dataclass(frozen=True)
class FastMSA(SequentialDataclass):
    """An MSA stored as a two-dimensional NumPy byte array."""

    array: np.ndarray
    headers: list[str] | None = None

    def __post_init__(self) -> None:
        if self.headers is not None:
            assert len(self.headers) == self.depth, "Number of headers must match depth."

    @property
    def depth(self) -> int:
        return self.array.shape[0]

    @property
    def seqlen(self) -> int:
        return self.array.shape[1]

    def __len__(self) -> int:
        return self.seqlen

    @classmethod
    def from_bytes(cls, data: bytes) -> FastMSA:
        array, headers = _parse_full_payload(data)
        return cls(array, headers)

    @classmethod
    def from_sequence_bytes(cls, data: bytes) -> FastMSA:
        return cls(_parse_sequence_payload(data))

    def __getitem__(
        self,
        indices: int | list[int] | slice | np.ndarray,
    ) -> FastMSA:
        column_indices = [indices] if isinstance(indices, int) else indices
        return dataclasses.replace(self, array=self.array[:, column_indices])

    def select_sequences(
        self,
        indices: Sequence[int] | np.ndarray,
    ) -> FastMSA:
        headers = None
        if self.headers is not None:
            headers = [self.headers[index] for index in indices]
        return dataclasses.replace(
            self,
            array=self.array[indices],
            headers=headers,
        )

    def select_random_sequences(self, num_seqs: int) -> FastMSA:
        if num_seqs >= self.depth:
            return self
        return self.select_sequences(_random_row_indices(self.depth, num_seqs))

    def pad_to_depth(self, depth: int) -> FastMSA:
        if depth < self.depth:
            raise ValueError(f"Cannot pad to depth {depth} when depth is {self.depth}")
        if depth == self.depth:
            return self
        row_count = depth - self.depth
        pad_value = ord("-") if self.array.dtype == np.uint8 else b"-"
        array = np.pad(
            self.array,
            ((0, row_count), (0, 0)),
            constant_values=pad_value,
        )
        headers = None if self.headers is None else self.headers + [""] * row_count
        return dataclasses.replace(self, array=array, headers=headers)

    @classmethod
    def concat(
        cls,
        msas: Sequence[FastMSA],
        join_token: str | None = None,
        allow_depth_mismatch: bool = False,
    ) -> FastMSA:
        if not msas:
            raise ValueError("Cannot concatenate an empty list of MSAs")
        if join_token not in (None, ""):
            raise NotImplementedError("join_token is not supported for FastMSA")
        depths = [msa.depth for msa in msas]
        if len(set(depths)) != 1:
            if not allow_depth_mismatch:
                raise ValueError("Depth mismatch in concatenating MSAs")
            maximum_depth = max(depths)
            msas = [msa.pad_to_depth(maximum_depth) for msa in msas]
        header_columns = (
            msa.headers if msa.headers is not None else [""] * msa.depth for msa in msas
        )
        headers = [
            "|".join(str(header) for header in row) for row in zip(*header_columns, strict=False)
        ]
        return cls(
            np.concatenate([msa.array for msa in msas], axis=1),
            headers,
        )

    @classmethod
    def stack(
        cls,
        msas: Sequence[FastMSA],
        remove_query_from_later_msas: bool = True,
    ) -> FastMSA:
        arrays = []
        headers = []
        for index, msa in enumerate(msas):
            start = 1 if index > 0 and remove_query_from_later_msas else 0
            arrays.append(msa.array[start:])
            if msa.headers is not None:
                headers.extend(msa.headers[start:])
        return cls(np.concatenate(arrays, axis=0), headers)

    def to_msa(self) -> MSA:
        headers = self.headers
        if headers is None:
            headers = [f"seq{index}" for index in range(self.depth)]
        entries = [
            FastaEntry(header, b"".join(row).decode())
            for header, row in zip(headers, self.array, strict=False)
        ]
        return MSA(entries)


@dataclass(frozen=True)
class MSA(SequentialDataclass):
    """An ordered set of aligned protein sequences and optional A3M metadata."""

    entries: list[FastaEntry]
    deletions: np.ndarray | None = dataclasses.field(default=None, compare=False)

    @cached_property
    def sequences(self) -> list[str]:
        return [entry.sequence for entry in self.entries]

    @cached_property
    def headers(self) -> list[str]:
        return [entry.header for entry in self.entries]

    @property
    def depth(self) -> int:
        return len(self.entries)

    @property
    def seqlen(self) -> int:
        return len(self.entries[0].sequence)

    @property
    def query(self) -> str:
        return self.entries[0].sequence

    @cached_property
    def array(self) -> np.ndarray:
        return np.array([list(sequence) for sequence in self.sequences], dtype="|S1")

    @cached_property
    def seqid(self) -> np.ndarray:
        byte_array = self.array.view(np.uint8)
        return (1 - cdist(byte_array[0][None], byte_array, "hamming"))[0]

    def __len__(self) -> int:
        return self.seqlen

    def __repr__(self) -> str:
        return f"MSA({self.entries[0].header}: Depth={self.depth}, Length={self.seqlen})"

    @classmethod
    def from_a3m(
        cls,
        path: PathOrBuffer,
        remove_insertions: bool = True,
        max_sequences: int | None = None,
    ) -> MSA:
        entries = []
        deletion_rows = []
        for header, raw_sequence in islice(read_sequences(path), max_sequences):
            deletion_rows.append(a3m_deletion_counts(raw_sequence))
            sequence = (
                remove_insertions_from_sequence(raw_sequence) if remove_insertions else raw_sequence
            )
            if entries:
                expected_length = len(entries[0].sequence)
                assert len(sequence) == expected_length, (
                    "Sequence length mismatch. "
                    f"Expected: {expected_length}, Received: {len(sequence)}"
                )
            entries.append(FastaEntry(header, sequence))
        deletions = None
        if deletion_rows:
            deletions = np.stack(deletion_rows).astype(np.float32)
        return cls(entries, deletions=deletions)

    @classmethod
    def from_stockholm(
        cls,
        path: PathOrBuffer,
        remove_insertions: bool = True,
        max_sequences: int | None = None,
    ) -> MSA:
        entries = []
        for record in islice(SeqIO.parse(path, "stockholm"), max_sequences):
            sequence = str(record.seq)
            if entries:
                expected_length = len(entries[0].sequence)
                assert len(sequence) == expected_length, (
                    "Sequence length mismatch. "
                    f"Expected: {expected_length}, Received: {len(sequence)}"
                )
            entries.append(FastaEntry(f"{record.id} {record.description}", sequence))
        msa = cls(entries)
        if remove_insertions:
            msa = msa.select_positions(
                [index for index, residue in enumerate(msa.query) if residue != "-"]
            )
        return msa

    @classmethod
    def from_sequences(
        cls,
        sequences: list[str],
        remove_insertions: bool = False,
    ) -> MSA:
        transform = (
            remove_insertions_from_sequence if remove_insertions else lambda sequence: sequence
        )
        return cls([FastaEntry("", transform(sequence)) for sequence in sequences])

    @classmethod
    def from_bytes(cls, data: bytes) -> MSA:
        array, headers = _parse_full_payload(data)
        return cls(
            [
                FastaEntry(header, b"".join(row).decode())
                for header, row in zip(headers, array, strict=False)
            ]
        )

    @classmethod
    def from_sequence_bytes(cls, data: bytes) -> MSA:
        array = _parse_sequence_payload(data)
        return cls([FastaEntry("", b"".join(row).decode()) for row in array])

    @classmethod
    def from_state_dict(cls, dct: dict[str, Any]) -> MSA:
        deletions = dct.get("deletions")
        return cls(
            [FastaEntry("", sequence) for sequence in dct["sequences"]],
            deletions=(None if deletions is None else np.asarray(deletions, dtype=np.float32)),
        )

    def to_a3m(self, path: PathOrBuffer) -> None:
        write_sequences(self.entries, path)

    def to_fast_msa(self) -> FastMSA:
        return FastMSA(self.array, self.headers)

    def to_bytes(self) -> bytes:
        return _full_payload(self.array, self.headers)

    def to_sequence_bytes(self) -> bytes:
        """Serialize aligned sequences without their headers."""

        return _sequence_payload(self.array)

    def state_dict(self, json_serializable: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {"sequences": self.sequences}
        if self.deletions is not None:
            result["deletions"] = self.deletions.tolist() if json_serializable else self.deletions
        return result

    def _aligned_deletions(self) -> np.ndarray | None:
        if self.deletions is None:
            return None
        if self.deletions.shape != (self.depth, self.seqlen):
            return None
        return self.deletions

    def _select_deletion_columns(self, indices) -> np.ndarray | None:
        if self.deletions is None or self.deletions.shape[1] != self.seqlen:
            return None
        return self.deletions[:, indices]

    def select_sequences(
        self,
        indices: Sequence[int] | np.ndarray,
    ) -> MSA:
        deletions = None if self.deletions is None else self.deletions[np.asarray(indices)]
        return dataclasses.replace(
            self,
            entries=[self.entries[index] for index in indices],
            deletions=deletions,
        )

    def select_positions(
        self,
        indices: Sequence[int] | np.ndarray,
    ) -> MSA:
        entries = [
            FastaEntry(
                entry.header,
                "".join(entry.sequence[index] for index in indices),
            )
            for entry in self.entries
        ]
        return dataclasses.replace(
            self,
            entries=entries,
            deletions=self._select_deletion_columns(indices),
        )

    def __getitem__(
        self,
        indices: int | list[int] | slice | np.ndarray,
    ) -> MSA:
        column_indices = [indices] if isinstance(indices, int) else indices
        entries = [
            FastaEntry(
                entry.header,
                slice_any_object(entry.sequence, column_indices),
            )
            for entry in self.entries
        ]
        return dataclasses.replace(
            self,
            entries=entries,
            deletions=self._select_deletion_columns(column_indices),
        )

    def greedy_select(self, num_seqs: int, mode: str = "max") -> MSA:
        assert mode in ("max", "min")
        if self.depth <= num_seqs:
            return self
        return self.select_sequences(greedy_select_indices(self.array, num_seqs, mode))

    def hhfilter(
        self,
        seqid: int = 90,
        diff: int = 0,
        cov: int = 0,
        qid: int = 0,
        qsc: float = -20.0,
        binary: str = "hhfilter",
    ) -> MSA:
        indices = hhfilter(
            self.sequences,
            seqid=seqid,
            diff=diff,
            cov=cov,
            qid=qid,
            qsc=qsc,
            binary=binary,
        )
        return self.select_sequences(indices)

    def select_random_sequences(self, num_seqs: int) -> MSA:
        if num_seqs >= self.depth:
            return self
        return self.select_sequences(_random_row_indices(self.depth, num_seqs))

    def select_diverse_sequences(self, num_seqs: int) -> MSA:
        if num_seqs >= self.depth:
            return self
        filtered = self.hhfilter(diff=num_seqs)
        if num_seqs < filtered.depth:
            filtered = filtered.select_random_sequences(num_seqs)
        return filtered

    def pad_to_depth(self, depth: int) -> MSA:
        if depth < self.depth:
            raise ValueError(f"Cannot pad to depth {depth} when depth is {self.depth}")
        if depth == self.depth:
            return self
        count = depth - self.depth
        extra = [FastaEntry("", "-" * self.seqlen) for _ in range(count)]
        deletions = self._aligned_deletions()
        if deletions is not None:
            zero_rows = np.zeros((count, self.seqlen), dtype=deletions.dtype)
            deletions = np.concatenate((deletions, zero_rows), axis=0)
        return dataclasses.replace(
            self,
            entries=self.entries + extra,
            deletions=deletions,
        )

    @classmethod
    def stack(
        cls,
        msas: Sequence[MSA],
        remove_query_from_later_msas: bool = True,
    ) -> MSA:
        entries = []
        deletion_arrays = []
        for index, msa in enumerate(msas):
            start = 1 if index > 0 and remove_query_from_later_msas else 0
            entries.extend(msa.entries[start:])
            aligned = msa._aligned_deletions()
            if aligned is not None:
                deletion_arrays.append(aligned[start:])
        deletions = None
        if (
            len(deletion_arrays) == len(msas)
            and len({array.shape[1] for array in deletion_arrays}) == 1
        ):
            deletions = np.concatenate(deletion_arrays, axis=0)
        return cls(entries=entries, deletions=deletions)

    @classmethod
    def concat(
        cls,
        msas: Sequence[MSA],
        join_token: str | None = "|",
        allow_depth_mismatch: bool = False,
    ) -> MSA:
        if not msas:
            raise ValueError("Cannot concatenate an empty list of MSAs")
        depths = [msa.depth for msa in msas]
        if len(set(depths)) != 1:
            if not allow_depth_mismatch:
                raise ValueError("Depth mismatch in concatenating MSAs")
            maximum_depth = max(depths)
            msas = [msa.pad_to_depth(maximum_depth) for msa in msas]
        headers = [
            "|".join(str(header) for header in row)
            for row in zip(*(msa.headers for msa in msas), strict=False)
        ]
        separator = "" if join_token is None else join_token
        sequences = [
            separator.join(row) for row in zip(*(msa.sequences for msa in msas), strict=False)
        ]
        deletions = None
        if separator == "":
            arrays = [msa._aligned_deletions() for msa in msas]
            if all(array is not None for array in arrays):
                deletions = np.concatenate(arrays, axis=1)  # type: ignore[arg-type]
        return cls(
            [
                FastaEntry(header, sequence)
                for header, sequence in zip(headers, sequences, strict=False)
            ],
            deletions=deletions,
        )
