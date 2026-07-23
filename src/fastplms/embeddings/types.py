"""Public value types for dataset embedding."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, overload

from torch import Tensor


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    """One named protein sequence supplied to :func:`embed_dataset`."""

    id: str
    sequence: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("EmbeddingInput.id must be a non-empty string.")
        if not isinstance(self.sequence, str) or not self.sequence:
            raise ValueError("EmbeddingInput.sequence must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class LazyTensorReference:
    """A tensor stored outside memory and loaded only when requested."""

    source: str
    key: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    _loader: Callable[[], Tensor] = field(repr=False, compare=False)

    def load(self, *, verify: bool = True) -> Tensor:
        """Load X and optionally verify its content digest."""

        if not isinstance(verify, bool):
            raise TypeError("verify must be a boolean.")
        X = self._loader()
        if not isinstance(X, Tensor):
            raise TypeError(f"Stored tensor loader for {self.key!r} must return a Tensor.")
        if tuple(X.shape) != self.shape:
            raise ValueError(
                f"Stored tensor {self.key!r} has shape {tuple(X.shape)}, expected {self.shape}."
            )
        dtype = str(X.dtype).removeprefix("torch.")
        if dtype != self.dtype:
            raise ValueError(
                f"Stored tensor {self.key!r} has dtype {dtype!r}, expected {self.dtype!r}."
            )
        if verify:
            from .storage import tensor_sha256

            digest = tensor_sha256(X)
            if digest != self.sha256:
                raise ValueError(f"Stored tensor {self.key!r} failed SHA-256 verification.")
        return X


TensorValue = Tensor | LazyTensorReference


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    """One ordered embedding result."""

    id: str
    sequence: str
    tensor: TensorValue

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("EmbeddingRecord.id must be a non-empty string.")
        if not isinstance(self.sequence, str) or not self.sequence:
            raise ValueError("EmbeddingRecord.sequence must be a non-empty string.")
        if not isinstance(self.tensor, (Tensor, LazyTensorReference)):
            raise TypeError("EmbeddingRecord.tensor must be a Tensor or LazyTensorReference.")

    def load_tensor(self, *, verify: bool = True) -> Tensor:
        """Return X regardless of whether this record is memory-backed or lazy."""

        if not isinstance(verify, bool):
            raise TypeError("verify must be a boolean.")
        if isinstance(self.tensor, LazyTensorReference):
            return self.tensor.load(verify=verify)
        return self.tensor


class EmbeddingResult(Sequence[EmbeddingRecord]):
    """Ordered embedding records and the metadata needed to reproduce them."""

    def __init__(
        self,
        records: Sequence[EmbeddingRecord],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.records: Sequence[EmbeddingRecord] = (
            records if getattr(records, "_fastplms_immutable_sequence", False) else tuple(records)
        )
        self.metadata = dict(metadata or {})

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[EmbeddingRecord]:
        return iter(self.records)

    @overload
    def __getitem__(self, index: int, /) -> EmbeddingRecord: ...

    @overload
    def __getitem__(self, index: slice, /) -> Sequence[EmbeddingRecord]: ...

    def __getitem__(self, index: int | slice) -> EmbeddingRecord | Sequence[EmbeddingRecord]:
        return self.records[index]

    def as_dict(
        self,
        *,
        key: Literal["id", "sequence"] = "id",
        duplicates: Literal["error", "first", "last"] = "error",
        materialize: bool = True,
    ) -> dict[str, TensorValue]:
        """Convert records to a mapping under an explicit duplicate policy."""

        if key not in {"id", "sequence"}:
            raise ValueError("key must be 'id' or 'sequence'.")
        if duplicates not in {"error", "first", "last"}:
            raise ValueError("duplicates must be 'error', 'first', or 'last'.")
        if not isinstance(materialize, bool):
            raise TypeError("materialize must be a boolean.")
        output: dict[str, TensorValue] = {}
        for record in self.records:
            record_key = getattr(record, key)
            if record_key in output:
                if duplicates == "error":
                    raise ValueError(
                        f"Duplicate {key} {record_key!r}; choose duplicates='first' "
                        "or duplicates='last' explicitly."
                    )
                if duplicates == "first":
                    continue
            output[record_key] = record.load_tensor() if materialize else record.tensor
        return output

    def materialize(self, *, verify: bool = True) -> EmbeddingResult:
        """Return an equivalent result with every X loaded into CPU memory."""

        if not isinstance(verify, bool):
            raise TypeError("verify must be a boolean.")
        return EmbeddingResult(
            [
                EmbeddingRecord(
                    id=record.id,
                    sequence=record.sequence,
                    tensor=record.load_tensor(verify=verify),
                )
                for record in self.records
            ],
            self.metadata,
        )


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Internal model-to-runner contract.

    ``X`` has shape ``(b, l, d)`` and ``residue_mask`` has shape ``(b, l)``.
    ``attentions`` may contain layer/head attention matrices for ``parti``.
    """

    X: Tensor
    residue_mask: Tensor
    attentions: Tensor | tuple[Tensor, ...] | None = None


__all__ = [
    "EmbeddingBatch",
    "EmbeddingInput",
    "EmbeddingRecord",
    "EmbeddingResult",
    "LazyTensorReference",
    "TensorValue",
]
