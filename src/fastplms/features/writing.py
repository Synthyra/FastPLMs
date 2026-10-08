"""Commit a window of rows to a feature store as one segment.

A pipeline that embeds a window of sequences at a time, and wants a run that dies to lose at most
that window, does the same thing after every window: keep the sequences the store lacks, write them
as a segment, and name the segment so a restart cannot commit the same rows under two names.
``write_rows`` is that step. The segment is named by the digest of its sequences, so the name
depends only on what the segment holds, and a window a restart repeats is skipped by calling
``FeatureStore.missing`` first, exactly as a run does before it embeds.
"""

from __future__ import annotations

import hashlib

from collections.abc import Mapping
from typing import Any

from .conversion import Row
from .store import FeatureStore, sequence_digest


DEFAULT_MAX_TENSOR_BYTES = 256 * 1024**2


def write_rows(
    store: FeatureStore,
    rows: Mapping[str, Row],
    *,
    metadata: Mapping[str, Any] | None = None,
    max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
) -> int:
    """Commit ``rows`` (sequence to row) as one segment and return how many rows it holds.

    A tensor for a dense or ragged feature, a ``SparseRow`` for csr, and a ``TopKRow`` for ragged
    top-k, as ``SegmentWriter.append`` takes them, in the order the mapping yields. An empty mapping
    writes nothing and returns zero. A sequence the store already holds raises, because a feature
    has one row per sequence; call ``store.missing`` first.

    ``metadata`` is plain data recorded in the segment's commit marker, for what the run wants to
    remember about these rows. A window too large for ``max_tensor_bytes`` is split into parts of a
    segment that commits or vanishes as one.
    """

    if not rows:
        return 0
    sequences = list(rows)
    digest = hashlib.sha256("\n".join(sequence_digest(sequence) for sequence in sequences).encode("utf-8"))
    with store.segment("rows-" + digest.hexdigest()[:16], metadata) as writer:
        writer.append_bounded(
            sequences, [rows[sequence] for sequence in sequences], max_tensor_bytes=max_tensor_bytes,
        )
    return len(sequences)
