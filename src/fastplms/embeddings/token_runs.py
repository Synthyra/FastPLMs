"""Fill feature stores from canonical proteins with CLS and EOS kept, through the asynchronous writer.

``embed_token_features`` is the fast path of ``embed_into_features(keep_special_tokens=True)``. It takes a
protein inventory, embeds only the rows a stream lacks, and writes each tap's rows into the store of its key.
The device loop (``TokenTapExecutor``) and the disk loop (``AsyncFeatureWriter``) run concurrently, joined by a
bounded queue of pinned host buffers, so the model is not paused for hashing, compression, or fsync.

Per batch, every per-token stream holds l + 2 rows per sequence (row 0 CLS, rows 1..l residues, row l + 1
EOS) and every pooled stream averages or maximizes over those same l + 2 rows. A legacy residue-only store
holds l rows; this path never writes one.

Symbols: b sequences of a batch; l residues of a sequence after the N-terminal crop; n = sum(l_i + 2) token rows.
"""

from __future__ import annotations

import hashlib
import json
import torch

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from .batches import _temporary_eval
from .pooling import POOLING_SEMANTICS_TOKENS
from .taps import Tap, plan_taps
from .token_batches import (
    CANONICAL_MAX_RESIDUES, SPECIAL_TOKEN_ROWS, BatchGeometry, TokenTapExecutor, plan_geometry_batches, plan_token_batches,
)
from .tokens import ResidueVocabulary, check_canonical_text
from ..features.async_writer import AsyncFeatureWriter
from ..features.store import FeatureStore, SegmentReceipt, StoredFeature, sequence_digest


GIB = 1024**3
TOKEN_RUN_SCHEMA = "token_features_v1"
BATCH_ALGORITHM = "token_budget_length_sorted_v1"
# The token budget's defaults, for a run without a geometry.
DEFAULT_MAX_SEQUENCES = 256
DEFAULT_MAX_TOKENS = 32768
DEFAULT_WINDOW = 65536


class TokenFeatureContract(Protocol):
    """What a token run asks of a scientific contract: validate it, name each row, and recheck before a commit."""

    def validate(
        self, model: Any, sequences: Sequence[str], features: Mapping[str, StoredFeature],
        taps: Sequence[Tap], options: Mapping[str, Any],
    ) -> None: ...

    def validate_cached(self, name: str, store: FeatureStore, sequences: Sequence[str]) -> None: ...

    def row_identities(
        self, name: str, sequences: Sequence[str], digests: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]: ...

    def check_row_keys(self, sequences: Sequence[str], digests: Sequence[str]) -> None:
        """Raise unless each digest is the key the contract holds for its sequence."""

    def before_commit(self) -> None: ...


def distinct_with_digests(
    sequences: Sequence[str], digests: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    """Distinct sequences in first-seen order and their row keys, hashing each exact text once.

    ``digests`` may supply the keys when the caller already verified them against the text.
    """
    if digests is not None and len(digests) != len(sequences):
        raise ValueError("digests must hold one key per sequence.")
    seen: set[str] = set()
    ordered: list[str] = []
    keys: list[str] = []
    for position, sequence in enumerate(sequences):
        if sequence in seen:
            continue
        seen.add(sequence)
        ordered.append(sequence)
        keys.append(sequence_digest(sequence) if digests is None else digests[position])
    return ordered, keys


def embed_token_features(
    model: Any,
    sequences: Sequence[str],
    root: str | Path,
    features: Mapping[str, StoredFeature],
    *,
    taps: Sequence[Tap],
    contract: TokenFeatureContract | None = None,
    digests: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    max_residues: int | None = CANONICAL_MAX_RESIDUES,
    max_sequences: int | None = None,
    max_tokens: int | None = None,
    window: int | None = None,
    dtype: torch.dtype | None = None,
    fixed_batch_size: int | None = None,
    geometry: BatchGeometry | None = None,
    part_bytes: int = GIB,
    segment_bytes: int = 8 * GIB,
    queue_bytes: int = 4 * GIB,
    workers: int = 4,
    verify_cached: bool = True,
    model_state_fingerprint: str | None = None,
    progress: Callable[[int], None] | None = None,
    on_plan: Callable[[int], None] | None = None,
    batch_watch: Callable[[Iterator[tuple[int, ...]]], Iterator[tuple[int, ...]]] | None = None,
) -> dict[str, tuple[SegmentReceipt, ...]]:
    """Fill each named feature under ``root`` from one pass over the sequences it lacks, special tokens kept.

    ``features`` maps a tap name to its feature and names every tap. ``max_residues`` is the N-terminal
    crop (``CANONICAL_MAX_RESIDUES`` keeps a sequence within 2048 tokens). Batches hold at most ``max_sequences`` sequences and
    ``max_tokens`` padded token rows, sorted by length inside windows of ``window`` sequences (256, 32768 and 65536 when
    None). A ``geometry`` instead runs every sequence at its bucket's fixed shape (``BatchGeometry``), whatever batch it
    falls in, and takes none of those three nor ``fixed_batch_size``. Parts are
    about ``part_bytes``, a segment commits every ``segment_bytes`` across all streams, and at most
    ``queue_bytes`` of finished rows wait for the writer. ``digests`` are the rows' SHA-256 keys when the
    caller already has them; otherwise each text is hashed once here. ``verify_cached=False`` skips the
    contract's re-read of rows already stored, which a resumed run of a large store does separately.
    ``progress`` receives the sequences of each batch once its rows are packed for writing, and ``on_plan`` the number of
    sequences this run will embed (fewer than ``sequences`` on a resume), before the first batch.

    Returns the committed segments of each stream that gained rows, oldest first; an empty result means the
    model never ran. A killed run commits whole segments only, so a rerun resumes from the last one.
    """
    names = {tap.name for tap in taps}
    if geometry is not None:
        if any(value is not None for value in (max_sequences, max_tokens, window, fixed_batch_size)):
            raise ValueError(
                "A geometry run takes its batch shapes from the geometry; pass no max_sequences, max_tokens, window "
                "or fixed_batch_size."
            )
        if max_residues is None or max_residues + SPECIAL_TOKEN_ROWS > geometry.max_columns:
            raise ValueError("A geometry run needs a crop whose l + 2 tokens fit the geometry's widest bucket.")
    else:
        max_sequences = DEFAULT_MAX_SEQUENCES if max_sequences is None else max_sequences
        max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        window = DEFAULT_WINDOW if window is None else window
    if fixed_batch_size is not None:
        if fixed_batch_size != max_sequences or max_residues is None or max_tokens < fixed_batch_size * (max_residues + 2):
            raise ValueError("Fixed batches require matching max_sequences and a token budget covering the full cropped context.")
    if set(features) != names:
        raise ValueError(
            "features must name exactly the taps this run takes.\n"
            f"  taps:     {sorted(names)}\n  features: {sorted(features)}"
        )
    if any(spec.positions for spec in features.values()):
        raise ValueError("A token run stores no argmax positions.")
    if contract is None and any(spec.descriptor.get("schema") is not None for spec in features.values()):
        raise ValueError("A descriptor that carries a schema (feature_spec_v1, v2 or v3) requires its contract.")
    if getattr(contract, "keep_special_tokens", None) is False:
        raise ValueError("A token run needs a contract captured with the special tokens kept.")
    ordered, keys = distinct_with_digests(sequences, digests)
    if not ordered:
        raise ValueError("embed_token_features needs at least one sequence.")
    # A row is keyed by the hash of its text, so the text must be the normalized one (uppercase, no whitespace):
    # a second spelling of a protein would otherwise become a second row. Checked for all before any forward.
    for sequence in ordered:
        check_canonical_text(sequence)
    if contract is not None:
        contract.check_row_keys(ordered, keys)  # a dict lookup per row: a caller's digest is never trusted
        # The contract compares these with the options it measured; its per-sequence checks are the caller's.
        extraction_options: dict[str, Any] = {"max_length": max_residues, "truncate": True, "dtype": dtype}
        if geometry is not None:
            extraction_options["geometry"] = geometry.describe()
        else:
            extraction_options.update(batch_size=max_sequences, batch_window_size=window, max_tokens_per_batch=max_tokens)
        if fixed_batch_size is not None:
            extraction_options["fixed_batch_size"] = fixed_batch_size
        contract.validate(model, (), features, taps, extraction_options)
    stores = {name: FeatureStore.open(root, spec, deep_verify=False) for name, spec in features.items()}
    # Each stream's missing rows come from one index pass over the keys, never a second hash of the text.
    wanted = {name: frozenset(set(keys) - store.present_digests(keys)) for name, store in stores.items()}
    if verify_cached and contract is not None:
        for name, store in stores.items():
            cached = [sequence for sequence, key in zip(ordered, keys, strict=True) if key not in wanted[name]]
            if cached:
                contract.validate_cached(name, store, cached)
    chosen = [position for position, key in enumerate(keys) if any(key in group for group in wanted.values())]
    if not chosen:
        return {}
    texts = [ordered[position] for position in chosen]
    text_keys = [keys[position] for position in chosen]
    if on_plan is not None:
        on_plan(len(texts))
    plan = plan_taps(list(taps), int(model.embedding_tap_state_count))
    executor = TokenTapExecutor(
        model, plan, vocabulary=ResidueVocabulary(model.tokenizer), max_residues=max_residues, dtype=dtype,
        fixed_batch_size=fixed_batch_size, geometry=geometry,
    )
    lengths = [len(executor.crop(text)) for text in texts]  # (count,) residues l after the crop
    batch_policy: dict[str, Any] = dict(geometry.describe()) if geometry is not None else {
        "algorithm": BATCH_ALGORITHM, "max_sequences": max_sequences, "max_tokens": max_tokens, "window": window,
    }
    run = {
        "schema": TOKEN_RUN_SCHEMA, "special_tokens": "kept", "max_residues": max_residues,
        "batch_policy": batch_policy,
        "pooling_semantics": dict(POOLING_SEMANTICS_TOKENS), "model_state_fingerprint": model_state_fingerprint,
        "storage_policy": {"max_part_bytes": part_bytes, "segment_bytes": segment_bytes},
    }
    if fixed_batch_size is not None:
        run["batch_policy"].update(algorithm="fixed_rows_duplicate_pad_v1", fixed_batch_size=fixed_batch_size)
    # Each stream's own missing rows name the segment. A kill after one stream committed leaves that stream
    # with fewer missing rows, so the rerun gets new segment names and never collides with the committed one.
    wanted_digests = {
        features[name].key: hashlib.sha256("".join(sorted(group)).encode("ascii")).hexdigest()
        for name, group in wanted.items()
    }
    fingerprint = hashlib.sha256(json.dumps({"run": run, "wanted": wanted_digests}, sort_keys=True)
                                 .encode("utf-8")).hexdigest()[:32]

    def identities(stream: str, batch_texts: Sequence[str], batch_keys: Sequence[str]) -> Sequence[Mapping[str, Any]]:
        if contract is None:
            return [{} for _ in batch_texts]
        # Identities bind the original text, whose length the crop policy turns into l; never the cropped text.
        return contract.row_identities(stream, batch_texts, batch_keys)

    writer = AsyncFeatureWriter(
        stores, fingerprint=fingerprint, metadata={**dict(metadata or {}), **run}, wanted=wanted,
        row_records=identities, part_bytes=part_bytes, segment_bytes=segment_bytes, queue_bytes=queue_bytes,
        before_commit=None if contract is None else contract.before_commit, workers=workers, progress=progress,
    )
    try:
        with _inference(model):
            if geometry is not None:
                batches = plan_geometry_batches(lengths, text_keys, geometry)
            else:
                batches = plan_token_batches(lengths, max_sequences=max_sequences, max_tokens=max_tokens, window=window)
            if batch_watch is not None:
                batches = batch_watch(batches)
            for members in batches:
                writer.submit(executor.run_batch([texts[i] for i in members], [text_keys[i] for i in members]))
    except BaseException:
        writer.abort()
        raise
    receipts = writer.close()
    return {name: tuple(group) for name, group in receipts.items() if group}


@contextmanager
def _inference(model: Any) -> Iterator[None]:
    """Evaluation mode and inference mode for the whole loop, restored afterwards."""
    with _temporary_eval(model), torch.inference_mode():
        yield


__all__ = ["BATCH_ALGORITHM", "TOKEN_RUN_SCHEMA", "TokenFeatureContract", "distinct_with_digests", "embed_token_features"]
