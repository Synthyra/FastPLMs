"""Run a tap plan and persist each tap into its own feature store.

This is the one place a model runs to fill the store. ``embed_into_features`` embeds only the
sequences the stores lack, takes one forward pass per batch for every tap, and writes each tap's
rows into the store of its key. A second call with the same sequences runs no model at all.

Each tap becomes one feature, so a run that taps the last hidden state, a mean-pooled vector, and
max-pooled sparse-autoencoder codes fills three stores from one pass. The store's layout decides
how a tap's tensor is stored: a pooled vector goes in dense, per-residue rows go in ragged, and a
sparse-autoencoder vector goes in csr, compressed to its exactly non-zero codes.

The caller owns the keys, because the key composes the model, its revision, the autoencoder, the
layer, the pooling, the dtype, and the residue limit, and only the caller knows the pinned
revisions. ``foundry.embedding.feature_key`` computes them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Protocol
from torch import Tensor

from .runner import embed_dataset
from .taps import Tap
from .types import TapRecord, TapRunReceipt
from ..features.layouts import DENSE, RAGGED, RAGGED_TOPK, SparseRow, TopKRow
from ..features.store import FeatureStore, SegmentReceipt, SegmentWriter, StoredFeature


class FeatureRunContract(Protocol):
    """Scientific identity policy supplied by the caller, independent of the storage format."""

    def validate(
        self, model: Any, sequences: Sequence[str], features: Mapping[str, StoredFeature],
        taps: Sequence[Tap], options: Mapping[str, Any],
    ) -> None: ...

    def validate_cached(self, name: str, store: FeatureStore, sequences: Sequence[str]) -> None: ...

    def bind_rows(self, name: str, records: Sequence[TapRecord]) -> Sequence[Mapping[str, Any]]: ...

    def before_commit(self) -> None: ...


def embed_into_features(
    model: Any,
    sequences: Sequence[str],
    root: str | Path,
    features: Mapping[str, StoredFeature],
    *,
    taps: Sequence[Tap],
    metadata: Mapping[str, Any] | None = None,
    contract: FeatureRunContract | None = None,
    max_part_bytes: int = 256 * 1024**2,
    keep_special_tokens: bool = False,
    **embed_kwargs: Any,
) -> dict[str, SegmentReceipt]:
    """Fill each named feature under ``root`` from one pass over the sequences it lacks.

    ``features`` maps a tap name to the spec of the feature it fills, and must name every tap.
    ``metadata`` is recorded on every segment this run commits, beside the run fingerprint.
    ``max_part_bytes`` bounds each part's encoded tensor payload, excluding its small safetensors
    header and row-identity sidecar. A single row exceeding it fails without being committed.
    Remaining keyword arguments go to ``embed_dataset``. Tensor outputs are streamed one bounded
    batch window at a time; input identities and part metadata still scale with inventory size.

    A contract that keeps the special tokens (``contract.keep_special_tokens`` is True) selects the canonical
    token path by itself, so a caller cannot forget the argument; a contract that keeps residues only (False)
    refuses ``keep_special_tokens=True``.

    ``keep_special_tokens=True`` is the canonical token path: every per-token stream holds l + 2 rows per
    protein (row 0 CLS, rows 1..l residues, row l + 1 EOS, shape (n, d) with n = sum(l_i + 2) overall), every
    pooled stream covers those same rows, and the run goes through ``embed_token_features`` and its
    asynchronous writer. Its ``embed_kwargs`` are the contract's ``embedding_options`` (``max_length`` is the
    crop in residues), and it returns the last committed segment of each stream; call ``embed_token_features``
    for every segment.

    Returns the committed segment of each feature that gained rows. A feature whose sequences were
    all present is absent from the result, and an empty result means the model never ran.
    """

    # The contract names the stored layout: l + 2 rows (True), l rows (False), or no claim (no attribute).
    contract_keeps = getattr(contract, "keep_special_tokens", None)
    if contract_keeps is True:
        keep_special_tokens = True
    elif contract_keeps is False and keep_special_tokens:
        raise ValueError("keep_special_tokens=True needs a contract captured with the special tokens kept.")

    if keep_special_tokens:
        from .token_batches import CANONICAL_MAX_RESIDUES
        from .token_runs import embed_token_features

        options = dict(embed_kwargs)
        settings: dict[str, Any] = {
            "max_residues": options.pop("max_length", CANONICAL_MAX_RESIDUES),
            "max_sequences": options.pop("batch_size", 256),
            "max_tokens": options.pop("max_tokens_per_batch", None) or 32768,
            "window": options.pop("batch_window_size", 65536),
            "dtype": options.pop("dtype", None),
        }
        options.pop("truncate", None)  # the canonical crop is always a prefix crop
        if "fixed_batch_size" in options:
            settings["fixed_batch_size"] = options.pop("fixed_batch_size")
        if options:
            raise ValueError(f"keep_special_tokens takes no other extraction options; received {sorted(options)}.")
        received = embed_token_features(
            model, sequences, root, features, taps=taps, contract=contract, metadata=metadata,
            part_bytes=max_part_bytes, **settings,
        )
        return {name: group[-1] for name, group in received.items()}

    if type(max_part_bytes) is not int or max_part_bytes <= 0:
        raise ValueError("max_part_bytes must be a positive integer.")
    if "tap_sink" in embed_kwargs:
        raise ValueError("embed_into_features owns its tap_sink destination.")
    tap_names = [tap.name for tap in taps]
    if set(features) != set(tap_names):
        raise ValueError(
            "features must name exactly the taps this run takes.\n"
            f"  taps:     {sorted(tap_names)}\n"
            f"  features: {sorted(features)}"
        )
    for name, spec in features.items():
        if spec.positions:
            raise ValueError(
                f"Feature {spec.key!r} stores the argmax residue of each code, and no tap carries "
                f"them, so this run cannot fill it from tap {name!r}. A pipeline that computes "
                "positions itself writes them through the store's own segment writer."
            )

    ordered = _distinct(sequences)
    if not ordered:
        raise ValueError("embed_into_features needs at least one sequence.")
    if contract is None and any(
        spec.descriptor.get("schema") == "feature_spec_v1" for spec in features.values()
    ):
        raise ValueError("Complete feature descriptors require a FeatureRunContract.")
    if contract is not None:
        contract.validate(model, ordered, features, taps, embed_kwargs)
    stores = {name: FeatureStore.open(root, spec) for name, spec in features.items()}
    wanted = {name: frozenset(store.missing(ordered)) for name, store in stores.items()}
    if contract is not None:
        for name, store in stores.items():
            contract.validate_cached(name, store, [s for s in ordered if s not in wanted[name]])
    to_embed = [sequence for sequence in ordered if any(sequence in group for group in wanted.values())]
    if not to_embed:
        return {}

    receipts: dict[str, SegmentReceipt] = {}
    with ExitStack() as stack:
        writers: dict[str, SegmentWriter] = {}
        # Recheck live dependencies and caller descriptors after all windows have been staged.
        check = None if contract is None else lambda: contract.validate(
            model, (), features, taps, embed_kwargs,
        )

        def append_window(records: Sequence[TapRecord], identity: Mapping[str, str]) -> None:
            for name, store in stores.items():
                embedded = [record for record in records if record.sequence in wanted[name]]
                if not embedded:
                    continue
                rows = [_row_for(store.spec, record, name) for record in embedded]
                identities = None if contract is None else contract.bind_rows(name, embedded)
                if name not in writers:
                    run_metadata = {
                        **dict(metadata or {}), "input_fingerprint": identity["input_fingerprint"],
                        "storage_policy": {"max_part_tensor_bytes": max_part_bytes},
                    }
                    writers[name] = stack.enter_context(store.segment(
                        identity["run_fingerprint"], run_metadata, before_commit=check,
                    ))
                writers[name].append_bounded(
                    [record.sequence for record in embedded], rows, row_metadata=identities,
                    max_tensor_bytes=max_part_bytes,
                )

        run_receipt = embed_dataset(
            model, to_embed, taps=list(taps), require_residue_identity=contract is not None,
            tap_sink=append_window, **embed_kwargs,
        )
        if not isinstance(run_receipt, TapRunReceipt) or run_receipt.record_count != len(to_embed):
            raise TypeError("Feature extraction did not complete delivery of every requested row.")
        for name, writer in writers.items():
            receipts[name] = writer.commit()
    return receipts


def _row_for(spec: StoredFeature, record: TapRecord, tap: str) -> Tensor | SparseRow | TopKRow:
    """One tap's output for one sequence, in the shape its store stores."""

    value = record.tensors[tap]  # (w,) pooled, (r_i, d) per residue, or a TopKRow of (r_i, k) tensors
    if spec.layout == RAGGED_TOPK:
        if not isinstance(value, TopKRow):
            raise ValueError(f"Feature {spec.key!r} needs sparse TopKRow output from tap {tap!r}.")
        return value  # TopKRow of (r_i, k) indices and values
    if not isinstance(value, Tensor):
        raise ValueError("Sparse residue tap output requires a ragged_topk feature.")
    if spec.layout == DENSE:
        if value.ndim != 1:
            raise ValueError(
                f"Feature {spec.key!r} is dense, so tap {tap!r} must give one vector per sequence; "
                f"received shape {tuple(value.shape)}. A per-residue tap needs a ragged feature."
            )
        return value  # (w,)
    if spec.layout == RAGGED:
        if value.ndim != 2:
            raise ValueError(
                f"Feature {spec.key!r} is ragged, so tap {tap!r} must give (r_i, d) residue rows; "
                f"received shape {tuple(value.shape)}."
            )
        return value  # (r_i, d)
    if value.ndim != 1:
        raise ValueError(
            f"Feature {spec.key!r} is csr, so tap {tap!r} must give one vector per sequence; "
            f"received shape {tuple(value.shape)}."
        )
    return SparseRow.from_dense(value)  # SparseRow of (nnz,) tensors, from a (w,) vector


def _distinct(sequences: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for sequence in sequences:
        if sequence not in seen:
            seen.add(sequence)
            ordered.append(sequence)
    return ordered


__all__ = ["embed_into_features"]
