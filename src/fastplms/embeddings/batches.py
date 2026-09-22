"""Execute model-specific batches and return ordered residue-aware CPU tensors."""

from __future__ import annotations

import torch
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from torch import Tensor

from .identity import _model_device
from .inputs import _planned_batches
from .pooling import Pooler
from .types import EmbeddingBatch, EmbeddingInput, EmbeddingRecord


_MAX_PARTI_RESIDUES = 2_048


def _validate_parti_length(M: Tensor) -> None:
    """Reject an oversized attention graph before model inference."""

    # M: (b, l)
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
    # last_hidden_state and each hidden_states entry: (b, l, d)
    if store_all_hidden_states:
        if not hidden_states:
            raise ValueError("store_all_hidden_states requires model hidden states.")
        # H has shape (b, n, l, d), where n follows the model's output order.
        return torch.stack(hidden_states, dim=1)  # (b, n, l, d)
    if hidden_state_index == -1:
        return last_hidden_state  # (b, l, d)
    if not hidden_states:
        raise ValueError("hidden_state_index requires model hidden states.")
    return hidden_states[hidden_state_index]  # (b, l, d)


def _residue_embeddings(X: Tensor, M: Tensor) -> list[Tensor]:
    """Copy every sample's biological residues to the host in one transfer.

    Boolean indexing packs the selected rows in batch order, so splitting the
    packed rows by residue count gives the values that indexing each sample
    would. Each returned tensor owns its storage, as a per-sample copy does.
    """
    # X: (b, l, d); M: (b, l)
    residue_counts = M.sum(dim=1).tolist()  # b counts r_i
    packed = X[M].detach().cpu()  # (sum of r_i, d)
    return [sample.clone() for sample in torch.split(packed, residue_counts)]  # each: (r_i, d)


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


def _biological_residue_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    tokenizer: Any,
) -> Tensor:
    """Remove padding and tokenizer-declared special tokens from M."""

    # input_ids, attention_mask: (b, l)
    M = attention_mask.to(dtype=torch.bool)  # (b, l)
    special_ids = tuple(int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ()))
    if special_ids:
        specials = torch.tensor(  # (n_special,)
            special_ids,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        M = M & ~torch.isin(input_ids, specials)  # (b, l)
    return M  # (b, l)


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
        X, M = output  # (b, l, d), (b, l)
        preparer = getattr(model, "prep_tokens", None)
        if preparer is not None and hasattr(preparer, "get_batch_kwargs"):
            prepared = preparer.get_batch_kwargs(sequences, device=X.device)
            input_ids = prepared["input_ids"]  # (b, l)
            boundary_ids = preparer.boundary_token_ids.to(  # (n_boundary,)
                device=input_ids.device, dtype=input_ids.dtype
            )
            # E1 wraps each raw sequence in BOS, context-label, terminal-label,
            # and EOS tokens. Only amino-acid rows are biological residues.
            M = M.to(dtype=torch.bool) & ~torch.isin(input_ids, boundary_ids)  # (b, l)
        if need_attentions:
            raise ValueError("parti is not available for tokenizer-free E1 embedding.")
        return EmbeddingBatch(  # X: (b, l, d); residue_mask: (b, l)
            X=X,
            residue_mask=M.to(dtype=torch.bool),
        )
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
    input_ids = encoded["input_ids"].to(device)  # (b, l)
    attention_mask = encoded.get(  # (b, l)
        "attention_mask",
        input_ids.new_ones(input_ids.shape),
    ).to(device)
    M = _biological_residue_mask(input_ids, attention_mask, tokenizer)  # (b, l)
    if need_attentions:
        # Validate l before either the backbone or its quadratic attention graph
        # is materialized. M has shape (b, l).
        _validate_parti_length(M)
    X = model._embed(input_ids, attention_mask, **model_kwargs)  # (b, l, d)
    attentions = None
    if need_attentions:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        attentions = getattr(output, "attentions", None)  # each: (b, h, l, l)
        if attentions is None:
            raise ValueError("The model did not return attentions required by parti.")
    return EmbeddingBatch(  # X: (b, l, d); M: (b, l)
        X=X,
        residue_mask=M,
        attentions=attentions,
    )


@dataclass(eq=False)
class BatchExecutor:
    """Model and batch policy for one bounded embedding window at a time."""

    model: Any
    batch_size: int
    max_tokens_per_batch: int | None
    max_length: int | None
    truncate: bool
    model_kwargs: dict[str, Any]
    hidden_state_source: str
    normalized_decoder_inputs: tuple[str, ...] | None
    decoder_input_ids: Tensor | None
    decoder_attention_mask: Tensor | None
    _embedding_batch_fn: Callable[..., EmbeddingBatch] | None
    tokenizer: Any | None
    store_all_hidden_states: bool
    full_embeddings: bool
    dtype: torch.dtype | None
    pooler: Pooler | None
    attention_backend: str | None
    need_attentions: bool
    model_type: str = field(init=False)
    resolved_tokenizer: Any = field(init=False)

    def __post_init__(self) -> None:
        config = getattr(self.model, "config", None)
        self.model_type = str(getattr(config, "model_type", "")).lower()
        self.resolved_tokenizer = (
            self.tokenizer if self.tokenizer is not None else getattr(self.model, "tokenizer", None)
        )

    def run_window(
        self,
        window_records: Sequence[EmbeddingInput],
        *,
        window_start: int,
    ) -> tuple[list[EmbeddingRecord], dict[str, tuple[int, int]]]:
        """Restore source order after length-bucketed inference and pooling."""

        pool_slices: dict[str, tuple[int, int]] = {}
        window_results: dict[int, EmbeddingRecord] = {}
        for local_positions in _planned_batches(
            window_records,
            range(len(window_records)),
            batch_size=self.batch_size,
            max_tokens_per_batch=self.max_tokens_per_batch,
            max_length=self.max_length,
            truncate=self.truncate,
        ):
            batch_positions = [window_start + position for position in local_positions]
            batch_records = [window_records[position] for position in local_positions]
            sequences = [
                record.sequence[: self.max_length]
                if self.truncate and self.max_length is not None
                else record.sequence
                for record in batch_records
            ]
            batch_model_kwargs = dict(self.model_kwargs)
            if self.model_type == "fast_ankh" or self.hidden_state_source == "decoder":
                batch_model_kwargs["hidden_state_source"] = self.hidden_state_source
                if self.normalized_decoder_inputs is not None:
                    batch_model_kwargs["decoder_inputs"] = [
                        self.normalized_decoder_inputs[position] for position in batch_positions
                    ]
                if self.decoder_input_ids is not None:
                    # decoder_input_ids: (n_records, l_decoder)
                    indices = torch.tensor(  # (b,)
                        batch_positions,
                        device=self.decoder_input_ids.device,
                        dtype=torch.long,
                    )
                    batch_model_kwargs["decoder_input_ids"] = (  # (b, l_decoder)
                        self.decoder_input_ids.index_select(0, indices)
                    )
                if self.decoder_attention_mask is not None:
                    # decoder_attention_mask: (n_records, l_decoder)
                    indices = torch.tensor(  # (b,)
                        batch_positions,
                        device=self.decoder_attention_mask.device,
                        dtype=torch.long,
                    )
                    batch_model_kwargs["decoder_attention_mask"] = (
                        self.decoder_attention_mask.index_select(0, indices)  # (b, l_decoder)
                    )
            custom_batch = self._embedding_batch_fn or getattr(self.model, "_embedding_batch", None)
            if custom_batch is not None:
                if self.model_type == "fast_ankh":
                    batch = custom_batch(
                        sequences,
                        tokenizer=self.resolved_tokenizer,
                        max_length=self.max_length,
                        truncate=self.truncate,
                        need_attentions=self.need_attentions,
                        **batch_model_kwargs,
                    )
                else:
                    batch = custom_batch(sequences, **batch_model_kwargs)
                if not isinstance(batch, EmbeddingBatch):
                    raise TypeError("_embedding_batch must return EmbeddingBatch.")
            else:
                batch = _generic_embedding_batch(
                    self.model,
                    sequences,
                    tokenizer=self.tokenizer,
                    max_length=self.max_length,
                    truncate=self.truncate,
                    need_attentions=self.need_attentions,
                    model_kwargs=batch_model_kwargs,
                )
            X = batch.X  # (b, l, d) or (b, n_states, l, d)
            raw_mask = batch.residue_mask  # (b, l)
            if not isinstance(X, Tensor) or not isinstance(raw_mask, Tensor):
                raise TypeError("Embedding batches must provide Tensor X and residue_mask.")
            if X.is_meta or raw_mask.is_meta:
                raise ValueError("Embedding batches cannot contain meta tensors.")
            if not X.is_floating_point():
                raise TypeError("Embedding batches must use a floating-point X dtype.")
            if raw_mask.is_complex() or not bool(torch.isfinite(raw_mask).all()):
                raise ValueError("Embedding residue_mask must contain finite binary values.")
            if not bool(((raw_mask == 0) | (raw_mask == 1)).all()):
                raise ValueError("Embedding residue_mask must contain finite binary values.")
            M = raw_mask.to(device=X.device, dtype=torch.bool)  # (b, l)
            valid_X_shape = (
                X.ndim == 3
                and X.shape[0] == len(batch_records)
                and X.shape[-1] > 0
                and M.shape == X.shape[:2]
            )
            valid_all_states_shape = (
                X.ndim == 4
                and self.store_all_hidden_states
                and self.full_embeddings
                and X.shape[0] == len(batch_records)
                and X.shape[1] > 0
                and X.shape[-1] > 0
                and M.shape == (X.shape[0], X.shape[2])
            )
            if not (valid_X_shape or valid_all_states_shape):
                raise ValueError(
                    "Embedding batches must provide X with shape (b, l, d), or "
                    "(b, states, l, d) when storing all hidden states, and "
                    "residue_mask with shape (b, l)."
                )
            if not bool(M.any(dim=1).all()):
                raise ValueError("Every embedding sample must contain a biological residue.")
            finite_selected = (  # X.shape
                torch.isfinite(X) | ~M.unsqueeze(-1)
                if X.ndim == 3
                else torch.isfinite(X) | ~M[:, None, :, None]
            )
            if not bool(finite_selected.all()):
                raise ValueError("Biological residue embeddings produced non-finite output.")
            if self.need_attentions:
                # Validate the biological graph only after mask integrity is established.
                _validate_parti_length(M)
            if self.dtype is not None:
                X = X.to(dtype=self.dtype)  # unchanged shape

            if self.full_embeddings:
                if X.ndim == 4:
                    values = [
                        X_i[:, M_i, :].detach().cpu()  # (n_states, r_i, d)
                        for X_i, M_i in zip(X, M, strict=True)
                    ]
                else:
                    values = _residue_embeddings(X, M)  # each: (r_i, d)
            else:
                if self.pooler is None:
                    raise RuntimeError(
                        "Pooled embedding output was requested without an initialized pooler."
                    )
                Y = self.pooler(  # (b, n_poolers * d)
                    X,
                    M,
                    attentions=batch.attentions,
                    attention_backend=self.attention_backend,
                )
                pool_slices = self.pooler.output_slices(X.shape[-1])
                values = list(Y.detach().cpu().unbind(0))  # each: (n_poolers * d,)
            for position, record, value in zip(batch_positions, batch_records, values, strict=True):
                window_results[position] = EmbeddingRecord(record.id, record.sequence, value)

        new_records = [
            window_results[position]
            for position in range(window_start, window_start + len(window_records))
        ]
        return new_records, pool_slices
