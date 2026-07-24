"""ESMFold2 integration for the shared FastPLMs embedding API."""

from __future__ import annotations

import torch
from typing import Any, ClassVar
from torch import Tensor

from ...embeddings import EmbeddingBatch, EmbeddingResult, embed_dataset
from .esmfold2_constants_esm3 import SEQUENCE_PAD_TOKEN, SEQUENCE_VOCAB


_TOKEN_TO_ID = {token: index for index, token in enumerate(SEQUENCE_VOCAB)}
_VALID_RESIDUES = frozenset(SEQUENCE_VOCAB[4:31]) - {".", "-", "|"}


def _encode_single_chain(sequence: str) -> list[int]:
    normalized = sequence.upper()
    if not normalized:
        raise ValueError("ESMFold2 dataset embedding requires at least one protein residue.")
    invalid = sorted(set(normalized) - _VALID_RESIDUES)
    if invalid:
        raise ValueError(
            "ESMFold2 dataset embedding accepts one ungapped protein chain; "
            f"invalid symbols: {invalid}."
        )
    return [_TOKEN_TO_ID[residue] for residue in normalized]


class ESMFold2EmbeddingMixin:
    """Learned ESMC sequence summaries for ESMFold2 models."""

    embedding_unsupported_pooling = frozenset({"cls", "parti"})
    embedding_layer = "all_81_esmc_states"
    embedding_projection = "esmfold2_learned_sequence_summary"
    embedding_token_policy: ClassVar[dict[str, object]] = {
        "unit": "residue",
        "normalization": "uppercase",
        "include": ["single-chain protein residues"],
        "exclude": [
            "BOS",
            "EOS",
            "padding",
            "chain delimiters",
            "non-protein tokens",
        ],
    }

    def project_esmc_hidden_states(
        self,
        hidden_states: Tensor,
        residue_mask: Tensor | None = None,
    ) -> Tensor:
        """Project H from ``(b, l, 81, 2560)`` to Z with shape ``(b, l, 256)``."""

        # hidden_states: (b, l, 81, d_model); residue_mask: (b, l) or None.
        # d_z is the learned projection width: 256 for released ESMFold2 checkpoints.
        if hidden_states.ndim != 4 or hidden_states.shape[-2] != 81:
            raise ValueError(
                "ESMFold2 projection requires the official ordered 81-state "
                "ESMC tensor H with shape (b, l, 81, d_model)."
            )
        return self.language_model.project_sequence(
            hidden_states, residue_mask
        )  # (b, l, d_z)

    def _embedding_batch(self, sequences: list[str], **kwargs: Any) -> EmbeddingBatch:
        if kwargs:
            raise TypeError(f"Unexpected ESMFold2 embedding options: {', '.join(sorted(kwargs))}.")
        if self._esmc is None:
            raise RuntimeError("ESMFold2 embeddings require load_esmc=True.")
        encoded = [_encode_single_chain(sequence) for sequence in sequences]
        sequence_length = max(map(len, encoded))  # l
        b = len(encoded)  # b
        device = self.device
        input_ids = torch.full(
            (b, sequence_length),
            SEQUENCE_PAD_TOKEN,
            dtype=torch.long,
            device=device,
        )  # (b, l)
        residue_mask = torch.zeros(
            (b, sequence_length), dtype=torch.bool, device=device
        )  # (b, l)
        for batch_index, token_ids in enumerate(encoded):
            length = len(token_ids)  # l_i
            input_ids[batch_index, :length] = torch.tensor(
                token_ids, dtype=torch.long, device=device
            )  # (l_i,) -> input_ids[batch_index, :l_i]: (l_i,)
            residue_mask[batch_index, :length] = True  # (l_i,)

        residue_index = torch.arange(sequence_length, device=device).expand(b, -1)  # (b, l)
        asym_id = torch.zeros_like(input_ids)  # (b, l)
        mol_type = torch.zeros_like(input_ids)  # (b, l)
        hidden_states = self._compute_lm_hidden_states(
            input_ids,
            asym_id,
            residue_index,
            mol_type,
            residue_mask,
        )  # (b, l, 81, d_model)
        projected = self.project_esmc_hidden_states(
            hidden_states, residue_mask
        )  # (b, l, d_z)
        return EmbeddingBatch(
            X=projected, residue_mask=residue_mask
        )  # X: (b, l, d_z); residue_mask: (b, l)

    def embed_dataset(self, inputs: Any, **kwargs: Any) -> EmbeddingResult:
        """Embed single-chain proteins using the learned 256-wide ESMFold2 summary."""

        return embed_dataset(self, inputs, **kwargs)


__all__ = ["ESMFold2EmbeddingMixin"]
