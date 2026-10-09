"""DPLM2 embeds tokens twice, as the official multimodal wrapper does, except for the direct 3B network."""

from __future__ import annotations

import pytest
import torch

from tests.unit.tiny_families import dplm_values

from fastplms.models.dplm2.modeling_dplm2 import DPLM2Config, DPLM2ForMaskedLM


# ESM token dropout scales unmasked embeddings by 1 - 0.15 * 0.8 when no token is masked.
TOKEN_DROPOUT_SCALE = 1 - 0.15 * 0.8
STRUCTURE_TOKEN = 50  # DPLM2 structure ids are >= 33; amino-acid and special ids are < 33.


def _model(dplm_type: str | None) -> DPLM2ForMaskedLM:
    values = dplm_values(64)
    values["token_dropout"] = True
    if dplm_type is not None:
        values["dplm_type"] = dplm_type
    torch.manual_seed(0)
    return DPLM2ForMaskedLM(DPLM2Config(**values)).eval()


def _packed_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Two rows of a structure track followed by an amino-acid track, the second row padded."""
    input_ids = torch.tensor(
        [
            [STRUCTURE_TOKEN] * 4 + [0, 5, 6, 2],
            [STRUCTURE_TOKEN] * 3 + [1] + [0, 5, 2, 1],
        ]
    )  # (b=2, l=8)
    attention_mask = input_ids.ne(1)  # (b=2, l=8)
    return input_ids, attention_mask  # each (b=2, l=8)


@pytest.mark.parametrize(
    ("dplm_type", "scale"),
    ((None, TOKEN_DROPOUT_SCALE**2), ("dplm_esm", TOKEN_DROPOUT_SCALE)),
    ids=("multimodal-twice", "dplm-esm-once"),
)
def test_embedding_scale_matches_the_official_path(dplm_type: str | None, scale: float) -> None:
    model = _model(dplm_type)
    input_ids, attention_mask = _packed_batch()  # each (b=2, l=8)

    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    word_embeddings = model.esm.embeddings.word_embeddings(input_ids)  # (b=2, l=8, d)
    expected = word_embeddings * scale * attention_mask.unsqueeze(-1)  # (b=2, l=8, d)
    torch.testing.assert_close(output.hidden_states[0], expected)


def test_forward_encoder_and_embedding_api_share_the_official_embedding() -> None:
    model = _model(None)
    input_ids, attention_mask = _packed_batch()  # each (b=2, l=8)
    type_ids = model._get_modality_type(input_ids, attention_mask)  # (b=2, l=8)

    with torch.inference_mode():
        masked_lm_hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        encoder_hidden = model.esm(
            input_ids=input_ids, attention_mask=attention_mask, type_ids=type_ids
        ).last_hidden_state
        embedded = model._embed(input_ids, attention_mask)
    # masked_lm_hidden, encoder_hidden, embedded: (b=2, l=8, d)

    assert torch.equal(masked_lm_hidden, encoder_hidden)
    assert torch.equal(masked_lm_hidden, embedded)
