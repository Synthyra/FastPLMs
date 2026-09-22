"""E1 reads its rotary table lengths once per forward instead of once per layer."""

from __future__ import annotations

import pytest
import torch

from fastplms.models.e1 import modeling_e1
from fastplms.models.e1.modeling_e1 import FAST_E1_ENCODER, E1Config, RotaryLengths


NUM_LAYERS = 4


def _encoder() -> FAST_E1_ENCODER:
    torch.manual_seed(7)
    return FAST_E1_ENCODER(
        E1Config(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=NUM_LAYERS,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_num_sequences=4,
            max_num_positions_within_seq=16,
            max_num_positions_global=32,
            # Layers alternate between within-sequence and global positions.
            global_attention_every_n_layers=2,
            attn_backend="sdpa",
            dtype="float32",
        )
    ).eval()


def _inputs() -> dict[str, torch.Tensor]:
    # Two packed proteins in row 0 and one in row 1; -1 marks padding. All: (b=2, l=5).
    return {
        "inputs_embeds": torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(1)),
        "within_seq_position_ids": torch.tensor(((0, 1, 0, 1, -1), (0, 1, 2, -1, -1))),
        "global_position_ids": torch.tensor(((0, 1, 2, 3, -1), (0, 1, 2, -1, -1))),
        "sequence_ids": torch.tensor(((0, 0, 1, 1, -1), (0, 0, 0, -1, -1))),
    }


def test_validation_reports_the_rotary_rows_each_position_kind_needs() -> None:
    inputs = _inputs()

    lengths = modeling_e1._validate_biological_indices(
        inputs["within_seq_position_ids"],
        inputs["global_position_ids"],
        inputs["sequence_ids"],
        _encoder().config,
    )

    assert lengths == RotaryLengths(within_seq=3, global_positions=4)


def test_layers_receive_the_lengths_and_never_read_the_positions_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[RotaryLengths | None, bool]] = []
    ensure_cache = modeling_e1.RotaryPositionalEmbedding._ensure_sin_cos_cache

    def recording_ensure_cache(
        self, position_ids, seq_len, device, rotary_lengths=None, within_seq_layer=True
    ) -> None:
        received.append((rotary_lengths, within_seq_layer))
        # Without position IDs the helper can size the table only from the record.
        ensure_cache(self, None, seq_len, device, rotary_lengths, within_seq_layer)

    monkeypatch.setattr(
        modeling_e1.RotaryPositionalEmbedding, "_ensure_sin_cos_cache", recording_ensure_cache
    )

    encoder = _encoder()
    encoder(**_inputs())

    lengths = RotaryLengths(within_seq=3, global_positions=4)
    assert len(received) == NUM_LAYERS
    assert {record for record, _ in received} == {lengths}
    assert sorted(within_seq_layer for _, within_seq_layer in received) == [
        False,
        False,
        True,
        True,
    ]
    table_rows = sorted(
        layer.norm_attn_norm.self_attn.rotary_emb.max_seq_len_cached for layer in encoder.layers
    )
    assert table_rows == [3, 3, 4, 4]


def test_shared_lengths_match_per_layer_lengths_bitwise(monkeypatch: pytest.MonkeyPatch) -> None:
    shared = _encoder()(**_inputs()).last_hidden_state  # (b, l, d)

    attention_forward = modeling_e1.Attention.forward

    def forward_without_lengths(self, *args, **kwargs):
        kwargs["rotary_lengths"] = None
        return attention_forward(self, *args, **kwargs)

    monkeypatch.setattr(modeling_e1.Attention, "forward", forward_without_lengths)
    per_layer = _encoder()(**_inputs()).last_hidden_state  # (b, l, d)

    assert torch.equal(shared, per_layer)
    assert torch.isfinite(shared).all()
