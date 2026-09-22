"""DPLM2 packed-layout detection and the split rotary path it selects."""

from __future__ import annotations

import pytest
import torch

from fastplms.models._esm_rotary import RotaryEmbedding, apply_rotary_pos_emb
from fastplms.models.dplm2.modeling_dplm2 import (
    ModifiedRotaryEmbedding,
    _has_packed_multimodal_layout,
)


AA, STRUCT, PAD = 0, 1, 2


def _layout(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)  # (b, l)


@pytest.mark.parametrize(
    ("type_ids", "expected"),
    (
        (None, False),
        (_layout([[AA, AA, STRUCT]]), False),  # odd length cannot split in half
        (_layout([[AA, AA, STRUCT, STRUCT]]), True),
        (_layout([[STRUCT, STRUCT, AA, AA]]), True),  # either track may come first
        (_layout([[AA, PAD, STRUCT, PAD]]), True),  # equal non-pad counts per half
        (_layout([[AA, AA, AA, AA]]), False),  # plain protein batch
        (_layout([[AA, AA, STRUCT, PAD]]), False),  # unequal non-pad counts
        (_layout([[PAD, PAD, PAD, PAD]]), False),  # an empty half is not packed
        (_layout([[AA, STRUCT, STRUCT, AA]]), False),  # modalities interleave
        # One unpacked row disqualifies the whole batch.
        (_layout([[AA, AA, STRUCT, STRUCT], [AA, AA, AA, AA]]), False),
        (_layout([[AA, AA, STRUCT, STRUCT], [STRUCT, PAD, AA, PAD]]), True),
    ),
)
def test_packed_multimodal_layout_truth_table(
    type_ids: torch.Tensor | None, expected: bool
) -> None:
    assert _has_packed_multimodal_layout(type_ids, AA, STRUCT, PAD) is expected


def _rotary(head_dim: int) -> ModifiedRotaryEmbedding:
    return ModifiedRotaryEmbedding(head_dim, aa_type=AA, struct_type=STRUCT, pad_type=PAD)


def test_packed_layout_rotates_each_half_from_position_zero() -> None:
    head_dim, half = 8, 3
    query = torch.randn(1, 2, 2 * half, head_dim)  # (b=1, h=2, l=6, d=8)
    key = torch.randn(1, 2, 2 * half, head_dim)  # (b=1, h=2, l=6, d=8)
    type_ids = _layout([[AA] * half + [STRUCT] * half])  # (b=1, l=6)

    actual_query, actual_key = _rotary(head_dim)(query, key, type_ids)  # each (b=1, h=2, l=6, d=8)

    probe = torch.zeros(1, 1, half, head_dim)  # (b=1, h=1, l=3, d=8)
    cos, sin = RotaryEmbedding(head_dim)._update_cos_sin_tables(  # each (1, 1, l=3, d=8)
        probe, seq_dimension=-2
    )
    for actual, source in ((actual_query, query), (actual_key, key)):
        expected = torch.cat(
            [apply_rotary_pos_emb(part, cos, sin) for part in source.chunk(2, dim=-2)],
            dim=-2,
        )  # (b=1, h=2, l=6, d=8)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("type_ids", (None, _layout([[AA] * 6])), ids=("no-type-ids", "plain"))
def test_unpacked_layout_rotates_the_full_sequence(type_ids: torch.Tensor | None) -> None:
    head_dim, seq_len = 8, 6
    query = torch.randn(1, 2, seq_len, head_dim)  # (b=1, h=2, l=6, d=8)
    key = torch.randn(1, 2, seq_len, head_dim)  # (b=1, h=2, l=6, d=8)

    actual_query, actual_key = _rotary(head_dim)(query, key, type_ids)  # each (b=1, h=2, l=6, d=8)

    probe = torch.zeros(1, 1, seq_len, head_dim)  # (b=1, h=1, l=6, d=8)
    cos, sin = RotaryEmbedding(head_dim)._update_cos_sin_tables(  # each (1, 1, l=6, d=8)
        probe, seq_dimension=-2
    )
    assert torch.equal(actual_query, apply_rotary_pos_emb(query, cos, sin))
    assert torch.equal(actual_key, apply_rotary_pos_emb(key, cos, sin))


def test_layout_change_between_calls_rebuilds_the_right_table() -> None:
    head_dim, seq_len = 8, 6
    rotary = _rotary(head_dim)
    query = torch.randn(1, 2, seq_len, head_dim)  # (b=1, h=2, l=6, d=8)
    key = torch.randn(1, 2, seq_len, head_dim)  # (b=1, h=2, l=6, d=8)
    packed = _layout([[AA] * 3 + [STRUCT] * 3])  # (b=1, l=6)

    # Each call returns (query, key), each (b=1, h=2, l=6, d=8); only the
    # rotated query is compared below.
    plain_first, _ = rotary(query, key, None)
    packed_query, _ = rotary(query, key, packed)
    plain_again, _ = rotary(query, key, None)

    assert torch.equal(plain_first, plain_again)
    assert not torch.equal(plain_first, packed_query)


@pytest.mark.parametrize(
    "type_ids",
    (_layout([[AA] * 3 + [STRUCT] * 3]), _layout([[AA] * 6])),
    ids=("packed", "plain"),
)
def test_rotary_call_detects_the_layout_once(
    type_ids: torch.Tensor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Layout detection synchronizes with the host, so one call must probe once."""
    from fastplms.models.dplm2 import modeling_dplm2

    probes: list[torch.Tensor | None] = []
    detect = modeling_dplm2._has_packed_multimodal_layout

    def counting_detect(type_ids: torch.Tensor | None, *args: int, **kwargs: int) -> bool:
        probes.append(type_ids)
        return detect(type_ids, *args, **kwargs)

    monkeypatch.setattr(modeling_dplm2, "_has_packed_multimodal_layout", counting_detect)
    query = torch.randn(1, 2, 6, 8)  # (b=1, h=2, l=6, d=8)
    key = torch.randn(1, 2, 6, 8)  # (b=1, h=2, l=6, d=8)

    _rotary(8)(query, key, type_ids)

    assert len(probes) == 1
