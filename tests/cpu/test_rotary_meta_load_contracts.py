"""Rotary frequency tables must survive meta-device checkpoint loading.

Every rotary module here registers ``inv_freq`` with ``persistent=False``, so no
checkpoint carries it and nothing refills it after ``load_state_dict``. It is
expected to come from ``__init__``. Transformers builds modules under
``torch.device("meta")`` while loading a checkpoint, which makes whatever
``__init__`` computed a meta tensor; materializing that yields uninitialized
memory instead of the frequency table.

The failure is quiet, which is why it needs a test rather than an assertion.
Uninitialized memory is usually finite, so the model keeps running and rotates
every position by frequencies unrelated to ``base`` and ``dim``. Only
occasionally is it NaN, and then every head returns NaN from clean queries.

The families defend differently and both defenses are covered here. E1 registers
an empty buffer and fills it on first use, so these tests fill it the way a
forward pass would before reading it. ESM++ and ESM3 regenerate in ``_apply``
after each device move, so their table is expected to be correct immediately.
"""

import pytest
import torch

from fastplms.models.e1.modeling_e1 import RotaryPositionalEmbedding
from fastplms.models.esm3.modeling_esm3 import RotaryEmbedding as Esm3RotaryEmbedding
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    RotaryEmbedding as EsmPlusPlusRotaryEmbedding,
)


HEAD_DIM = 64
BASE = 10000.0
CACHE_LENGTH = 8

ROTARY_CLASSES = [
    pytest.param(EsmPlusPlusRotaryEmbedding, id="esm_plusplus"),
    pytest.param(Esm3RotaryEmbedding, id="esm3"),
    pytest.param(RotaryPositionalEmbedding, id="e1"),
]


def _expected_inv_freq() -> torch.Tensor:
    return 1.0 / (
        BASE ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )  # (head_dim / 2,)


def _materialize_from_meta(rotary_class) -> torch.nn.Module:
    """Build the module the way a checkpoint load does, then bring it to CPU."""
    with torch.device("meta"):
        module = rotary_class(dim=HEAD_DIM)

    # These modules carry buffers and no parameters, so a parameters-only meta
    # check reports nothing and ``.to`` then fails on the meta buffers.
    on_meta = any(tensor.is_meta for tensor in module.parameters()) or any(
        tensor.is_meta for tensor in module.buffers()
    )
    return module.to_empty(device="cpu") if on_meta else module.to("cpu")


def _frequency_table(module: torch.nn.Module) -> torch.Tensor:
    """Return the module's frequency table, filling a lazy one first.

    A zero-width buffer is the lazy strategy rather than a fault, and the first
    forward is what fills it. Doing that here keeps every family under the same
    assertions instead of exempting one.
    """
    if module.inv_freq.numel() == 0:
        module._set_sin_cos_cache(seq_len=CACHE_LENGTH, device=torch.device("cpu"))
    return module.inv_freq  # (head_dim / 2,)


@pytest.mark.parametrize("rotary_class", ROTARY_CLASSES)
def test_inverse_frequencies_are_valid_after_meta_load(rotary_class) -> None:
    module = _materialize_from_meta(rotary_class)

    inv_freq = _frequency_table(module)  # (head_dim / 2,)

    assert torch.isfinite(inv_freq).all()
    torch.testing.assert_close(inv_freq, _expected_inv_freq())


@pytest.mark.parametrize("rotary_class", ROTARY_CLASSES)
def test_meta_load_matches_direct_construction(rotary_class) -> None:
    """A loaded module must hold the same table a directly built one holds."""
    loaded = _frequency_table(_materialize_from_meta(rotary_class))  # (head_dim / 2,)
    direct = _frequency_table(rotary_class(dim=HEAD_DIM))  # (head_dim / 2,)

    torch.testing.assert_close(loaded, direct)


@pytest.mark.parametrize("rotary_class", ROTARY_CLASSES)
def test_frequency_table_is_not_persisted(rotary_class) -> None:
    """Pins the premise: nothing restores this buffer, so nothing may rely on it."""
    module = rotary_class(dim=HEAD_DIM)

    assert "inv_freq" not in module.state_dict()
