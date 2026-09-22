"""The chunked triangle contraction prepares its right stream once and changes no value.

The reference below is the per-chunk ``torch.einsum`` the official model runs.
"""

from __future__ import annotations

import pytest
import torch

from fastplms.models.esmfold2 import modeling_esmfold2_common as common


LATENT_CHANNELS = 6
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _per_chunk_einsum(
    block: common.TriangleMultiplicativeBlock,
    left_stream: torch.Tensor,
    right_stream: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    outgoing = block.flow == "outgoing"
    length = left_stream.shape[1] if outgoing else left_stream.shape[2]
    chunks = []
    for start in range(0, length, chunk_size):
        stop = start + chunk_size
        rows = left_stream[:, start:stop] if outgoing else left_stream[:, :, start:stop]
        chunks.append(torch.einsum(block._einsum_equation, rows, right_stream))
    return torch.cat(chunks, dim=1)  # (b, i, j, d)


def _streams(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(8)
    routed = torch.randn(2, 11, 11, 2 * LATENT_CHANNELS, generator=generator).to(device)
    left_stream, right_stream = routed.chunk(2, dim=-1)  # each (b, l, l, d), views as in forward
    return left_stream, right_stream


def _assert_contractions_match(device: torch.device, flow: str, chunk_size: int) -> None:
    block = common.TriangleMultiplicativeBlock(8, LATENT_CHANNELS, flow=flow)
    left_stream, right_stream = _streams(device)
    for autocast in (False, True):
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            expected = _per_chunk_einsum(block, left_stream, right_stream, chunk_size)
            contracted = block._triangular_contract_chunked(left_stream, right_stream, chunk_size)
        assert contracted.dtype == expected.dtype
        assert contracted.dtype == (torch.bfloat16 if autocast else torch.float32)
        assert torch.equal(contracted, expected)


@pytest.mark.parametrize("flow", ("outgoing", "incoming"))
@pytest.mark.parametrize("chunk_size", (1, 4, 11, 64))
def test_chunked_contraction_matches_per_chunk_einsum_bitwise(flow: str, chunk_size: int) -> None:
    _assert_contractions_match(torch.device("cpu"), flow, chunk_size)


@requires_cuda
@pytest.mark.gpu
@pytest.mark.parametrize("flow", ("outgoing", "incoming"))
@pytest.mark.parametrize("chunk_size", (4, 64))
def test_chunked_contraction_matches_per_chunk_einsum_bitwise_on_cuda(
    flow: str, chunk_size: int
) -> None:
    _assert_contractions_match(torch.device("cuda"), flow, chunk_size)


def test_right_stream_is_laid_out_once_per_contraction(monkeypatch: pytest.MonkeyPatch) -> None:
    block = common.TriangleMultiplicativeBlock(8, LATENT_CHANNELS, flow="outgoing")
    left_stream, right_stream = _streams(torch.device("cpu"))
    products: list[tuple[torch.Tensor, torch.Tensor]] = []
    batched_product = torch.bmm

    def recording_product(rows: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
        products.append((rows, columns))
        return batched_product(rows, columns)

    monkeypatch.setattr(torch, "bmm", recording_product)
    block._triangular_contract_chunked(left_stream, right_stream, 4)

    assert [rows.shape[1] for rows, _ in products] == [4, 4, 3]
    # Every chunk multiplies by the same prepared tensor, not by a fresh copy.
    assert len({columns.data_ptr() for _, columns in products}) == 1
    assert products[0][1].is_contiguous()


def test_swiglu_reuses_its_activation_buffer_without_changing_a_value() -> None:
    torch.manual_seed(4)
    ffn = common.SwiGLU(in_features=8, hidden_features=16).eval()
    pair = torch.randn(2, 5, 5, 8)  # (b, l, l, d)
    original = pair.clone()
    for autocast in (False, True):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            with torch.enable_grad():
                with_autograd = ffn(pair)
            with torch.no_grad():
                without_autograd = ffn(pair)
        assert torch.equal(with_autograd.detach(), without_autograd)
        assert torch.equal(pair, original)


@pytest.mark.parametrize("flow", ("outgoing", "incoming"))
@pytest.mark.parametrize("chunk_size", (None, 4))
def test_triangle_block_matches_the_unreleased_reference_bitwise(
    flow: str, chunk_size: int | None
) -> None:
    """The block frees dead tensors early; the arithmetic must be the official one."""
    torch.manual_seed(6)
    block = common.TriangleMultiplicativeBlock(8, LATENT_CHANNELS, flow=flow).eval()
    block.set_chunk_size(chunk_size)
    pair = torch.randn(2, 11, 11, 8)  # (b, l, l, d)
    visibility = (torch.rand(2, 11, 11) > 0.2).float()  # (b, l, l)

    def reference() -> torch.Tensor:
        normalized = block.norm_start(pair)
        signal, gate_logits = block.proj_bundle(normalized).split(2 * LATENT_CHANNELS, dim=-1)
        routed = signal * torch.sigmoid(gate_logits) * visibility.unsqueeze(-1)
        left_stream, right_stream = routed.float().chunk(2, dim=-1)
        if chunk_size is None:
            contracted = torch.einsum(block._einsum_equation, left_stream, right_stream)
        else:
            contracted = _per_chunk_einsum(block, left_stream, right_stream, chunk_size)
        mixed = block.proj_emit(block.norm_mix(contracted))
        return mixed * torch.sigmoid(block.proj_gate(normalized))

    for autocast in (False, True):
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            assert torch.equal(block(pair, visibility), reference())
