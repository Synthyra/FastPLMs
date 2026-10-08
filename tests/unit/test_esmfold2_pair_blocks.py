"""ESMFold2 pair-stack and MSA blocks at toy widths: shapes, masks, chunking, and kernel-backend switches.

Every module here is built with a few units per axis and random weights, so each test states a property of the block
rather than a recorded output: a chunked pass equals an unchunked pass, a masked entry contributes nothing, and a
backend that is not installed is refused before any module changes.

Shapes: `b` batch, `l` tokens, `m` MSA rows, `d` channel width.
"""

import pytest
import torch

from torch import nn

from fastplms.models.esmfold2 import modeling_esmfold2_common as common
from fastplms.models.esmfold2.modeling_esmfold2_common import (
    FoldingTrunk,
    LanguageModelShim,
    MSAPairWeightedAveraging,
    OuterProductMean,
    PairUpdateBlock,
    SingleToPair,
    Transition,
    TransitionLayer,
    TriangleMultiplicativeBlock,
    TriangleMultiplicativeUpdate,
)


D = 8  # channel width of the toy pair tensors
L = 5  # tokens
BATCH = 2


def seeded(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def random_pair(width: int = D, length: int = L, batch: int = BATCH, seed: int = 0) -> torch.Tensor:
    return torch.randn(batch, length, length, width, generator=seeded(seed))  # (b, l, l, d)


@pytest.fixture(autouse=True)
def fixed_weights():
    torch.manual_seed(1234)


def test_a_single_representation_becomes_a_pair_from_the_products_and_differences_of_its_rows():
    block = SingleToPair(input_dim=6, downproject_dim=4, output_dim=D).eval()
    single = torch.randn(BATCH, L, 6, generator=seeded(1))  # (b, l, 6)

    pair = block(single)  # (b, l, l, d)

    projected = block.downproject(single)  # (b, l, 4)
    on_the_diagonal = block.output_mlp(torch.cat([projected**2, torch.zeros_like(projected)], dim=-1))  # (b, l, d)
    assert pair.shape == (BATCH, L, L, D) and torch.isfinite(pair).all()
    torch.testing.assert_close(pair[:, range(L), range(L)], on_the_diagonal)


def test_language_model_states_project_to_a_pair_with_optional_dropout():
    shim = LanguageModelShim(d_z=D, d_model=12, num_layers=3).eval()
    states = torch.randn(BATCH, L, 4, 12, generator=seeded(2))  # (b, l, n_layers + 1, d_model)

    plain = shim(states)  # (b, l, l, d_z)
    again = shim(states)
    dropped = shim(states, lm_dropout=0.5)

    assert plain.shape == (BATCH, L, L, D) and torch.equal(plain, again)
    assert dropped.shape == plain.shape and not torch.equal(dropped, plain)
    assert torch.equal(shim.base_z_mlp(shim.project_sequence(states)), plain)


def test_the_triangle_block_exposes_its_weight_halves_and_flow_direction_for_the_kernel():
    outgoing = TriangleMultiplicativeBlock(input_channels=D, latent_channels=4, flow="outgoing")
    incoming = TriangleMultiplicativeBlock(input_channels=D, latent_channels=4, flow="incoming")

    projection, gate = outgoing.split_kernel_weights()

    assert projection.shape == (8, D) and gate.shape == (8, D)
    assert torch.equal(torch.cat([projection, gate]), outgoing.proj_bundle.weight)
    assert (outgoing._kernel_flow_direction(), incoming._kernel_flow_direction()) == ("outgoing", "incoming")
    with pytest.raises(ValueError, match="Invalid flow"):
        TriangleMultiplicativeBlock(input_channels=D, latent_channels=4, flow="sideways")


def test_the_triangle_update_gives_the_same_result_chunked_or_not_and_refuses_missing_kernel_backends():
    update = TriangleMultiplicativeUpdate(dim=D, _outgoing=True).eval()
    pair = random_pair()
    mask = torch.ones(BATCH, L, L)  # (b, l, l)
    mask[:, :, -1] = 0.0

    update.set_chunk_size(None)
    whole = update(pair, mask=mask)
    update.set_chunk_size(2)
    chunked = update(pair, mask=mask)

    assert update._engine._chunk_size == 2
    torch.testing.assert_close(chunked, whole, atol=1e-5, rtol=1e-5)
    update.set_kernel_backend(None)
    assert update._engine._use_kernels is False
    for backend, reason in (("fused", "does not bundle"), ("cuequivariance", "requires cuequivariance_torch"), ("nonsense", "backend must be one of")):
        if (backend == "cuequivariance" and common.CUE_AVAILABLE) or (backend == "fused" and common.TRITON_KERNELS_AVAILABLE):
            continue
        with pytest.raises((RuntimeError, ValueError), match=reason):
            update.set_kernel_backend(backend)
    assert update._engine._use_kernels is False


def test_the_pair_transition_is_chunked_along_rows_without_changing_its_values_and_swiglu_pieces_agree():
    transition = Transition(D, expansion_ratio=2).eval()
    pair = random_pair(length=7)

    transition.set_chunk_size(None)
    whole = transition(pair)
    transition.set_chunk_size(3)
    chunked = transition(pair)
    normed = transition.norm(pair)
    hidden = transition._swiglu_pre_w3(normed)  # (b, l, l, d_inner)

    torch.testing.assert_close(chunked, whole, atol=1e-6, rtol=1e-5)
    assert hidden.shape[-1] == transition.ffn.hidden_features == 2 * D
    torch.testing.assert_close(transition.ffn.w3(hidden), transition.ffn(normed))
    torch.testing.assert_close(transition._addmm_residual(pair, hidden), pair + transition.ffn.w3(hidden), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(whole, pair + transition.ffn(normed), atol=1e-6, rtol=1e-5)


def test_the_pair_transition_uses_the_fused_path_only_for_a_fused_backend_on_cuda_in_bfloat16():
    transition = Transition(D).eval()

    transition.set_kernel_backend(None)

    assert transition._kernel_backend is None and transition._fused_swiglu is None
    assert transition._can_use_fused_path(random_pair().bfloat16()) is False
    with pytest.raises(ValueError, match="backend must be one of"):
        transition.set_kernel_backend("nonsense")


def test_a_pair_update_block_forwards_chunk_sizes_and_backends_to_its_parts():
    block = PairUpdateBlock(d_pair=D, expansion_ratio=2).eval()
    pair = random_pair()

    block.set_chunk_size(2)
    chunked = block(pair)
    block.set_chunk_size(None)
    whole = block(pair)
    block.set_kernel_backend(None)

    assert block.tri_mul_out._engine._chunk_size is None and block.pair_transition._chunk_size is None
    torch.testing.assert_close(chunked, whole, atol=1e-5, rtol=1e-5)
    assert block._kernel_backend is None and block._can_use_fused_trimul_with_residual(pair) is False
    with pytest.raises(ValueError, match="backend must be one of"):
        block.set_kernel_backend("nonsense")


def test_the_fused_triangle_call_receives_bfloat16_weights_of_the_engine_for_its_direction(monkeypatch):
    block = PairUpdateBlock(d_pair=D, expansion_ratio=2).eval()
    pair = random_pair().bfloat16()
    mask = torch.ones(BATCH, L, L)
    calls = []

    def record(received, direction, **keywords):
        calls.append((received, direction, keywords))
        return received

    monkeypatch.setattr(common, "_fused_trimul_with_residual", record)

    returned = block._fused_trimul_with_residual(pair, "incoming", mask)
    block._fused_trimul_with_residual(pair, "outgoing", None)

    received, direction, keywords = calls[0]
    engine = block.tri_mul_in._engine
    assert returned is pair and received is pair and direction == "incoming" and keywords["mask"] is mask
    assert keywords["residual"] is pair and keywords["drop_mask"] is None and keywords["eps"] == common._EPS
    assert all(keywords[name].dtype == torch.bfloat16 for name in keywords if name.endswith(("weight", "bias")))
    torch.testing.assert_close(keywords["p_in_weight"].float(), engine.split_kernel_weights()[0].bfloat16().float())
    torch.testing.assert_close(keywords["norm_out_bias"].float(), engine.norm_mix.bias.bfloat16().float())
    assert calls[1][1] == "outgoing" and calls[1][2]["mask"] is None


def test_the_folding_trunk_forwards_chunk_sizes_to_every_block_and_keeps_the_pair_shape():
    trunk = FoldingTrunk(n_layers=2, d_pair=D, expansion_ratio=2).eval()
    pair = random_pair()

    trunk.set_chunk_size(2)
    trunk.set_kernel_backend(None)
    with torch.no_grad():
        refined = trunk(pair, pair_attention_mask=torch.ones(BATCH, L, L))

    assert [block.tri_mul_in._engine._chunk_size for block in trunk.blocks] == [2, 2]
    assert refined.shape == pair.shape and torch.isfinite(refined).all() and not torch.equal(refined, pair)


def test_the_outer_product_mean_is_chunked_along_rows_and_ignores_masked_rows():
    block = OuterProductMean(d_msa=6, d_hidden=3, d_pair=D).eval()
    msa = torch.randn(BATCH, L, 4, 6, generator=seeded(3))  # (b, l, m, d_msa)
    mask = torch.ones(BATCH, L, 4)  # (b, l, m)
    mask[:, :, 3] = 0.0

    whole = block(msa, mask)
    block.set_chunk_size(2)
    chunked = block(msa, mask)
    changed = msa.clone()
    changed[:, :, 3] += 5.0
    ignored = block(changed, mask)

    assert whole.shape == (BATCH, L, L, D) and block._chunk_size == 2
    torch.testing.assert_close(chunked, whole, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ignored, whole)


def test_dividing_the_outer_product_before_the_projection_changes_only_how_the_bias_is_scaled():
    after = OuterProductMean(d_msa=6, d_hidden=3, d_pair=D).eval()
    before = OuterProductMean(d_msa=6, d_hidden=3, d_pair=D, divide_outer_before_proj=True).eval()
    before.load_state_dict(after.state_dict())
    msa = torch.randn(1, L, 4, 6, generator=seeded(4))
    mask = torch.ones(1, L, 4)

    without_bias_after = after.Wout.bias.detach().clone()
    with torch.no_grad():
        after.Wout.bias.zero_()
        before.Wout.bias.zero_()
    torch.testing.assert_close(before(msa, mask), after(msa, mask), atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        after.Wout.bias.copy_(without_bias_after)
        before.Wout.bias.copy_(without_bias_after)
    unchunked = before(msa, mask)
    before.set_chunk_size(2)
    assert not torch.allclose(unchunked, after(msa, mask))
    torch.testing.assert_close(before(msa, mask), unchunked, atol=1e-6, rtol=1e-5)


def test_msa_rows_are_averaged_over_positions_with_pair_biased_weights_that_skip_masked_positions():
    block = MSAPairWeightedAveraging(d_msa=6, d_pair=D, n_heads=2, head_width=3).eval()
    msa = torch.randn(BATCH, L, 4, 6, generator=seeded(5))  # (b, l, m, d_msa)
    pair = random_pair(seed=6)
    mask = torch.ones(BATCH, L, L, dtype=torch.bool)
    mask[:, :, 0] = False

    averaged = block(msa, pair, mask)  # (b, l, m, d_msa)
    changed = msa.clone()
    changed[:, 0] += torch.randn(BATCH, 4, 6, generator=seeded(9))
    unchanged_rows = block(changed, pair, mask)

    assert averaged.shape == msa.shape and torch.isfinite(averaged).all()
    torch.testing.assert_close(unchanged_rows[:, 1:], averaged[:, 1:])
    assert not torch.allclose(unchanged_rows[:, 0], averaged[:, 0])


def test_the_swiglu_transition_layer_matches_its_definition():
    layer = TransitionLayer(D, n=2).eval()
    x = torch.randn(BATCH, L, D, generator=seeded(7))  # (b, l, d)

    transformed = layer(x)

    normed = layer.norm(x)
    expected = layer.out_proj(nn.functional.silu(layer.a_proj(normed)) * layer.b_proj(normed))
    assert transformed.shape == x.shape
    torch.testing.assert_close(transformed, expected)
