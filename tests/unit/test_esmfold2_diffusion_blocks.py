"""ESMFold2 atom and diffusion blocks at toy widths: encoders, decoders, conditioning, denoising, and sampler helpers.

Modules are built a few units wide with random weights, so each test states a property rather than a recorded output:
a block that starts as the identity still is one, a cached pass reuses what it cached, and the sampler's noise schedule,
random rotations, and rigid alignment satisfy their definitions.

Shapes: `b` batch, `a` atoms, `l` tokens, `s` diffusion samples, `bs` = `b * s`.
"""

import math
import pytest
import torch

from tests.unit.tiny_families import tiny_esmfold2_config

from fastplms.models.esmfold2 import modeling_esmfold2_common as common
from fastplms.models.esmfold2.modeling_esmfold2_common import (
    AttentionPairBias,
    DiffusionConditioning,
    DiffusionModule,
    DiffusionStructureHead,
    DiffusionTransformer,
    ESMFold2AtomDecoder,
    ESMFold2AtomEncoder,
    FourierEmbedding,
    SwiGLUFFN,
    SWAAtomBlock,
)


pytestmark = pytest.mark.filterwarnings("ignore:CUDA is not available or torch_xla is imported:UserWarning")

D_ATOM = 16  # a head must be 8 wide for the toy rotary table of four frequency pairs
D_TOKEN = 8
D_PAIR = 8
D_INPUTS = 10
ATOMS = 6
TOKENS = 3
ROPE = {"spatial_rope_base_frequency": 20.0, "n_spatial_rope_pairs_per_axis": 1, "n_uid_rope_pairs": 1}


def seeded(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


@pytest.fixture(autouse=True)
def fixed_weights():
    torch.manual_seed(4321)


def atom_inputs(batch: int = 1, seed: int = 0) -> dict[str, torch.Tensor]:
    """Atom features of `ATOMS` atoms in `TOKENS` tokens of two atoms each."""
    generator = seeded(seed)
    owner = torch.arange(ATOMS).div(2, rounding_mode="floor").expand(batch, -1).clone()  # (b, a)
    return {  # feature name -> (b, a, ...) tensor
        "ref_pos": torch.randn(batch, ATOMS, 3, generator=generator),
        "atom_attention_mask": torch.ones(batch, ATOMS, dtype=torch.bool),
        "ref_space_uid": owner,
        "ref_charge": torch.zeros(batch, ATOMS),
        "ref_element": torch.nn.functional.one_hot(torch.randint(0, 128, (batch, ATOMS), generator=generator), 128).float(),
        "ref_atom_name_chars": torch.nn.functional.one_hot(torch.randint(0, 64, (batch, ATOMS, 4), generator=generator), 64).float(),
        "atom_to_token": owner.clone(),
    }


def encoder(structure_prediction: bool = True, blocks: int = 1) -> ESMFold2AtomEncoder:
    return ESMFold2AtomEncoder(
        d_atom=D_ATOM, d_token=D_TOKEN, n_blocks=blocks, n_heads=2, swa_window_size=8,
        structure_prediction=structure_prediction, **ROPE,
    ).eval()


def test_fourier_features_are_cosines_of_the_noise_times_for_any_number_of_times():
    embedding = FourierEmbedding(6)

    scalar = embedding(torch.tensor(0.3))  # (1, c)
    batch = embedding(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))  # (4, c)

    assert scalar.shape == (1, 6) and batch.shape == (4, 6) and scalar.abs().max() <= 1.0
    expected = torch.cos(2 * torch.pi * (0.3 * embedding.w + embedding.b))
    torch.testing.assert_close(scalar[0], expected)
    assert torch.equal(embedding(torch.tensor(0.3)), scalar)


def test_the_atom_feed_forward_block_rounds_its_hidden_width_and_casts_inputs_to_its_weights():
    block = SwiGLUFFN(D_ATOM, expansion_ratio=2).eval()
    x = torch.randn(2, 4, D_ATOM, generator=seeded(1), dtype=torch.float64)  # (b, a, d)

    activations = block(x)

    assert block.w_up.out_features % 512 == 0 and activations.dtype == torch.float32 and activations.shape == (2, 4, D_ATOM)
    up_first, up_second = block.w_up(x.float()).chunk(2, dim=-1)
    torch.testing.assert_close(activations, block.w_down(torch.nn.functional.silu(up_first) * up_second))


def test_the_fused_adaptive_norm_and_gated_residual_helpers_match_their_formulas():
    x = torch.randn(2, 4, 8, generator=seeded(2))
    scale = torch.randn(2, 1, 8, generator=seeded(3))
    shift = torch.randn(2, 1, 8, generator=seeded(4))
    gate = torch.randn(2, 1, 8, generator=seeded(5))

    adapted = common._rms_adaln_raw(x, scale, shift)
    updated = common._gated_residual_raw(x, gate, adapted)

    torch.testing.assert_close(adapted, torch.nn.functional.rms_norm(x, (8,)) * (1 + scale) + shift)
    torch.testing.assert_close(updated, x + gate * adapted)


def test_an_atom_block_starts_as_the_identity_and_changes_its_input_once_its_modulation_is_trained():
    block = SWAAtomBlock(d_atom=D_ATOM, n_heads=2, half_window=4).eval()
    inputs = atom_inputs()
    cos, sin = common.build_3d_rope(inputs["ref_pos"], inputs["ref_space_uid"], head_dim=D_ATOM // 2, n_spatial_per_axis=1, n_uid_pairs=1)
    _, indices, cu_seqlens, max_seqlen, _ = common._prepare_atom_encoder_metadata(inputs["atom_attention_mask"], inputs["atom_to_token"], 1)
    parameters = (cos, sin, indices, cu_seqlens, max_seqlen)
    x = torch.randn(1, ATOMS, D_ATOM, generator=seeded(6))  # (b, a, d_atom)
    condition = torch.randn(1, ATOMS, D_ATOM, generator=seeded(7))

    untouched = block(x, condition, parameters)
    with torch.no_grad():
        block.adaln_modulation[1].weight.normal_(std=0.5)
    trained = block(x, condition, parameters)
    per_sample = block(x, condition[:, 0], parameters)

    torch.testing.assert_close(untouched, x)
    assert trained.shape == x.shape and not torch.allclose(trained, x) and per_sample.shape == x.shape


def test_the_atom_encoder_pools_atoms_into_tokens_reuses_cached_features_and_adds_noisy_coordinates():
    module = encoder()
    inputs = atom_inputs()
    noisy = torch.randn(1, ATOMS, 3, generator=seeded(8))  # (b, a, 3)
    cache: dict = {}

    tokens, atoms, condition, parameters, intermediates = module(**inputs, r_l=noisy, inference_cache=cache, return_intermediates=True)
    moved_inputs = {**inputs, "ref_pos": inputs["ref_pos"] + 5.0}
    reused, *_ = module(**moved_inputs, r_l=noisy, inference_cache=cache)
    fresh, *_ = module(**moved_inputs, r_l=noisy)
    quiet, *_ = module(**inputs)

    assert tokens.shape == (1, TOKENS, D_TOKEN) and atoms.shape == condition.shape == (1, ATOMS, D_ATOM) and len(intermediates) == 1
    assert len(parameters) == 5 and set(cache["atomencoder"]) >= {"c_base", "attention_params", "mask_exp", "n_tokens", "atom_to_token_exp"}
    torch.testing.assert_close(reused, tokens)
    assert not torch.allclose(fresh, tokens) and quiet.shape == tokens.shape and not torch.allclose(quiet, tokens)


def test_the_atom_encoder_of_the_input_embedder_gives_half_width_tokens_and_samples_expand_the_batch():
    module = encoder(structure_prediction=False)
    samples = encoder()
    inputs = atom_inputs()

    tokens, _, condition, _, _ = module(**inputs)
    sampled, atoms, _, _, _ = samples(**inputs, r_l=torch.zeros(2, ATOMS, 3), num_diffusion_samples=2)

    assert tokens.shape == (1, TOKENS, D_TOKEN // 2) and condition.shape == (1, ATOMS, D_ATOM)
    assert sampled.shape == (2, TOKENS, D_TOKEN) and atoms.shape == (2, ATOMS, D_ATOM)


def test_the_atom_decoder_turns_token_states_and_atom_states_into_coordinate_updates():
    atom_encoder = encoder()
    decoder = ESMFold2AtomDecoder(d_atom=D_ATOM, d_token=D_TOKEN, n_blocks=1, n_heads=2, swa_window_size=8, **ROPE).eval()
    inputs = atom_inputs()
    tokens, atoms, condition, parameters, _ = atom_encoder(**inputs, r_l=torch.zeros(1, ATOMS, 3))

    update, intermediates = decoder(tokens, atoms, condition, parameters, inputs["atom_to_token"], inputs["atom_attention_mask"], return_intermediates=True)
    plain, none_kept = decoder(tokens, atoms, condition, parameters, inputs["atom_to_token"], inputs["atom_attention_mask"])

    assert update.shape == (1, ATOMS, 3) and torch.isfinite(update).all() and len(intermediates) == 1
    torch.testing.assert_close(plain, update)
    assert none_kept == []


def test_attention_with_a_pair_bias_names_zero_betas_and_refuses_unknown_kernel_backends():
    block = AttentionPairBias(d_model=D_TOKEN, d_pair=D_PAIR, num_heads=2)

    assert block._is_zero_beta(0.0) and block._is_zero_beta(0) and not block._is_zero_beta(0.5)
    assert block._is_zero_beta(torch.zeros(3)) and not block._is_zero_beta(torch.tensor([0.0, 1.0]))
    block.set_kernel_backend(None)
    assert block._kernel_backend is None
    with pytest.raises(ValueError, match="backend must be one of"):
        block.set_kernel_backend("nonsense")
    assert block._kernel_backend is None


def test_the_token_transformer_forwards_a_backend_choice_to_each_attention_block_and_caches_pair_biases():
    transformer = DiffusionTransformer(d_model=D_TOKEN, d_pair=D_PAIR, num_heads=2, num_blocks=2, d_cond=D_TOKEN).eval()
    tokens = torch.randn(1, TOKENS, D_TOKEN, generator=seeded(9))  # (b, l, d)
    condition = torch.randn(1, TOKENS, D_TOKEN, generator=seeded(10))
    pair = torch.randn(1, TOKENS, TOKENS, D_PAIR, generator=seeded(11))
    cache: dict = {}

    transformer.set_kernel_backend(None)
    first, intermediates = transformer(tokens, condition, pair, inference_cache=cache, return_intermediates=True)
    later, _ = transformer(tokens, condition, pair * 2.0, inference_cache=cache)

    assert [block._kernel_backend for block in transformer.attn_blocks] == [None, None]
    assert first.shape == tokens.shape and len(intermediates) == 2 and set(cache["token_pair_bias"]) == {0, 1}
    torch.testing.assert_close(later, first)


def test_noise_conditioning_expands_samples_caches_the_pair_and_accepts_one_time_or_one_per_sample():
    conditioning = DiffusionConditioning(c_z=D_PAIR, c_s=D_TOKEN, c_s_inputs=D_INPUTS, fourier_dim=8, transition_multiplier=2).eval()
    inputs = torch.randn(1, TOKENS, D_INPUTS, generator=seeded(12))  # (b, l, c_s_inputs)
    trunk = torch.randn(1, TOKENS, TOKENS, D_PAIR, generator=seeded(13))
    relative = torch.randn(1, TOKENS, TOKENS, D_PAIR, generator=seeded(14))
    cache: dict = {}

    single, pair = conditioning(torch.tensor(8.0), inputs, None, trunk, relative, num_diffusion_samples=2, inference_cache=cache)
    per_sample, _ = conditioning(torch.tensor([8.0, 32.0]), inputs, None, trunk, relative, num_diffusion_samples=2)
    _, cached_pair = conditioning(torch.tensor(8.0), inputs, None, trunk * 3.0, relative, num_diffusion_samples=2, inference_cache=cache)

    assert single.shape == (2, TOKENS, D_TOKEN) and pair.shape == (1, TOKENS, TOKENS, D_PAIR) and cached_pair is pair
    torch.testing.assert_close(per_sample[0], single[0])
    assert not torch.allclose(per_sample[1], single[1])


def diffusion_module() -> DiffusionModule:
    return DiffusionModule(
        c_atom=D_ATOM, c_token=D_TOKEN, c_z=D_PAIR, c_s_inputs=D_INPUTS, sigma_data=16.0, fourier_dim=8,
        atom_num_blocks=1, atom_num_heads=2, token_num_blocks=1, token_num_heads=2, transition_multiplier=2,
        swa_window_size=8, **ROPE,
    ).eval()


def denoise(module: DiffusionModule, noisy: torch.Tensor, time: float, samples: int = 1, **extra) -> dict:
    # noisy: (bs, a, 3)
    inputs = atom_inputs()
    ids = torch.zeros(1, TOKENS, dtype=torch.long)
    pair = torch.randn(1, TOKENS, TOKENS, D_PAIR, generator=seeded(15))
    return module(
        x_noisy=noisy, t_hat=torch.tensor(time), ref_pos=inputs["ref_pos"], ref_charge=inputs["ref_charge"],
        ref_mask=inputs["atom_attention_mask"], ref_element=inputs["ref_element"], ref_atom_name_chars=inputs["ref_atom_name_chars"],
        ref_space_uid=inputs["ref_space_uid"], tok_idx=inputs["atom_to_token"],
        s_inputs=torch.randn(1, TOKENS, D_INPUTS, generator=seeded(16)), s_trunk=None, z_trunk=pair,
        relative_position_encoding=pair, asym_id=ids, residue_index=ids, entity_id=ids, token_index=ids, sym_id=ids,
        token_attention_mask=torch.ones(1, TOKENS, dtype=torch.bool), num_diffusion_samples=samples, **extra,
    )


def test_denoising_returns_coordinates_for_every_sample_and_optional_token_and_atom_states():
    module = diffusion_module()
    noisy = torch.randn(2, ATOMS, 3, generator=seeded(17)) * 16.0  # (bs, a, 3)

    with torch.no_grad():
        denoised = denoise(module, noisy, 8.0, samples=2, return_token_repr=True, return_atom_repr=True)
        bare = denoise(module, noisy, 8.0, samples=2)

    assert denoised["x_denoised"].shape == (2, ATOMS, 3) and torch.isfinite(denoised["x_denoised"]).all()
    assert denoised["token_repr"].shape == (2, TOKENS, D_TOKEN) and denoised["atom_intermediates"].shape == (2, ATOMS, 2, D_ATOM)
    assert bare["token_repr"] is None and bare["atom_intermediates"] is None
    torch.testing.assert_close(bare["x_denoised"], denoised["x_denoised"])


def test_a_noise_level_of_zero_leaves_the_coordinates_unchanged():
    module = diffusion_module()
    noisy = torch.randn(1, ATOMS, 3, generator=seeded(18))

    with torch.no_grad():
        denoised = denoise(module, noisy, 0.0)

    torch.testing.assert_close(denoised["x_denoised"], noisy)


def test_the_diffusion_module_forwards_a_backend_choice_to_its_token_transformer():
    module = diffusion_module()

    module.set_kernel_backend(None)

    assert [block._kernel_backend for block in module.token_transformer.attn_blocks] == [None]


def sampler_head() -> DiffusionStructureHead:
    return DiffusionStructureHead(tiny_esmfold2_config())


def test_the_karras_noise_schedule_falls_from_the_largest_to_the_smallest_sigma_and_ends_at_zero():
    head = sampler_head()
    head.set_kernel_backend(None)

    single = head.inference_noise_schedule(1)
    schedule = head.inference_noise_schedule(5)
    default = head.inference_noise_schedule()

    assert single.tolist() == pytest.approx([head.inference_s_max * head.sigma_data, 0.0])
    assert schedule.shape == (6,) and schedule[0].item() == pytest.approx(head.sigma_data * head.inference_s_max, rel=1e-5)
    assert schedule[-2].item() == pytest.approx(head.sigma_data * head.inference_s_min, rel=1e-4) and schedule[-1] == 0.0
    assert (schedule[:-1] > schedule[1:]).all() and default.shape == (head.inference_num_steps + 1,)


def test_random_rotations_are_proper_rotation_matrices():
    torch.manual_seed(5)

    rotations = DiffusionStructureHead._random_rotations(6, torch.float32, torch.device("cpu"))  # (n, 3, 3)

    assert rotations.shape == (6, 3, 3)
    torch.testing.assert_close(rotations @ rotations.transpose(-1, -2), torch.eye(3).expand(6, 3, 3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(rotations), torch.ones(6), atol=1e-5, rtol=1e-5)


def test_random_augmentation_centers_then_moves_every_coordinate_set_rigidly():
    head = sampler_head()
    torch.manual_seed(6)
    coordinates = torch.randn(2, 7, 3, generator=seeded(19)) * 5.0  # (b, a, 3)
    mask = torch.ones(2, 7)
    mask[:, -2:] = 0.0

    moved, partner = head._center_random_augmentation(coordinates, mask, second_coords=coordinates.clone())
    alone, none_returned = head._center_random_augmentation(coordinates, mask)

    torch.testing.assert_close(torch.cdist(moved, moved), torch.cdist(coordinates, coordinates), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(partner, moved)
    assert none_returned is None
    assert alone.shape == coordinates.shape and not torch.allclose(alone, moved)


def test_weighted_rigid_alignment_recovers_the_target_from_a_rotated_and_shifted_copy():
    angle = 0.8
    rotation = torch.tensor([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    target = torch.randn(2, 9, 3, generator=seeded(20)) * 4.0  # (b, n, 3)
    moved = target @ rotation.T + torch.tensor([3.0, -1.0, 2.0])
    weights = torch.rand(2, 9, generator=seeded(21)) + 0.5
    mask = torch.ones(2, 9)
    mask[:, -1] = 0.0

    aligned = DiffusionStructureHead._weighted_rigid_align(moved, target, weights, mask)  # (b, n, 3)

    torch.testing.assert_close(aligned, target, atol=1e-4, rtol=1e-4)
