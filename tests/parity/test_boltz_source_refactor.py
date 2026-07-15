"""Focused parity checks for independently maintained Boltz runtime helpers."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from difflib import SequenceMatcher
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from fastplms.models.boltz import vb_const
from fastplms.models.boltz import vb_layers_attention as pair_attention
from fastplms.models.boltz import vb_layers_attentionv2 as pair_attention_v2
from fastplms.models.boltz import vb_layers_confidence_utils as confidence
from fastplms.models.boltz import vb_layers_dropout as dropout
from fastplms.models.boltz import vb_layers_outer_product_mean as outer_product
from fastplms.models.boltz import vb_layers_pair_averaging as pair_averaging
from fastplms.models.boltz import vb_layers_pairformer as pairformer
from fastplms.models.boltz import vb_layers_transition as transition
from fastplms.models.boltz import vb_layers_triangular_mult as triangular_mult
from fastplms.models.boltz import vb_loss_diffusionv2 as diffusion_loss
from fastplms.models.boltz import vb_modules_diffusion_conditioning as conditioning
from fastplms.models.boltz import vb_modules_encodersv2 as encoders
from fastplms.models.boltz import vb_modules_transformersv2 as diffusion_transformers
from fastplms.models.boltz import vb_modules_utils as module_utils
from fastplms.models.boltz import vb_potentials_potentials as potentials
from fastplms.models.boltz import vb_potentials_schedules as schedules
from fastplms.models.boltz import vb_tri_attn_attention as triangle_attention
from fastplms.models.boltz import vb_tri_attn_primitives as primitives
from fastplms.models.boltz import vb_tri_attn_utils as attention_utils
from fastplms.models.boltz.minimal_featurizer import build_boltz2_features

pytestmark = pytest.mark.structure

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_ROOT = REPOSITORY_ROOT / "vendor" / "upstream" / "boltz" / "src"

REWRITTEN_SOURCE_PAIRS = (
    ("vb_const.py", "boltz/data/const.py"),
    ("vb_layers_attention.py", "boltz/model/layers/attention.py"),
    ("vb_layers_attentionv2.py", "boltz/model/layers/attentionv2.py"),
    ("vb_layers_initialize.py", "boltz/model/layers/initialize.py"),
    ("vb_layers_outer_product_mean.py", "boltz/model/layers/outer_product_mean.py"),
    ("vb_layers_pair_averaging.py", "boltz/model/layers/pair_averaging.py"),
    ("vb_layers_pairformer.py", "boltz/model/layers/pairformer.py"),
    ("vb_layers_transition.py", "boltz/model/layers/transition.py"),
    ("vb_layers_triangular_mult.py", "boltz/model/layers/triangular_mult.py"),
    ("vb_layers_dropout.py", "boltz/model/layers/dropout.py"),
    ("vb_potentials_schedules.py", "boltz/model/potentials/schedules.py"),
    (
        "vb_tri_attn_attention.py",
        "boltz/model/layers/triangular_attention/attention.py",
    ),
    (
        "vb_tri_attn_utils.py",
        "boltz/model/layers/triangular_attention/utils.py",
    ),
    (
        "vb_tri_attn_primitives.py",
        "boltz/model/layers/triangular_attention/primitives.py",
    ),
    ("vb_layers_confidence_utils.py", "boltz/model/layers/confidence_utils.py"),
    ("vb_loss_diffusionv2.py", "boltz/model/loss/diffusionv2.py"),
    (
        "vb_modules_diffusion_conditioning.py",
        "boltz/model/modules/diffusion_conditioning.py",
    ),
    ("vb_modules_transformersv2.py", "boltz/model/modules/transformersv2.py"),
    ("vb_modules_trunkv2.py", "boltz/model/modules/trunkv2.py"),
    ("vb_modules_utils.py", "boltz/model/modules/utils.py"),
    ("vb_modules_encodersv2.py", "boltz/model/modules/encodersv2.py"),
    ("vb_potentials_potentials.py", "boltz/model/potentials/potentials.py"),
)


def _install_import_only_dependency_stubs() -> None:
    """Provide initialization-only reference shims outside the core extras."""

    sys.modules.setdefault("einx", ModuleType("einx"))
    if "scipy.stats" in sys.modules:
        return

    class _InitializationOnlyTruncatedNormal:
        @staticmethod
        def std(*args: object, **kwargs: object) -> float:
            return 1.0

        @staticmethod
        def rvs(*args: object, **kwargs: object) -> np.ndarray:
            del args
            return np.zeros(kwargs["size"], dtype=np.float32)

    scipy = ModuleType("scipy")
    scipy.__path__ = []  # type: ignore[attr-defined]
    stats = ModuleType("scipy.stats")
    stats.truncnorm = _InitializationOnlyTruncatedNormal  # type: ignore[attr-defined]
    scipy.stats = stats  # type: ignore[attr-defined]
    sys.modules["scipy"] = scipy
    sys.modules["scipy.stats"] = stats


def _load_standalone(relative_path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, UPSTREAM_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("local_name", "upstream_name"), REWRITTEN_SOURCE_PAIRS)
def test_boltz_runtime_is_not_an_upstream_source_relocation(
    local_name: str,
    upstream_name: str,
) -> None:
    local_path = REPOSITORY_ROOT / "src" / "fastplms" / "models" / "boltz" / local_name
    upstream_path = UPSTREAM_ROOT / upstream_name
    local_lines = local_path.read_text(encoding="utf-8").splitlines()
    upstream_lines = upstream_path.read_text(encoding="utf-8").splitlines()
    similarity = SequenceMatcher(None, local_lines, upstream_lines).ratio()
    assert similarity < 0.75, f"{local_name} has line similarity {similarity:.3f}"


@pytest.fixture(scope="module")
def upstream_const() -> ModuleType:
    return _load_standalone("boltz/data/const.py", "fastplms_test_upstream_const")


@pytest.fixture(scope="module")
def upstream_dropout() -> ModuleType:
    return _load_standalone(
        "boltz/model/layers/dropout.py",
        "fastplms_test_upstream_dropout",
    )


@pytest.fixture(scope="module")
def upstream_schedules() -> ModuleType:
    return _load_standalone(
        "boltz/model/potentials/schedules.py",
        "fastplms_test_upstream_schedules",
    )


@pytest.fixture(scope="module")
def upstream_attention_utils() -> ModuleType:
    return _load_standalone(
        "boltz/model/layers/triangular_attention/utils.py",
        "fastplms_test_upstream_attention_utils",
    )


@pytest.fixture(scope="module")
def upstream_diffusion_loss() -> ModuleType:
    _install_import_only_dependency_stubs()
    return _load_standalone(
        "boltz/model/loss/diffusionv2.py",
        "fastplms_test_upstream_diffusion_loss",
    )


@pytest.fixture(scope="module")
def upstream_package() -> ModuleType:
    sys.path.insert(0, str(UPSTREAM_ROOT))
    try:
        yield importlib.import_module("boltz")
    finally:
        sys.path.remove(str(UPSTREAM_ROOT))


def test_runtime_constants_match_upstream(upstream_const: ModuleType) -> None:
    retained_names = (
        "chain_types",
        "chain_type_ids",
        "canonical_tokens",
        "tokens",
        "token_ids",
        "num_tokens",
        "prot_letter_to_token",
        "ref_atoms",
        "protein_backbone_atom_names",
        "nucleic_backbone_atom_names",
        "protein_backbone_atom_index",
        "nucleic_backbone_atom_index",
        "res_to_center_atom",
        "res_to_disto_atom",
        "num_elements",
        "bond_types",
        "contact_conditioning_info",
        "chunk_size_threshold",
        "method_types_ids",
        "num_method_types",
        "vdw_radii",
    )
    for name in retained_names:
        assert getattr(vb_const, name) == getattr(upstream_const, name), name


@pytest.mark.parametrize("columnwise", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_dropout_mask_matches_upstream(
    upstream_dropout: ModuleType,
    columnwise: bool,
    training: bool,
) -> None:
    pair = torch.empty(2, 5, 7, 3)
    torch.manual_seed(917)
    expected = upstream_dropout.get_dropout_mask(0.2, pair, training, columnwise)
    torch.manual_seed(917)
    actual = dropout.get_dropout_mask(0.2, pair, training, columnwise)
    assert torch.equal(actual, expected)


def test_parameter_schedules_match_upstream(upstream_schedules: ModuleType) -> None:
    reference_exp = upstream_schedules.ExponentialInterpolation(0.1, 4.0, 2.5)
    local_exp = schedules.ExponentialInterpolation(0.1, 4.0, 2.5)
    reference_step = upstream_schedules.PiecewiseStepFunction((0.2, 0.7), (1, 2, 5))
    local_step = schedules.PiecewiseStepFunction((0.2, 0.7), (1, 2, 5))
    for time in (0.0, 0.2, 0.21, 0.7, 0.9, 1.0):
        assert local_exp.compute(time) == reference_exp.compute(time)
        assert local_step.compute(time) == reference_step.compute(time)


@pytest.mark.parametrize("low_mem", [False, True])
def test_chunk_layer_matches_upstream(
    upstream_attention_utils: ModuleType,
    low_mem: bool,
) -> None:
    inputs = {
        "left": torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4),
        "right": torch.tensor([[[2.0, 1.0, 0.0, -1.0]]]),
    }

    def layer(left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"sum": left + right, "product": left * right}

    expected = upstream_attention_utils.chunk_layer(
        layer,
        inputs,
        chunk_size=4,
        no_batch_dims=2,
        low_mem=low_mem,
    )
    actual = attention_utils.chunk_layer(
        layer,
        inputs,
        chunk_size=4,
        no_batch_dims=2,
        low_mem=low_mem,
    )
    assert torch.equal(actual["sum"], expected["sum"])
    assert torch.equal(actual["product"], expected["product"])


@pytest.mark.parametrize(
    ("start", "stop"),
    [(0, 1), (1, 7), (3, 19), (0, 24), (17, 24)],
)
def test_low_memory_flat_slice_matches_upstream(
    upstream_attention_utils: ModuleType,
    start: int,
    stop: int,
) -> None:
    tensor = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    expected = upstream_attention_utils._chunk_slice(tensor, start, stop, 3)
    actual = attention_utils._chunk_slice(tensor, start, stop, 3)
    assert torch.equal(actual, expected)


def test_rigid_alignment_matches_upstream(upstream_diffusion_loss: ModuleType) -> None:
    generator = torch.Generator().manual_seed(761)
    true_coords = torch.randn(2, 8, 3, generator=generator)
    pred_coords = torch.randn(2, 8, 3, generator=generator)
    weights = torch.rand(2, 8, generator=generator)
    mask = torch.ones(2, 8)
    expected = upstream_diffusion_loss.weighted_rigid_align(
        true_coords,
        pred_coords,
        weights,
        mask,
    )
    actual = diffusion_loss.weighted_rigid_align(
        true_coords,
        pred_coords,
        weights,
        mask,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_smooth_lddt_matches_upstream(upstream_diffusion_loss: ModuleType) -> None:
    generator = torch.Generator().manual_seed(177)
    pred_coords = torch.randn(2, 10, 3, generator=generator)
    true_coords = torch.randn(2, 10, 3, generator=generator)
    is_nucleotide = torch.tensor([[0.0] * 5 + [1.0] * 5] * 2)
    coords_mask = torch.ones(2, 10)
    expected = upstream_diffusion_loss.smooth_lddt_loss(
        pred_coords,
        true_coords,
        is_nucleotide,
        coords_mask,
    )
    actual = diffusion_loss.smooth_lddt_loss(
        pred_coords,
        true_coords,
        is_nucleotide,
        coords_mask,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_confidence_scalar_helpers_match_upstream(upstream_package: ModuleType) -> None:
    del upstream_package
    reference = importlib.import_module("boltz.model.layers.confidence_utils")
    logits = torch.randn(2, 4, 50, generator=torch.Generator().manual_seed(93))
    expected = reference.compute_aggregated_metric(logits)
    actual = confidence.compute_aggregated_metric(logits)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    distances = torch.linspace(0, 32, 17)
    residues = torch.tensor([[25.0], [300.0]])
    expected_tm = reference.tm_function(distances, residues)
    actual_tm = confidence.tm_function(distances, residues)
    torch.testing.assert_close(actual_tm, expected_tm, rtol=0, atol=0)


def test_attention_state_and_forward_match_upstream(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    _install_import_only_dependency_stubs()
    reference_module = importlib.import_module("boltz.model.layers.triangular_attention.primitives")
    torch.manual_seed(12)
    reference = reference_module.Attention(8, 8, 8, 4, 2, gating=True)
    torch.manual_seed(12)
    local = primitives.Attention(8, 8, 8, 4, 2, gating=True)
    assert local.state_dict().keys() == reference.state_dict().keys()
    for name, tensor in reference.state_dict().items():
        assert torch.equal(local.state_dict()[name], tensor), name

    generator = torch.Generator().manual_seed(33)
    query = torch.randn(2, 5, 8, generator=generator)
    key_value = torch.randn(2, 7, 8, generator=generator)
    triangle_bias = torch.randn(2, 2, 5, 7, generator=generator)
    mask_bias = torch.zeros(2, 2, 5, 7)
    mask = torch.ones(2, 5, 7)
    expected = reference(query, key_value, triangle_bias, mask_bias, mask)
    actual = local(query, key_value, triangle_bias, mask_bias, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_transition_state_and_dense_or_chunked_forward_match_upstream(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    _install_import_only_dependency_stubs()
    reference_class = importlib.import_module("boltz.model.layers.transition").Transition
    local = transition.Transition(dim=8, hidden=10, out_dim=6).eval()
    reference = reference_class(dim=8, hidden=10, out_dim=6).eval()
    reference.load_state_dict(local.state_dict(), strict=True)
    assert reference.state_dict().keys() == local.state_dict().keys()

    # X is a deterministic transition input with shape (b, l, d).
    input_tensor = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(271))
    for chunk_size in (None, 4, 16):
        expected = reference(input_tensor, chunk_size=chunk_size)
        actual = local(input_tensor, chunk_size=chunk_size)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _fill_parameters(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))


def test_pair_biased_attention_matches_upstream_and_cache_contract(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    _install_import_only_dependency_stubs()
    reference_class = importlib.import_module("boltz.model.layers.attention").AttentionPairBias
    local = pair_attention.AttentionPairBias(8, 4, 2).eval()
    _fill_parameters(local, seed=281)
    reference = reference_class(8, 4, 2).eval()
    reference.load_state_dict(local.state_dict(), strict=True)

    generator = torch.Generator().manual_seed(283)
    sequence_states = torch.randn(2, 5, 8, generator=generator)
    pair_states = torch.randn(2, 5, 5, 4, generator=generator)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.float32)
    expected_cache: dict[str, torch.Tensor] = {}
    actual_cache: dict[str, torch.Tensor] = {}
    expected = reference(sequence_states, pair_states, mask, model_cache=expected_cache)
    actual = local(sequence_states, pair_states, mask, model_cache=actual_cache)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_cache["z"], expected_cache["z"], rtol=0, atol=0)

    replacement_pairs = torch.randn(2, 5, 5, 4, generator=generator)
    expected_cached = reference(
        sequence_states,
        replacement_pairs,
        mask,
        model_cache=expected_cache,
    )
    actual_cached = local(
        sequence_states,
        replacement_pairs,
        mask,
        model_cache=actual_cache,
    )
    torch.testing.assert_close(actual_cached, expected_cached, rtol=0, atol=0)


@pytest.mark.parametrize("compute_pair_bias", [False, True])
def test_pair_biased_cross_attention_matches_upstream(
    upstream_package: ModuleType,
    compute_pair_bias: bool,
) -> None:
    del upstream_package
    _install_import_only_dependency_stubs()
    reference_class = importlib.import_module("boltz.model.layers.attentionv2").AttentionPairBias
    kwargs = {
        "c_s": 8,
        "c_z": 4,
        "num_heads": 2,
        "compute_pair_bias": compute_pair_bias,
    }
    local = pair_attention_v2.AttentionPairBias(**kwargs).eval()
    _fill_parameters(local, seed=293)
    reference = reference_class(**kwargs).eval()
    reference.load_state_dict(local.state_dict(), strict=True)

    generator = torch.Generator().manual_seed(307)
    query_states = torch.randn(2, 3, 8, generator=generator)
    key_states = torch.randn(2, 5, 8, generator=generator)
    pair_width = 4 if compute_pair_bias else 2
    pair_states = torch.randn(2, 3, 5, pair_width, generator=generator)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.float32)
    expected = reference(query_states, pair_states, mask, key_states)
    actual = local(query_states, pair_states, mask, key_states)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_outer_product_mean_dense_and_chunked_match_upstream(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    _install_import_only_dependency_stubs()
    reference_class = importlib.import_module(
        "boltz.model.layers.outer_product_mean"
    ).OuterProductMean
    local = outer_product.OuterProductMean(c_in=5, c_hidden=3, c_out=7).eval()
    _fill_parameters(local, seed=311)
    reference = reference_class(c_in=5, c_hidden=3, c_out=7).eval()
    reference.load_state_dict(local.state_dict(), strict=True)

    generator = torch.Generator().manual_seed(313)
    msa_states = torch.randn(2, 3, 4, 5, generator=generator)
    mask = torch.tensor(
        [
            [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]],
            [[1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 0]],
        ],
        dtype=torch.float32,
    )
    for chunk_size in (None, 2, 8):
        expected = reference(msa_states, mask, chunk_size=chunk_size)
        actual = local(msa_states, mask, chunk_size=chunk_size)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("chunk_heads", [False, True])
def test_pair_weighted_averaging_matches_upstream(
    upstream_package: ModuleType,
    chunk_heads: bool,
) -> None:
    del upstream_package
    _install_import_only_dependency_stubs()
    reference_class = importlib.import_module(
        "boltz.model.layers.pair_averaging"
    ).PairWeightedAveraging
    kwargs = {"c_m": 6, "c_z": 5, "c_h": 2, "num_heads": 3}
    local = pair_averaging.PairWeightedAveraging(**kwargs).eval()
    _fill_parameters(local, seed=317)
    reference = reference_class(**kwargs).eval()
    reference.load_state_dict(local.state_dict(), strict=True)

    generator = torch.Generator().manual_seed(331)
    msa_states = torch.randn(2, 3, 4, 6, generator=generator)
    pair_states = torch.randn(2, 4, 4, 5, generator=generator)
    mask = torch.tensor(
        [
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    expected = reference(msa_states, pair_states, mask, chunk_heads=chunk_heads)
    actual = local(msa_states, pair_states, mask, chunk_heads=chunk_heads)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("class_name", "local_class"),
    [
        (
            "TriangleMultiplicationOutgoing",
            triangular_mult.TriangleMultiplicationOutgoing,
        ),
        (
            "TriangleMultiplicationIncoming",
            triangular_mult.TriangleMultiplicationIncoming,
        ),
    ],
)
def test_triangular_multiplication_matches_upstream(
    upstream_package: ModuleType,
    class_name: str,
    local_class: type[torch.nn.Module],
) -> None:
    del upstream_package
    _install_import_only_dependency_stubs()
    reference_class = getattr(
        importlib.import_module("boltz.model.layers.triangular_mult"),
        class_name,
    )
    local = local_class(dim=4).eval()
    _fill_parameters(local, seed=337)
    reference = reference_class(dim=4).eval()
    reference.load_state_dict(local.state_dict(), strict=True)

    pair_states = torch.randn(2, 3, 3, 4, generator=torch.Generator().manual_seed(347))
    mask = torch.tensor(
        [
            [[1, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
        ],
        dtype=torch.float32,
    )
    expected = reference(pair_states, mask)
    actual = local(pair_states, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("class_name", "local_class"),
    [
        (
            "TriangleAttentionStartingNode",
            triangle_attention.TriangleAttentionStartingNode,
        ),
        (
            "TriangleAttentionEndingNode",
            triangle_attention.TriangleAttentionEndingNode,
        ),
    ],
)
def test_triangle_attention_dense_and_chunked_match_upstream(
    upstream_package: ModuleType,
    class_name: str,
    local_class: type[torch.nn.Module],
) -> None:
    del upstream_package
    _install_import_only_dependency_stubs()
    reference_class = getattr(
        importlib.import_module("boltz.model.layers.triangular_attention.attention"),
        class_name,
    )
    local = local_class(c_in=8, c_hidden=4, no_heads=2).eval()
    _fill_parameters(local, seed=349)
    reference = reference_class(c_in=8, c_hidden=4, no_heads=2).eval()
    reference.load_state_dict(local.state_dict(), strict=True)

    pair_states = torch.randn(2, 3, 3, 8, generator=torch.Generator().manual_seed(353))
    mask = torch.tensor(
        [
            [[1, 1, 1], [1, 1, 1], [1, 1, 0]],
            [[1, 1, 1], [1, 1, 0], [1, 0, 0]],
        ],
        dtype=torch.float32,
    )
    for chunk_size in (None, 2):
        expected = reference(pair_states, mask, chunk_size=chunk_size)
        actual = local(pair_states, mask, chunk_size=chunk_size)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_conditioned_diffusion_transformer_matches_upstream(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    _install_import_only_dependency_stubs()
    reference_module = importlib.import_module("boltz.model.modules.transformersv2")
    kwargs = {
        "depth": 2,
        "heads": 2,
        "dim": 8,
        "dim_single_cond": 6,
        "post_layer_norm": True,
    }
    local = diffusion_transformers.DiffusionTransformer(**kwargs).eval()
    _fill_parameters(local, seed=359)
    reference = reference_module.DiffusionTransformer(**kwargs).eval()
    reference.load_state_dict(local.state_dict(), strict=True)
    assert reference.state_dict().keys() == local.state_dict().keys()

    generator = torch.Generator().manual_seed(367)
    activations = torch.randn(2, 4, 8, generator=generator)
    conditioning_states = torch.randn(2, 4, 6, generator=generator)
    pair_bias = torch.randn(2, 4, 4, 4, generator=generator)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.float32)
    expected = reference(activations, conditioning_states, pair_bias, mask)
    actual = local(activations, conditioning_states, pair_bias, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rotation_and_coordinate_augmentation_utilities_match_upstream(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    reference = importlib.import_module("boltz.model.modules.utils")
    for function_name in ("random_quaternions", "random_rotations"):
        torch.manual_seed(373)
        expected = getattr(reference, function_name)(5, dtype=torch.float64)
        torch.manual_seed(373)
        actual = getattr(module_utils, function_name)(5, dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    generator = torch.Generator().manual_seed(379)
    coordinates = torch.randn(2, 5, 3, generator=generator)
    second = torch.randn(2, 5, 3, generator=generator)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.float32)
    torch.manual_seed(383)
    expected = reference.center_random_augmentation(
        coordinates,
        mask,
        s_trans=0.4,
        return_second_coords=True,
        second_coords=second,
    )
    torch.manual_seed(383)
    actual = module_utils.center_random_augmentation(
        coordinates,
        mask,
        s_trans=0.4,
        return_second_coords=True,
        second_coords=second,
    )
    assert isinstance(actual, tuple) and isinstance(expected, tuple)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_exponential_moving_average_matches_upstream(upstream_package: ModuleType) -> None:
    del upstream_package
    reference_class = importlib.import_module("boltz.model.modules.utils").ExponentialMovingAverage
    local_parameters = [
        torch.nn.Parameter(torch.tensor([1.0, 2.0])),
        torch.nn.Parameter(torch.tensor([[3.0], [4.0]])),
    ]
    reference_parameters = [
        torch.nn.Parameter(parameter.detach().clone()) for parameter in local_parameters
    ]
    local = module_utils.ExponentialMovingAverage(local_parameters, decay=0.95)
    reference = reference_class(reference_parameters, decay=0.95)
    with torch.no_grad():
        for local_parameter, reference_parameter in zip(
            local_parameters,
            reference_parameters,
            strict=True,
        ):
            local_parameter.add_(0.75)
            reference_parameter.add_(0.75)
    local.update(local_parameters)
    reference.update(reference_parameters)
    assert local.num_updates == reference.num_updates
    for actual, expected in zip(
        local.shadow_params,
        reference.shadow_params,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_joint_and_pair_only_pairformer_layers_match_upstream(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    _install_import_only_dependency_stubs()
    reference_module = importlib.import_module("boltz.model.layers.pairformer")
    joint_kwargs = {
        "token_s": 8,
        "token_z": 4,
        "num_heads": 2,
        "dropout": 0.0,
        "pairwise_head_width": 2,
        "pairwise_num_heads": 2,
        "post_layer_norm": True,
        "v2": True,
    }
    local_joint = pairformer.PairformerLayer(**joint_kwargs).eval()
    _fill_parameters(local_joint, seed=389)
    reference_joint = reference_module.PairformerLayer(**joint_kwargs).eval()
    reference_joint.load_state_dict(local_joint.state_dict(), strict=True)
    assert reference_joint.state_dict().keys() == local_joint.state_dict().keys()

    generator = torch.Generator().manual_seed(397)
    sequence_states = torch.randn(2, 3, 8, generator=generator)
    pair_states = torch.randn(2, 3, 3, 4, generator=generator)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.float32)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    expected_s, expected_z = reference_joint(
        sequence_states,
        pair_states,
        mask,
        pair_mask,
        chunk_size_tri_attn=2,
    )
    actual_s, actual_z = local_joint(
        sequence_states,
        pair_states,
        mask,
        pair_mask,
        chunk_size_tri_attn=2,
    )
    torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)
    torch.testing.assert_close(actual_z, expected_z, rtol=0, atol=0)

    pair_kwargs = {
        "token_z": 4,
        "dropout": 0.0,
        "pairwise_head_width": 2,
        "pairwise_num_heads": 2,
    }
    local_pair = pairformer.PairformerNoSeqLayer(**pair_kwargs).eval()
    _fill_parameters(local_pair, seed=401)
    reference_pair = reference_module.PairformerNoSeqLayer(**pair_kwargs).eval()
    reference_pair.load_state_dict(local_pair.state_dict(), strict=True)
    expected_z = reference_pair(pair_states, pair_mask, chunk_size_tri_attn=2)
    actual_z = local_pair(pair_states, pair_mask, chunk_size_tri_attn=2)
    torch.testing.assert_close(actual_z, expected_z, rtol=0, atol=0)


def test_diffusion_conditioning_state_schema_matches_upstream(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    _install_import_only_dependency_stubs()
    reference_class = importlib.import_module(
        "boltz.model.modules.diffusion_conditioning"
    ).DiffusionConditioning
    kwargs = {
        "token_s": 8,
        "token_z": 6,
        "atom_s": 4,
        "atom_z": 5,
        "atoms_per_window_queries": 2,
        "atoms_per_window_keys": 4,
        "atom_encoder_depth": 2,
        "atom_encoder_heads": 2,
        "token_transformer_depth": 3,
        "token_transformer_heads": 2,
        "atom_decoder_depth": 2,
        "atom_decoder_heads": 2,
        "atom_feature_dim": 10,
        "conditioning_transition_layers": 1,
    }
    torch.manual_seed(41)
    reference = reference_class(**kwargs)
    torch.manual_seed(41)
    local = conditioning.DiffusionConditioning(**kwargs)
    assert local.state_dict().keys() == reference.state_dict().keys()
    for name, tensor in reference.state_dict().items():
        assert local.state_dict()[name].shape == tensor.shape, name


def _to_device(
    tree: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in tree.items()}


def _load_local_state(reference: torch.nn.Module, local: torch.nn.Module) -> None:
    assert reference.state_dict().keys() == local.state_dict().keys()
    reference.load_state_dict(local.state_dict(), strict=True)


@pytest.mark.gpu
def test_encoder_components_match_upstream_on_gpu(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    assert torch.cuda.is_available(), "CUDA is required for Boltz encoder parity"
    _install_import_only_dependency_stubs()
    reference_module = importlib.import_module("boltz.model.modules.encodersv2")
    device = torch.device("cuda")

    torch.manual_seed(101)
    local_fourier = encoders.FourierEmbedding(8).to(device)
    reference_fourier = reference_module.FourierEmbedding(8).to(device)
    _load_local_state(reference_fourier, local_fourier)
    times = torch.tensor([0.1, 0.7], device=device)
    torch.testing.assert_close(
        local_fourier(times),
        reference_fourier(times),
        rtol=0,
        atol=0,
    )

    relative_kwargs = {
        "token_z": 7,
        "r_max": 4,
        "s_max": 2,
        "fix_sym_check": True,
        "cyclic_pos_enc": True,
    }
    local_relative = encoders.RelativePositionEncoder(**relative_kwargs).to(device)
    reference_relative = reference_module.RelativePositionEncoder(**relative_kwargs).to(device)
    _load_local_state(reference_relative, local_relative)
    relative_features = {
        "asym_id": torch.tensor([[0, 0, 1, 1]], device=device),
        "residue_index": torch.tensor([[0, 1, 0, 1]], device=device),
        "entity_id": torch.tensor([[0, 0, 1, 1]], device=device),
        "token_index": torch.tensor([[0, 1, 2, 3]], device=device),
        "sym_id": torch.tensor([[0, 1, 0, 1]], device=device),
        "cyclic_period": torch.tensor([[0.0, 2.0, 0.0, 0.0]], device=device),
    }
    torch.testing.assert_close(
        local_relative(relative_features),
        reference_relative(relative_features),
        rtol=0,
        atol=0,
    )

    local_single = encoders.SingleConditioning(
        sigma_data=16.0,
        token_s=4,
        dim_fourier=6,
        num_transitions=2,
    ).to(device)
    reference_single = reference_module.SingleConditioning(
        sigma_data=16.0,
        token_s=4,
        dim_fourier=6,
        num_transitions=2,
    ).to(device)
    _load_local_state(reference_single, local_single)
    generator = torch.Generator(device=device).manual_seed(401)
    s_trunk = torch.randn(2, 5, 4, device=device, generator=generator)
    s_inputs = torch.randn(2, 5, 4, device=device, generator=generator)
    local_single_output = local_single(times, s_trunk, s_inputs)
    reference_single_output = reference_single(times, s_trunk, s_inputs)
    torch.testing.assert_close(
        local_single_output[0],
        reference_single_output[0],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        local_single_output[1],
        reference_single_output[1],
        rtol=0,
        atol=0,
    )

    local_pair = encoders.PairwiseConditioning(6, 3, num_transitions=2).to(device)
    reference_pair = reference_module.PairwiseConditioning(
        6,
        3,
        num_transitions=2,
    ).to(device)
    _load_local_state(reference_pair, local_pair)
    pair_trunk = torch.randn(2, 5, 5, 6, device=device, generator=generator)
    relative_pair = torch.randn(2, 5, 5, 3, device=device, generator=generator)
    torch.testing.assert_close(
        local_pair(pair_trunk, relative_pair),
        reference_pair(pair_trunk, relative_pair),
        rtol=0,
        atol=0,
    )

    local_index = encoders.get_indexing_matrix(3, 4, 8, device)
    reference_index = reference_module.get_indexing_matrix(3, 4, 8, device)
    assert torch.equal(local_index, reference_index)
    single = torch.randn(2, 12, 5, device=device, generator=generator)
    torch.testing.assert_close(
        encoders.single_to_keys(single, local_index, 4, 8),
        reference_module.single_to_keys(single, reference_index, 4, 8),
        rtol=0,
        atol=0,
    )


@pytest.mark.gpu
def test_atom_encoder_matches_upstream_on_gpu(upstream_package: ModuleType) -> None:
    del upstream_package
    assert torch.cuda.is_available(), "CUDA is required for Boltz atom parity"
    _install_import_only_dependency_stubs()
    reference_module = importlib.import_module("boltz.model.modules.encodersv2")
    device = torch.device("cuda")
    features, _ = build_boltz2_features("ACDE")
    features = _to_device(features, device)
    kwargs = {
        "atom_s": 8,
        "atom_z": 6,
        "token_s": 8,
        "token_z": 6,
        "atoms_per_window_queries": 32,
        "atoms_per_window_keys": 128,
        "atom_feature_dim": 388,
        "structure_prediction": True,
    }
    local = encoders.AtomEncoder(**kwargs).to(device)
    reference = reference_module.AtomEncoder(**kwargs).to(device)
    _load_local_state(reference, local)
    generator = torch.Generator(device=device).manual_seed(812)
    num_tokens = features["token_index"].shape[1]
    s_trunk = torch.randn(1, num_tokens, 8, device=device, generator=generator)
    z_trunk = torch.randn(
        1,
        num_tokens,
        num_tokens,
        6,
        device=device,
        generator=generator,
    )
    local_output = local(features, s_trunk=s_trunk, z=z_trunk)
    reference_output = reference(features, s_trunk=s_trunk, z=z_trunk)
    for local_tensor, reference_tensor in zip(local_output[:3], reference_output[:3], strict=True):
        torch.testing.assert_close(local_tensor, reference_tensor, rtol=0, atol=0)
    probe = torch.randn(
        1,
        features["ref_pos"].shape[1],
        3,
        device=device,
        generator=generator,
    )
    torch.testing.assert_close(
        local_output[3](probe),
        reference_output[3](probe),
        rtol=0,
        atol=0,
    )


@pytest.mark.gpu
def test_atom_attention_encoder_and_decoder_match_upstream_on_gpu(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    assert torch.cuda.is_available(), "CUDA is required for Boltz atom parity"
    _install_import_only_dependency_stubs()
    reference_module = importlib.import_module("boltz.model.modules.encodersv2")
    device = torch.device("cuda")
    features, _ = build_boltz2_features("ACDE")
    features = _to_device(features, device)
    generator = torch.Generator(device=device).manual_seed(912)
    num_atoms = features["ref_pos"].shape[1]
    num_tokens = features["token_index"].shape[1]

    atom_kwargs = {
        "atom_s": 8,
        "atom_z": 2,
        "token_s": 8,
        "token_z": 6,
        "atoms_per_window_queries": 32,
        "atoms_per_window_keys": 128,
        "atom_feature_dim": 388,
        "structure_prediction": True,
    }
    atom_encoder = encoders.AtomEncoder(**atom_kwargs).to(device)
    s_trunk = torch.randn(1, num_tokens, 8, device=device, generator=generator)
    z_trunk = torch.randn(
        1,
        num_tokens,
        num_tokens,
        6,
        device=device,
        generator=generator,
    )
    q, c, _, to_keys = atom_encoder(features, s_trunk=s_trunk, z=z_trunk)
    atom_bias = torch.randn(
        1,
        num_atoms // 32,
        32,
        128,
        2,
        device=device,
        generator=generator,
    )

    encoder_kwargs = {
        "atom_s": 8,
        "token_s": 8,
        "atoms_per_window_queries": 32,
        "atoms_per_window_keys": 128,
        "atom_encoder_depth": 1,
        "atom_encoder_heads": 2,
        "structure_prediction": True,
    }
    local_encoder = encoders.AtomAttentionEncoder(**encoder_kwargs).to(device)
    reference_encoder = reference_module.AtomAttentionEncoder(**encoder_kwargs).to(device)
    _load_local_state(reference_encoder, local_encoder)
    coordinates = torch.randn(1, num_atoms, 3, device=device, generator=generator)
    local_encoded = local_encoder(
        features,
        q,
        c,
        atom_bias,
        to_keys,
        r=coordinates,
    )
    reference_encoded = reference_encoder(
        features,
        q,
        c,
        atom_bias,
        to_keys,
        r=coordinates,
    )
    for local_tensor, reference_tensor in zip(
        local_encoded[:3],
        reference_encoded[:3],
        strict=True,
    ):
        torch.testing.assert_close(local_tensor, reference_tensor, rtol=0, atol=0)

    decoder_kwargs = {
        "atom_s": 8,
        "token_s": 8,
        "attn_window_queries": 32,
        "attn_window_keys": 128,
        "atom_decoder_depth": 1,
        "atom_decoder_heads": 2,
    }
    local_decoder = encoders.AtomAttentionDecoder(**decoder_kwargs).to(device)
    reference_decoder = reference_module.AtomAttentionDecoder(**decoder_kwargs).to(device)
    _load_local_state(reference_decoder, local_decoder)
    token_update = torch.randn(
        1,
        num_tokens,
        16,
        device=device,
        generator=generator,
    )
    local_decoded = local_decoder(
        token_update,
        local_encoded[1],
        local_encoded[2],
        atom_bias,
        features,
        to_keys,
    )
    reference_decoded = reference_decoder(
        token_update,
        reference_encoded[1],
        reference_encoded[2],
        atom_bias,
        features,
        to_keys,
    )
    torch.testing.assert_close(local_decoded, reference_decoded, rtol=0, atol=0)


def _assert_tree_equal(local: object, reference: object) -> None:
    if isinstance(reference, torch.Tensor):
        assert isinstance(local, torch.Tensor)
        assert local.dtype == reference.dtype
        assert local.device == reference.device
        torch.testing.assert_close(local, reference, rtol=0, atol=0, equal_nan=True)
        return
    if isinstance(reference, (tuple, list)):
        assert isinstance(local, type(reference))
        assert len(local) == len(reference)
        for local_item, reference_item in zip(local, reference, strict=True):
            _assert_tree_equal(local_item, reference_item)
        return
    assert local == reference


@pytest.mark.gpu
def test_potential_geometry_and_gradients_match_upstream_on_gpu(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    assert torch.cuda.is_available(), "CUDA is required for Boltz potential parity"
    _install_import_only_dependency_stubs()
    reference_module = importlib.import_module("boltz.model.potentials.potentials")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1211)
    coordinates = torch.randn(2, 9, 3, device=device, generator=generator)

    pair_index = torch.tensor([[0, 2, 4], [1, 5, 8]], device=device)
    for local, reference, is_distance in (
        (
            potentials.ConnectionsPotential(),
            reference_module.ConnectionsPotential(),
            True,
        ),
        (
            potentials.ChiralAtomPotential(),
            reference_module.ChiralAtomPotential(),
            False,
        ),
        (
            potentials.PlanarBondPotential(),
            reference_module.PlanarBondPotential(),
            False,
        ),
    ):
        index = (
            pair_index
            if is_distance
            else torch.tensor(
                [[0, 1], [1, 2], [3, 4], [5, 6]],
                device=device,
            )
        )
        _assert_tree_equal(
            local.compute_variable(coordinates, index, compute_gradient=False),
            reference.compute_variable(coordinates, index, compute_gradient=False),
        )
        _assert_tree_equal(
            local.compute_variable(coordinates, index, compute_gradient=True),
            reference.compute_variable(coordinates, index, compute_gradient=True),
        )

    values = torch.tensor([[-2.0, 0.5, 4.0]], device=device)
    k = torch.tensor([1.0, 2.0, 3.0], device=device)
    lower = torch.tensor([-1.0, 0.0, 1.0], device=device)
    upper = torch.tensor([1.0, 2.0, 3.0], device=device)
    _assert_tree_equal(
        potentials.FlatBottomPotential.compute_function(
            object(),
            values,
            k,
            lower,
            upper,
            compute_derivative=True,
        ),
        reference_module.FlatBottomPotential.compute_function(
            object(),
            values,
            k,
            lower,
            upper,
            compute_derivative=True,
        ),
    )

    connection_features = {"connected_atom_index": pair_index[None]}
    parameters = {"buffer": 2.5}
    local_connection = potentials.ConnectionsPotential()
    reference_connection = reference_module.ConnectionsPotential()
    _assert_tree_equal(
        local_connection.compute(coordinates, connection_features, parameters),
        reference_connection.compute(coordinates, connection_features, parameters),
    )
    _assert_tree_equal(
        local_connection.compute_gradient(
            coordinates,
            connection_features,
            parameters,
        ),
        reference_connection.compute_gradient(
            coordinates,
            connection_features,
            parameters,
        ),
    )


@pytest.mark.gpu
def test_potential_argument_builders_match_upstream_on_gpu(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    assert torch.cuda.is_available(), "CUDA is required for Boltz potential parity"
    _install_import_only_dependency_stubs()
    reference_module = importlib.import_module("boltz.model.potentials.potentials")
    device = torch.device("cuda")
    atom_to_token = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )[None]
    ref_element = torch.zeros(1, 8, 128, device=device)
    ref_element[..., 6] = 1
    features = {
        "atom_to_token": atom_to_token,
        "asym_id": torch.tensor([[0, 0, 1, 1]], device=device),
        "atom_pad_mask": torch.ones(1, 8, device=device),
        "ref_element": ref_element,
        "connected_chain_index": torch.tensor([[[0], [1]]], device=device),
        "symmetric_chain_index": torch.tensor([[[0], [1]]], device=device),
        "rdkit_bounds_index": torch.tensor(
            [[[0, 2, 4], [1, 3, 5]]],
            device=device,
        ),
        "rdkit_lower_bounds": torch.tensor([[1.0, 1.5, 2.0]], device=device),
        "rdkit_upper_bounds": torch.tensor([[2.0, 2.5, 3.0]], device=device),
        "rdkit_bounds_bond_mask": torch.tensor(
            [[True, False, True]],
            device=device,
        ),
        "rdkit_bounds_angle_mask": torch.tensor(
            [[False, True, True]],
            device=device,
        ),
        "stereo_bond_index": torch.tensor(
            [[[[0, 1], [1, 2], [2, 3], [3, 4]]]],
            device=device,
        ).squeeze(1),
        "stereo_bond_orientations": torch.tensor([[True, False]], device=device),
        "chiral_atom_index": torch.tensor(
            [[[[0, 1], [1, 2], [2, 3], [3, 4]]]],
            device=device,
        ).squeeze(1),
        "chiral_atom_orientations": torch.tensor([[False, True]], device=device),
        "planar_bond_index": torch.tensor(
            [[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]],
            device=device,
        ),
        "contact_pair_index": torch.tensor(
            [[[0, 2, 4], [1, 3, 5]]],
            device=device,
        ),
        "contact_union_index": torch.tensor([[0, 0, 1]], device=device),
        "contact_negation_mask": torch.tensor(
            [[False, True, False]],
            device=device,
        ),
        "contact_thresholds": torch.tensor([[3.0, 4.0, 5.0]], device=device),
    }
    cases = (
        (
            "PoseBustersPotential",
            {"bond_buffer": 0.1, "angle_buffer": 0.2, "clash_buffer": 0.15},
        ),
        ("VDWOverlapPotential", {"buffer": 0.225}),
        ("SymmetricChainCOMPotential", {"buffer": 1.5}),
        ("StereoBondPotential", {"buffer": 0.3}),
        ("ChiralAtomPotential", {"buffer": 0.3}),
        ("PlanarBondPotential", {"buffer": 0.2}),
        ("ContactPotentital", {"union_lambda": 2.0}),
    )
    for class_name, parameters in cases:
        local = getattr(potentials, class_name)()
        reference = getattr(reference_module, class_name)()
        _assert_tree_equal(
            local.compute_args(features, parameters),
            reference.compute_args(features, parameters),
        )


def test_potential_stack_and_schedules_match_upstream(
    upstream_package: ModuleType,
) -> None:
    del upstream_package
    _install_import_only_dependency_stubs()
    reference_module = importlib.import_module("boltz.model.potentials.potentials")
    steering = {
        "fk_steering": True,
        "physical_guidance_update": True,
        "contact_guidance_update": True,
    }
    local_stack = potentials.get_potentials(steering, boltz2=True)
    reference_stack = reference_module.get_potentials(steering, boltz2=True)
    assert [type(item).__name__ for item in local_stack] == [
        type(item).__name__ for item in reference_stack
    ]
    for local, reference in zip(local_stack, reference_stack, strict=True):
        for time in (0.0, 0.5, 1.0):
            assert local.compute_parameters(time) == reference.compute_parameters(time)
