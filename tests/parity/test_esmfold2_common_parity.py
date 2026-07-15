"""Exact differentials for the independently organized ESMFold2 core blocks."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from fastplms.models.esmfold2 import modeling_esmfold2_common as local

pytestmark = [pytest.mark.compliance, pytest.mark.gpu, pytest.mark.structure]

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = ROOT / "vendor/upstream/biohub-transformers/src/transformers/models/esmfold2"
_MISSING = object()


def _package(name: str) -> types.ModuleType:
    package = types.ModuleType(name)
    package.__path__ = []  # type: ignore[attr-defined]
    return package


@contextmanager
def _temporary_modules(modules: dict[str, types.ModuleType]) -> Iterator[None]:
    previous = {name: sys.modules.get(name, _MISSING) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module  # type: ignore[assignment]


def _load_source(name: str, path: Path, aliases: dict[str, types.ModuleType]) -> types.ModuleType:
    assert path.is_file(), f"pinned source is missing: {path}"
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    with _temporary_modules({**aliases, name: module}):
        specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def official() -> types.ModuleType:
    import transformers.configuration_utils as configuration_utils

    root_name = "_fastplms_pinned_esmfold2_common"
    aliases = {
        root_name: _package(root_name),
        f"{root_name}.models": _package(f"{root_name}.models"),
        f"{root_name}.models.esmfold2": _package(f"{root_name}.models.esmfold2"),
        f"{root_name}.configuration_utils": configuration_utils,
    }
    config_name = f"{root_name}.models.esmfold2.configuration_esmfold2"
    config = _load_source(config_name, OFFICIAL_ROOT / "configuration_esmfold2.py", aliases)
    aliases[config_name] = config
    return _load_source(
        f"{root_name}.models.esmfold2.modeling_esmfold2_common",
        OFFICIAL_ROOT / "modeling_esmfold2_common.py",
        aliases,
    )


def _assert_tensor_exact(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert torch.equal(actual, expected)


def _alias_groups(module: nn.Module) -> set[tuple[str, ...]]:
    names_by_id: dict[int, list[str]] = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        names_by_id.setdefault(id(parameter), []).append(name)
    return {tuple(names) for names in names_by_id.values() if len(names) > 1}


_STATE_SPECS: tuple[tuple[str, tuple[Any, ...], dict[str, Any]], ...] = (
    ("TransitionLayer", (8, 2), {}),
    ("AdaptiveLayerNorm", (8, 6), {}),
    ("FourierEmbedding", (8,), {}),
    ("SwiGLUMLP", (8, 2), {}),
    ("SWA3DRoPEAttention", (32, 4), {"half_window": 2}),
    ("AttentionPairBias", (8, 6, 2), {"d_cond": 8}),
    ("ConditionedTransitionBlock", (8,), {"d_cond": 8}),
    ("DiffusionTransformer", (8, 6, 2, 2), {"d_cond": 8}),
    ("RowAttentionPooling", (6, 8), {}),
    ("ResIdxAsymIdSymIdEntityIdEncoding", (2, 1, 8), {}),
    ("SingleToPair", (8, 4, 6), {}),
    ("LanguageModelShim", (8, 12, 2), {}),
    ("TriangleMultiplicativeBlock", (8, 4, "outgoing"), {}),
    ("TriangleMultiplicativeUpdate", (8, True), {}),
    ("Transition", (8, 2), {}),
    ("PairUpdateBlock", (8, 2), {}),
    ("FoldingTrunk", (2, 8, 2), {}),
    ("OuterProductMean", (6, 4, 8), {}),
    ("MSAPairWeightedAveraging", (6, 8, 2, 4), {}),
)


@pytest.mark.parametrize(("name", "args", "kwargs"), _STATE_SPECS, ids=lambda x: x)
def test_state_schema_and_aliases_match_pinned_biohub(
    official: types.ModuleType,
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    torch.manual_seed(1907)
    expected = getattr(official, name)(*args, **kwargs)
    torch.manual_seed(1907)
    actual = getattr(local, name)(*args, **kwargs)
    actual_state = actual.state_dict()
    expected_state = expected.state_dict()
    assert tuple(actual_state) == tuple(expected_state)
    for key in expected_state:
        _assert_tensor_exact(actual_state[key], expected_state[key])
    assert _alias_groups(actual) == _alias_groups(expected)


def _paired_modules(
    official: types.ModuleType,
    name: str,
    *args: Any,
    **kwargs: Any,
) -> tuple[nn.Module, nn.Module]:
    expected = getattr(official, name)(*args, **kwargs).eval().cuda()
    actual = getattr(local, name)(*args, **kwargs).eval().cuda()
    actual.load_state_dict(expected.state_dict(), strict=True)
    return actual, expected


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
def test_projection_and_pair_blocks_match_exactly(
    official: types.ModuleType, dtype: torch.dtype
) -> None:
    assert torch.cuda.is_available(), "the pinned common-module differential needs CUDA"
    torch.manual_seed(8675309)
    with (
        torch.no_grad(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16),
    ):
        actual_lm, expected_lm = _paired_modules(official, "LanguageModelShim", 8, 12, 2)
        actual_lm.to(dtype=dtype)
        expected_lm.to(dtype=dtype)
        hidden_states = torch.randn(2, 5, 3, 12, device="cuda", dtype=dtype)
        _assert_tensor_exact(actual_lm(hidden_states), expected_lm(hidden_states))
        sequence_projection = actual_lm.project_sequence(hidden_states)
        expected_projection = (
            expected_lm.base_z_combine.softmax(0) @ expected_lm.base_z_linear(hidden_states)
        ).squeeze(-2)
        _assert_tensor_exact(sequence_projection, expected_projection)

        actual_triangle, expected_triangle = _paired_modules(
            official, "TriangleMultiplicativeBlock", 8, 4, "outgoing"
        )
        actual_triangle.to(dtype=dtype)
        expected_triangle.to(dtype=dtype)
        actual_triangle.set_chunk_size(None)
        expected_triangle.set_chunk_size(None)
        pair = torch.randn(1, 4, 4, 8, device="cuda", dtype=dtype)
        pair_mask = torch.tensor(
            [[[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]]],
            device="cuda",
            dtype=torch.bool,
        )
        _assert_tensor_exact(actual_triangle(pair, pair_mask), expected_triangle(pair, pair_mask))

        actual_trunk, expected_trunk = _paired_modules(official, "FoldingTrunk", 2, 8, 2)
        actual_trunk.to(dtype=dtype)
        expected_trunk.to(dtype=dtype)
        actual_trunk.set_chunk_size(None)
        expected_trunk.set_chunk_size(None)
        _assert_tensor_exact(actual_trunk(pair, pair_mask), expected_trunk(pair, pair_mask))


def test_attention_and_diffusion_blocks_match_exactly(
    official: types.ModuleType,
) -> None:
    torch.manual_seed(4401)
    with torch.no_grad():
        actual, expected = _paired_modules(official, "DiffusionTransformer", 8, 6, 2, 2, d_cond=8)
        token = torch.randn(2, 5, 8, device="cuda")
        condition = torch.randn(2, 5, 8, device="cuda")
        pair = torch.randn(2, 5, 5, 6, device="cuda")
        mask = torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
            device="cuda",
            dtype=torch.bool,
        )
        actual_out, actual_intermediates = actual(
            token, condition, pair, attention_mask=mask, return_intermediates=True
        )
        expected_out, expected_intermediates = expected(
            token, condition, pair, attention_mask=mask, return_intermediates=True
        )
        _assert_tensor_exact(actual_out, expected_out)
        assert len(actual_intermediates) == len(expected_intermediates)
        for actual_value, expected_value in zip(
            actual_intermediates, expected_intermediates, strict=True
        ):
            _assert_tensor_exact(actual_value, expected_value)


def test_input_pair_and_msa_blocks_match_exactly(
    official: types.ModuleType,
) -> None:
    torch.manual_seed(923)
    with torch.no_grad():
        actual_relative, expected_relative = _paired_modules(
            official, "ResIdxAsymIdSymIdEntityIdEncoding", 2, 1, 8
        )
        residue_index = torch.tensor([[0, 1, 2, 0]], device="cuda")
        asym_id = torch.tensor([[0, 0, 0, 1]], device="cuda")
        sym_id = torch.tensor([[0, 0, 0, 1]], device="cuda")
        entity_id = torch.tensor([[0, 0, 0, 0]], device="cuda")
        token_index = torch.tensor([[0, 1, 2, 0]], device="cuda")
        inputs = (residue_index, asym_id, sym_id, entity_id, token_index)
        _assert_tensor_exact(actual_relative(*inputs), expected_relative(*inputs))

        actual_opm, expected_opm = _paired_modules(official, "OuterProductMean", 6, 4, 8)
        actual_opm.set_chunk_size(None)
        expected_opm.set_chunk_size(None)
        msa = torch.randn(2, 4, 3, 6, device="cuda")
        msa_mask = torch.tensor(
            [
                [[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]],
                [[1, 1, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0]],
            ],
            device="cuda",
            dtype=torch.bool,
        )
        _assert_tensor_exact(actual_opm(msa, msa_mask), expected_opm(msa, msa_mask))

        actual_avg, expected_avg = _paired_modules(official, "MSAPairWeightedAveraging", 6, 8, 2, 4)
        pair = torch.randn(2, 4, 4, 8, device="cuda")
        pair_mask = torch.ones(2, 4, 4, device="cuda", dtype=torch.bool)
        _assert_tensor_exact(actual_avg(msa, pair, pair_mask), expected_avg(msa, pair, pair_mask))


def test_tensor_utilities_match_pinned_biohub(official: types.ModuleType) -> None:
    token = torch.arange(48, device="cuda", dtype=torch.float32).reshape(2, 4, 6)
    atom_to_token = torch.tensor([[0, 0, 2, 3, 3], [0, 1, 1, 2, 3]], device="cuda")
    for name, args in (
        ("gather_token_to_atom", (token, atom_to_token)),
        (
            "scatter_atom_to_token",
            (
                torch.randn(2, 5, 6, device="cuda"),
                atom_to_token,
                4,
                torch.tensor(
                    [[1, 1, 1, 1, 0], [1, 1, 0, 1, 1]],
                    device="cuda",
                    dtype=torch.bool,
                ),
            ),
        ),
        (
            "gather_rep_atom_coords",
            (
                torch.randn(2, 5, 3, device="cuda"),
                torch.tensor([[0, 2, 4], [1, 3, 4]], device="cuda"),
            ),
        ),
    ):
        _assert_tensor_exact(getattr(local, name)(*args), getattr(official, name)(*args))


def test_atom_rope_attention_and_rigid_alignment_match_exactly(
    official: types.ModuleType,
) -> None:
    torch.manual_seed(314159)
    ref_pos = torch.randn(2, 7, 3, device="cuda")
    space_uid = torch.tensor([[0, 0, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 5, 5]], device="cuda")
    actual_cos, actual_sin = local.build_3d_rope(
        ref_pos, space_uid, head_dim=16, n_spatial_per_axis=2, n_uid_pairs=1
    )
    expected_cos, expected_sin = official.build_3d_rope(
        ref_pos, space_uid, head_dim=16, n_spatial_per_axis=2, n_uid_pairs=1
    )
    _assert_tensor_exact(actual_cos, expected_cos)
    _assert_tensor_exact(actual_sin, expected_sin)

    with torch.no_grad():
        actual_attention, expected_attention = _paired_modules(
            official, "SWA3DRoPEAttention", 64, 4, half_window=2
        )
        atom = torch.randn(2, 7, 64, device="cuda")
        _assert_tensor_exact(
            actual_attention(atom, (actual_cos, actual_sin)),
            expected_attention(atom, (expected_cos, expected_sin)),
        )

    mobile = torch.randn(2, 7, 3, device="cuda")
    target = torch.randn(2, 7, 3, device="cuda")
    weights = torch.rand(2, 7, device="cuda")
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
        device="cuda",
        dtype=torch.bool,
    )
    _assert_tensor_exact(
        local.DiffusionStructureHead._weighted_rigid_align(mobile, target, weights, mask),
        official.DiffusionStructureHead._weighted_rigid_align(mobile, target, weights, mask),
    )


def test_shared_public_class_surface_is_preserved(official: types.ModuleType) -> None:
    def classes(module: types.ModuleType) -> set[str]:
        return {
            name
            for name, value in vars(module).items()
            if inspect.isclass(value) and value.__module__ == module.__name__
        }

    assert classes(local) == classes(official)
