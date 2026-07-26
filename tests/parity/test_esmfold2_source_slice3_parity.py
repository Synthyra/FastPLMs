"""Differential contracts for independently organized ESMFold2 utilities."""

from __future__ import annotations

import importlib.util
import io
import sys
import types
import biotite.structure as bs
import numpy as np
import pytest
import torch
import zstandard
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastplms.models.esmfold2 import esmfold2_affine3d as local_affine
from fastplms.models.esmfold2 import esmfold2_misc as local_misc
from fastplms.models.esmfold2 import esmfold2_mmcif_parsing as local_mmcif
from fastplms.models.esmfold2 import esmfold2_msa as local_msa
from fastplms.models.esmfold2 import esmfold2_msa_filter_sequences as local_filter
from fastplms.models.esmfold2 import esmfold2_parsing as local_parsing
from fastplms.models.esmfold2 import esmfold2_residue_constants as local_residues
from fastplms.models.esmfold2 import esmfold2_sequential_dataclass as local_sequential
from fastplms.models.esmfold2 import esmfold2_system as local_system
from fastplms.models.esmfold2 import esmfold2_utils_types as local_types
from fastplms.models.esmfold2.esmfold2_constants_esm3 import CHAIN_BREAK_STR


pytestmark = [pytest.mark.compliance, pytest.mark.gpu, pytest.mark.structure]

ROOT = Path(__file__).resolve().parents[2]
BIOHUB_ESM = ROOT / "vendor/upstream/biohub-esm/esm"
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


def _load_source(
    module_name: str,
    path: Path,
    aliases: dict[str, types.ModuleType],
) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with _temporary_modules({**aliases, module_name: module}):
        spec.loader.exec_module(module)
    return module


def _base_aliases() -> dict[str, types.ModuleType]:
    return {
        "esm": _package("esm"),
        "esm.utils": _package("esm.utils"),
        "esm.utils.constants": _package("esm.utils.constants"),
        "esm.utils.msa": _package("esm.utils.msa"),
        "esm.utils.structure": _package("esm.utils.structure"),
    }


def _official_misc() -> types.ModuleType:
    constants = types.ModuleType("esm.utils.constants.esm3")
    constants.CHAIN_BREAK_STR = CHAIN_BREAK_STR
    zstd_adapter = types.ModuleType("zstd")
    decompress = zstandard.ZstdDecompressor().decompress
    zstd_adapter.decompress = decompress  # type: ignore[attr-defined]
    zstd_adapter.ZSTD_uncompress = decompress  # type: ignore[attr-defined]
    return _load_source(
        "_fastplms_pinned_biohub_misc",
        BIOHUB_ESM / "utils/misc.py",
        {
            **_base_aliases(),
            "esm.utils.constants.esm3": constants,
            "esm.utils.types": local_types,
            "zstd": zstd_adapter,
        },
    )


def _official_affine() -> types.ModuleType:
    return _load_source(
        "_fastplms_pinned_biohub_affine3d",
        BIOHUB_ESM / "utils/structure/affine3d.py",
        {**_base_aliases(), "esm.utils.misc": local_misc},
    )


def _official_msa() -> types.ModuleType:
    return _load_source(
        "_fastplms_pinned_biohub_msa",
        BIOHUB_ESM / "utils/msa/msa.py",
        {
            **_base_aliases(),
            "esm.utils.misc": local_misc,
            "esm.utils.msa.filter_sequences": local_filter,
            "esm.utils.parsing": local_parsing,
            "esm.utils.sequential_dataclass": local_sequential,
            "esm.utils.system": local_system,
        },
    )


def _official_mmcif() -> types.ModuleType:
    return _load_source(
        "_fastplms_pinned_biohub_mmcif",
        BIOHUB_ESM / "utils/structure/mmcif_parsing.py",
        {**_base_aliases(), "esm.utils.residue_constants": local_residues},
    )


def _assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # actual: (...), expected: (...)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def _assert_array_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_misc_tensor_contracts_match_pinned_biohub(device: str) -> None:
    assert torch.cuda.is_available(), "the utility parity suite requires CUDA"
    official = _official_misc()
    torch.manual_seed(730)
    # values: (2, 4)
    values = torch.randn((2, 4), device=device)
    _assert_tensor_equal(
        local_misc.rbf(values, -2.0, 3.0, 9),
        official.rbf(values, -2.0, 3.0, 9),
    )

    # data: (2, 3, 5, 2)
    data = torch.arange(2 * 3 * 5 * 2, device=device).reshape(2, 3, 5, 2)
    # indices: (2, 3, 2)
    indices = torch.tensor(
        [[[0, 3], [2, 1], [4, 0]], [[4, 1], [0, 2], [3, 3]]],
        device=device,
    )
    _assert_tensor_equal(
        local_misc.batched_gather(data, indices, dim=2, no_batch_dims=2),
        official.batched_gather(data, indices, dim=2, no_batch_dims=2),
    )

    # coords: (2, 3, 3)
    coords = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ],
        device=device,
    )
    # coord_mask: (2, 3)
    coord_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool, device=device)
    # padding_mask: (2, 3)
    padding_mask = torch.tensor([[0, 0, 1], [0, 0, 0]], dtype=torch.bool, device=device)
    # sequence_id: (2, 3)
    sequence_id = torch.tensor([[0, 0, 0], [0, 1, 1]], device=device)
    actual_edges = local_misc.knn_graph(
        coords,
        coord_mask,
        padding_mask,
        sequence_id,
        no_knn=3,
    )
    expected_edges = official.knn_graph(
        coords,
        coord_mask,
        padding_mask,
        sequence_id,
        no_knn=3,
    )
    for actual, expected in zip(actual_edges, expected_edges, strict=True):
        _assert_tensor_equal(actual, expected)

    rows = [torch.arange(3, device=device), torch.arange(5, device=device)]
    _assert_tensor_equal(
        local_misc.stack_variable_length_tensors(rows, -1),
        official.stack_variable_length_tensors(rows, -1),
    )
    # packed: (5, 4, 2)
    packed = torch.arange(5 * 4 * 2, device=device).reshape(5, 4, 2)
    # ids: (2, 4)
    ids = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 2]], device=device)
    actual_bin = local_misc.binpack(packed, ids, -7)
    expected_bin = official.binpack(packed, ids, -7)
    _assert_tensor_equal(actual_bin, expected_bin)
    _assert_tensor_equal(
        local_misc.unbinpack(actual_bin, ids, -7),
        official.unbinpack(expected_bin, ids, -7),
    )


def test_misc_python_contracts_match_pinned_biohub() -> None:
    official = _official_misc()
    indices = np.asarray([True, False, True, False])
    for value in ("ABCD", [1, 2, 3, 4], (1, 2, 3, 4)):
        assert local_misc.slice_python_object_as_numpy(value, indices) == (
            official.slice_python_object_as_numpy(value, indices)
        )
    assert local_misc.merge_ranges(
        [range(8, 10), range(1, 3), range(4, 8)], merge_gap_max=1
    ) == official.merge_ranges([range(8, 10), range(1, 3), range(4, 8)], merge_gap_max=1)
    annotations = [
        local_types.FunctionAnnotation(label="helix", start=2, end=5),
        local_types.FunctionAnnotation(label="helix", start=7, end=8),
        local_types.FunctionAnnotation(label="site", start=4, end=4),
    ]
    assert local_misc.merge_annotations(annotations, 1) == official.merge_annotations(
        annotations, 1
    )
    sequence = list(f"AC{CHAIN_BREAK_STR}DE")
    _assert_array_equal(
        local_misc.get_chainbreak_boundaries_from_sequence(sequence),
        official.get_chainbreak_boundaries_from_sequence(sequence),
    )
    for value in (None, [1, float("inf"), -float("inf")]):
        assert local_misc.replace_inf(value) == official.replace_inf(value)
    assert local_misc.join_lists([[1, 2], [3], [4]], [0]) == official.join_lists(
        [[1, 2], [3], [4]], [0]
    )
    _assert_array_equal(
        local_misc.concat_objects([np.asarray([1, 2]), np.asarray([3])], separator=0),
        official.concat_objects([np.asarray([1, 2]), np.asarray([3])], separator=0),
    )


def test_misc_conversion_serialization_and_concat_match_pinned_biohub() -> None:
    official = _official_misc()
    tensor_list = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    _assert_tensor_equal(
        local_misc.maybe_tensor(tensor_list),
        official.maybe_tensor(tensor_list),
    )
    nested = [[1.0, None], [float("inf"), -2.0]]
    _assert_tensor_equal(
        local_misc.maybe_tensor(nested, convert_none_to_nan=True),
        official.maybe_tensor(nested, convert_none_to_nan=True),
    )
    # values: (2, 2)
    values = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
    assert local_misc.maybe_list(values, convert_nan_to_none=True) == (
        official.maybe_list(values, convert_nan_to_none=True)
    )
    assert local_misc.concat_objects(["AB", "CD"], "|") == (
        official.concat_objects(["AB", "CD"], "|")
    )
    assert local_misc.concat_objects([[1, 2], [3]], 0) == official.concat_objects([[1, 2], [3]], 0)
    _assert_tensor_equal(
        local_misc.concat_objects([torch.tensor([1, 2]), torch.tensor([3])], separator=0),
        official.concat_objects([torch.tensor([1, 2]), torch.tensor([3])], separator=0),
    )
    assert list(local_misc.iterate_with_intermediate([1, 2, 3], 0)) == list(
        official.iterate_with_intermediate([1, 2, 3], 0)
    )

    buffer = io.BytesIO()
    torch.save({"X": torch.arange(6).reshape(2, 3)}, buffer)
    compressed = zstandard.ZstdCompressor().compress(buffer.getvalue())
    actual = local_misc.deserialize_tensors(compressed)
    expected = official.deserialize_tensors(compressed)
    assert actual.keys() == expected.keys()
    _assert_tensor_equal(actual["X"], expected["X"])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_affine_rotation_contracts_match_pinned_biohub(device: str) -> None:
    assert torch.cuda.is_available(), "the affine parity suite requires CUDA"
    official = _official_affine()
    generator = torch.Generator(device=device).manual_seed(2026)
    # quaternions: (2, 5, 4)
    quaternions = torch.randn((2, 5, 4), generator=generator, device=device)
    # points: (2, 5, 3)
    points = torch.randn((2, 5, 3), generator=generator, device=device)

    actual_quat = local_affine.RotationQuat(quaternions, normalized=True)
    expected_quat = official.RotationQuat(quaternions, normalized=True)
    _assert_tensor_equal(actual_quat.tensor, expected_quat.tensor)
    _assert_tensor_equal(actual_quat.as_matrix().tensor, expected_quat.as_matrix().tensor)
    _assert_tensor_equal(actual_quat.apply(points), expected_quat.apply(points))
    _assert_tensor_equal(actual_quat.invert().tensor, expected_quat.invert().tensor)
    _assert_tensor_equal(
        actual_quat.compose(actual_quat).tensor,
        expected_quat.compose(expected_quat).tensor,
    )

    actual_matrix = actual_quat.as_matrix()
    expected_matrix = expected_quat.as_matrix()
    _assert_tensor_equal(actual_matrix.as_quat().tensor, expected_matrix.as_quat().tensor)
    _assert_tensor_equal(actual_matrix.apply(points), expected_matrix.apply(points))
    _assert_tensor_equal(actual_matrix.invert().tensor, expected_matrix.invert().tensor)
    _assert_tensor_equal(
        actual_matrix.compose(actual_matrix).tensor,
        expected_matrix.compose(expected_matrix).tensor,
    )

    # batched_points: (2, 7, 3)
    batched_points = torch.randn((2, 7, 3), generator=generator, device=device)
    actual_single = local_affine.RotationMatrix.identity((2, 1), device=device)
    expected_single = official.RotationMatrix.identity((2, 1), device=device)
    _assert_tensor_equal(
        actual_single.apply(batched_points),
        expected_single.apply(batched_points),
    )


def test_affine_frames_and_coordinate_fallback_match_pinned_biohub() -> None:
    assert torch.cuda.is_available(), "the affine parity suite requires CUDA"
    official = _official_affine()
    generator = torch.Generator(device="cuda").manual_seed(912)
    # translation: (2, 4, 3)
    translation = torch.randn((2, 4, 3), generator=generator, device="cuda")
    # quaternion: (2, 4, 4)
    quaternion = torch.randn((2, 4, 4), generator=generator, device="cuda")
    # encoded: (...)
    encoded = torch.cat((quaternion, translation), dim=-1)
    actual = local_affine.Affine3D.from_tensor(encoded)
    expected = official.Affine3D.from_tensor(encoded)
    _assert_tensor_equal(actual.tensor, expected.tensor)
    _assert_tensor_equal(actual.invert().tensor, expected.invert().tensor)
    _assert_tensor_equal(
        actual.compose(actual).tensor,
        expected.compose(expected).tensor,
    )
    # mask: (2, 4)
    mask = torch.tensor(
        [[1, 0, 1, 0], [0, 1, 1, 0]],
        dtype=torch.bool,
        device="cuda",
    )
    _assert_tensor_equal(actual.mask(mask).tensor, expected.mask(mask).tensor)
    _assert_tensor_equal(
        actual.mask(mask, with_zero=True).tensor,
        expected.mask(mask, with_zero=True).tensor,
    )

    # coords: (2, 6, 3, 3)
    coords = torch.randn((2, 6, 3, 3), generator=generator, device="cuda")
    coords[0, 2] = torch.nan
    coords[1, 4] = 2e6
    actual_frame, actual_mask = local_affine.build_affine3d_from_coordinates(coords)
    expected_frame, expected_mask = official.build_affine3d_from_coordinates(coords)
    _assert_tensor_equal(actual_frame.tensor, expected_frame.tensor)
    _assert_tensor_equal(actual_mask, expected_mask)


def test_affine_encodings_and_collection_operations_match_pinned_biohub() -> None:
    assert torch.cuda.is_available(), "the affine parity suite requires CUDA"
    official = _official_affine()
    generator = torch.Generator(device="cuda").manual_seed(661)
    # translations: (2, 3)
    translations = torch.randn((2, 3), generator=generator, device="cuda")
    # compact_quat: (2, 3)
    compact_quat = torch.randn((2, 3), generator=generator, device="cuda")
    # full_quat: (2, 4)
    full_quat = torch.randn((2, 4), generator=generator, device="cuda")
    # matrix: (...)
    matrix = torch.eye(3, device="cuda").expand(2, -1, -1)
    # matrix4: (...)
    matrix4 = torch.eye(4, device="cuda").expand(2, -1, -1).clone()
    matrix4[..., :3, 3] = translations
    encodings = (
        matrix4,
        torch.cat((compact_quat, translations), dim=-1),
        torch.cat((full_quat, translations), dim=-1),
        torch.cat((matrix.flatten(-2), translations), dim=-1),
    )
    for encoded in encodings:
        actual = local_affine.Affine3D.from_tensor(encoded)
        expected = official.Affine3D.from_tensor(encoded)
        _assert_tensor_equal(actual.tensor, expected.tensor)

    # x_axis: (2, 3)
    x_axis = torch.randn((2, 3), generator=generator, device="cuda")
    # origin: (2, 3)
    origin = torch.randn((2, 3), generator=generator, device="cuda")
    # plane: (2, 3)
    plane = torch.randn((2, 3), generator=generator, device="cuda")
    actual = local_affine.Affine3D.from_graham_schmidt(x_axis, origin, plane)
    expected = official.Affine3D.from_graham_schmidt(x_axis, origin, plane)
    _assert_tensor_equal(actual.tensor, expected.tensor)
    # points: (2, 3)
    points = torch.randn((2, 3), generator=generator, device="cuda")
    _assert_tensor_equal(actual.apply(points), expected.apply(points))
    _assert_tensor_equal(actual.scale(2.5).tensor, expected.scale(2.5).tensor)
    _assert_tensor_equal(
        actual.compose_rotation(actual.rot).tensor,
        expected.compose_rotation(expected.rot).tensor,
    )
    _assert_tensor_equal(
        local_affine.Affine3D.cat([actual, actual], dim=0).tensor,
        official.Affine3D.cat([expected, expected], dim=0).tensor,
    )
    _assert_tensor_equal(
        actual.tensor_apply(lambda component: component + 1).tensor,
        expected.tensor_apply(lambda component: component + 1).tensor,
    )

    torch.manual_seed(444)
    actual_random = local_affine.Affine3D.random((2, 3), device="cuda")
    torch.manual_seed(444)
    expected_random = official.Affine3D.random((2, 3), device="cuda")
    _assert_tensor_equal(actual_random.tensor, expected_random.tensor)


def _normalize_entries(msa: Any) -> list[tuple[str, str]]:
    return [(entry.header, entry.sequence) for entry in msa.entries]


def _assert_msa_equal(actual: Any, expected: Any) -> None:
    assert _normalize_entries(actual) == _normalize_entries(expected)
    if actual.deletions is None or expected.deletions is None:
        assert actual.deletions is expected.deletions
    else:
        _assert_array_equal(actual.deletions, expected.deletions)


def test_msa_a3m_state_and_selection_match_pinned_biohub() -> None:
    official = _official_msa()
    a3m = ">query\nACD-EF\n>hit one\nAqCD-EF\n>hit two\nACdD-EF\n"
    actual = local_msa.MSA.from_a3m(io.StringIO(a3m))
    expected = official.MSA.from_a3m(io.StringIO(a3m))
    _assert_msa_equal(actual, expected)
    _assert_array_equal(
        local_msa.a3m_deletion_counts("AqrC.D"),
        official.a3m_deletion_counts("AqrC.D"),
    )
    assert [local_msa.is_a3m_insertion(value) for value in "a.A-"] == [
        official.is_a3m_insertion(value) for value in "a.A-"
    ]
    assert actual.to_bytes() == expected.to_bytes()
    _assert_msa_equal(
        local_msa.MSA.from_bytes(actual.to_bytes()),
        official.MSA.from_bytes(expected.to_bytes()),
    )
    assert actual.to_sequence_bytes() == expected.to_sequence_bytes()
    _assert_msa_equal(actual.select_sequences([0, 2]), expected.select_sequences([0, 2]))
    _assert_msa_equal(actual.select_positions([0, 2, 4]), expected.select_positions([0, 2, 4]))
    _assert_msa_equal(actual[[1, 3, 5]], expected[[1, 3, 5]])
    _assert_msa_equal(actual.pad_to_depth(5), expected.pad_to_depth(5))
    assert actual.state_dict(json_serializable=True) == expected.state_dict(json_serializable=True)
    _assert_msa_equal(
        local_msa.MSA.from_state_dict(actual.state_dict()),
        official.MSA.from_state_dict(expected.state_dict()),
    )


def test_msa_composition_and_fast_representation_match_pinned_biohub() -> None:
    official = _official_msa()
    left_sequences = ["ACD", "A-D", "AC-"]
    right_sequences = ["EF", "E-", "-F"]
    # left_deletions: (3, 3)
    left_deletions = np.arange(9, dtype=np.float32).reshape(3, 3)
    # right_deletions: (3, 2)
    right_deletions = np.arange(6, dtype=np.float32).reshape(3, 2)
    actual_left = local_msa.MSA.from_state_dict(
        {"sequences": left_sequences, "deletions": left_deletions}
    )
    expected_left = official.MSA.from_state_dict(
        {"sequences": left_sequences, "deletions": left_deletions}
    )
    actual_right = local_msa.MSA.from_state_dict(
        {"sequences": right_sequences, "deletions": right_deletions}
    )
    expected_right = official.MSA.from_state_dict(
        {"sequences": right_sequences, "deletions": right_deletions}
    )
    _assert_msa_equal(
        local_msa.MSA.concat([actual_left, actual_right], join_token=""),
        official.MSA.concat([expected_left, expected_right], join_token=""),
    )
    _assert_msa_equal(
        local_msa.MSA.stack([actual_left, actual_left]),
        official.MSA.stack([expected_left, expected_left]),
    )
    _assert_array_equal(actual_left.seqid, expected_left.seqid)
    assert repr(actual_left) == repr(expected_left)
    assert local_msa.remove_insertions_from_sequence("AqCdeD") == (
        official.remove_insertions_from_sequence("AqCdeD")
    )
    np.random.seed(71)
    actual_random = actual_left.select_random_sequences(2)
    np.random.seed(71)
    expected_random = expected_left.select_random_sequences(2)
    _assert_msa_equal(actual_random, expected_random)

    actual_fast = actual_left.to_fast_msa()
    expected_fast = expected_left.to_fast_msa()
    _assert_array_equal(actual_fast.array, expected_fast.array)
    assert actual_fast.headers == expected_fast.headers
    _assert_array_equal(
        actual_fast.pad_to_depth(5).array,
        expected_fast.pad_to_depth(5).array,
    )
    actual_fast_concat = local_msa.FastMSA.concat([actual_fast, actual_fast])
    expected_fast_concat = official.FastMSA.concat([expected_fast, expected_fast])
    _assert_array_equal(actual_fast_concat.array, expected_fast_concat.array)
    assert actual_fast_concat.headers == expected_fast_concat.headers
    assert _normalize_entries(actual_fast.to_msa()) == _normalize_entries(expected_fast.to_msa())
    actual_fast_bytes = local_msa.FastMSA.from_bytes(actual_left.to_bytes())
    expected_fast_bytes = official.FastMSA.from_bytes(expected_left.to_bytes())
    _assert_array_equal(actual_fast_bytes.array, expected_fast_bytes.array)
    assert actual_fast_bytes.headers == expected_fast_bytes.headers
    actual_sequence_only = local_msa.FastMSA.from_sequence_bytes(actual_left.to_sequence_bytes())
    expected_sequence_only = official.FastMSA.from_sequence_bytes(expected_left.to_sequence_bytes())
    _assert_array_equal(actual_sequence_only.array, expected_sequence_only.array)

    short_actual = local_msa.MSA.from_sequences(["AC", "A-"])
    short_expected = official.MSA.from_sequences(["AC", "A-"])
    _assert_msa_equal(
        local_msa.MSA.concat([actual_left, short_actual], allow_depth_mismatch=True),
        official.MSA.concat([expected_left, short_expected], allow_depth_mismatch=True),
    )


class _FakeColumn:
    def __init__(self, values, mask=None):
        self.values = np.asarray(values)
        self.mask = mask

    def as_array(self, dtype):
        return self.values.astype(dtype)

    def as_item(self):
        return self.values.item()


def _category(**columns):
    return {name: _FakeColumn(values) for name, values in columns.items()}


def _structure() -> bs.AtomArray:
    atoms = bs.AtomArray(7)
    # coord: (7, 3)
    atoms.coord = np.arange(21, dtype=np.float32).reshape(7, 3)
    atoms.chain_id = np.asarray(["A", "A", "A", "B", "B", "L", "L"])
    atoms.res_id = np.asarray([10, 10, 11, 5, 6, 1, 1])
    atoms.res_name = np.asarray(["ALA", "ALA", "CYS", "GLY", "SER", "ATP", "ATP"])
    atoms.atom_name = np.asarray(["N", "CA", "N", "N", "N", "P", "O1"])
    atoms.element = np.asarray(["N", "C", "N", "N", "N", "P", "O"])
    atoms.hetero = np.asarray([False, False, False, False, False, True, True])
    return atoms


def _fake_block() -> dict[str, Any]:
    return {
        "pdbx_database_status": _category(recvd_initial_deposition_date="2025-06-04"),
        "refine": _category(ls_d_res_high="1.75"),
        "exptl": _category(method="electron microscopy"),
        "entity": _category(
            id=["1", "2", "3"],
            type=["polymer", "polymer", "non-polymer"],
        ),
        "entity_poly": _category(
            entity_id=["1", "2"],
            pdbx_seq_one_letter_code_can=["AC D", "GS"],
            pdbx_strand_id=["A", "B"],
        ),
        "struct_asym": _category(id=["A", "B", "L"], entity_id=["1", "2", "3"]),
        "pdbx_poly_seq_scheme": _category(
            asym_id=["A", "A", "A", "B", "B"],
            seq_id=["1", "2", "3", "1", "2"],
            auth_seq_num=["10", "10", "?", "5", "6"],
            pdb_ins_code=[".", "A", "?", ".", "."],
            hetflag=["N", "Y", "N", "N", "N"],
            pdb_strand_id=["A", "A", "A", "B", "B"],
        ),
        "pdbx_entity_nonpoly": _category(entity_id=["3"], comp_id=["ATP"]),
        "atom_site": _category(
            label_asym_id=["A", "A", "A", "B", "B", "L", "L"],
            label_entity_id=["1", "1", "1", "2", "2", "3", "3"],
            label_comp_id=["ALA", "ALA", "CYS", "GLY", "SER", "ATP", "ATP"],
        ),
    }


def _normalized_mapping(wrapper: Any):
    return {
        chain: {
            index: (residue.residue_number, residue.insertion_code, residue.hetflag)
            for index, residue in mapping.items()
        }
        for chain, mapping in wrapper.seqres_to_structure.items()
    }


def test_mmcif_metadata_sequence_and_ligand_contracts_match_pinned_biohub() -> None:
    official = _official_mmcif()
    block = _fake_block()
    raw = types.SimpleNamespace(block=block)
    actual = local_mmcif.MmcifWrapper("case")
    expected = official.MmcifWrapper("case")
    actual.raw = raw
    expected.raw = raw
    actual.structure = _structure()
    expected.structure = _structure()
    for wrapper in (actual, expected):
        wrapper._parse_header()
        wrapper._parse_entities()
        wrapper._parse_sequences()
    assert (
        actual.header.release_date,
        actual.header.resolution,
        actual.header.structure_method,
    ) == (
        expected.header.release_date,
        expected.header.resolution,
        expected.header.structure_method,
    )
    assert actual.entities == expected.entities
    assert actual.chain_to_seqres == expected.chain_to_seqres
    assert _normalized_mapping(actual) == _normalized_mapping(expected)
    actual_ligands = actual._parse_nonpoly_from_mmcif()
    expected_ligands = expected._parse_nonpoly_from_mmcif()
    assert actual_ligands.keys() == expected_ligands.keys()
    for key in actual_ligands:
        _assert_array_equal(actual_ligands[key].coord, expected_ligands[key].coord)
    actual_fallback = actual._parse_nonpoly_fallback()
    expected_fallback = expected._parse_nonpoly_fallback()
    assert actual_fallback.keys() == expected_fallback.keys()
    for key in actual_fallback:
        _assert_array_equal(actual_fallback[key].coord, expected_fallback[key].coord)


def test_mmcif_rounding_matches_pinned_biohub() -> None:
    official = _official_mmcif()
    columns = {
        "Cartn_x": _FakeColumn([1.23456, -2.34567]),
        "Cartn_y": _FakeColumn([3.45678, 4.56789]),
        "Cartn_z": _FakeColumn([5.67891, 6.78912]),
        "B_iso_or_equiv": _FakeColumn([92.345, 10.005]),
        "label_atom_id": _FakeColumn(["CA", "N"]),
    }
    actual_file = types.SimpleNamespace(block={"atom_site": dict(columns)})
    expected_file = types.SimpleNamespace(block={"atom_site": dict(columns)})
    local_mmcif.round_mmcif_columns(actual_file)
    official.round_mmcif_columns(expected_file)
    for name in ("Cartn_x", "Cartn_y", "Cartn_z", "B_iso_or_equiv"):
        _assert_array_equal(
            actual_file.block["atom_site"][name].as_array(str),
            expected_file.block["atom_site"][name].as_array(str),
        )
    actual_empty = types.SimpleNamespace(block={})
    expected_empty = types.SimpleNamespace(block={})
    local_mmcif.round_mmcif_columns(actual_empty)
    official.round_mmcif_columns(expected_empty)
    assert actual_empty.block == expected_empty.block
