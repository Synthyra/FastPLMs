"""Exact pinned parity for ESMFold2 residue constants and configuration."""

from __future__ import annotations

import importlib.util
import re
import struct
import sys
import tokenize
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fastplms.models.esmfold2 import configuration_esmfold2 as local_config
from fastplms.models.esmfold2 import esmfold2_residue_constants as local_residues

pytestmark = [pytest.mark.compliance, pytest.mark.gpu, pytest.mark.structure]

ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = ROOT / "src/fastplms/models/esmfold2"
OFFICIAL_RESIDUES = ROOT / "vendor/upstream/biohub-esm/esm/utils/residue_constants.py"
OFFICIAL_CONFIG = (
    ROOT
    / "vendor/upstream/biohub-transformers/src/transformers/models/esmfold2"
    / "configuration_esmfold2.py"
)
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
    aliases: dict[str, types.ModuleType] | None = None,
) -> types.ModuleType:
    assert path.is_file(), f"pinned source is missing: {path}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    with _temporary_modules({**(aliases or {}), module_name: module}):
        specification.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def official_residues() -> types.ModuleType:
    return _load_source("_fastplms_pinned_biohub_residue_constants", OFFICIAL_RESIDUES)


@pytest.fixture(scope="module")
def official_config() -> types.ModuleType:
    import transformers.configuration_utils as configuration_utils

    root_name = "_fastplms_pinned_biohub_transformers"
    aliases = {
        root_name: _package(root_name),
        f"{root_name}.models": _package(f"{root_name}.models"),
        f"{root_name}.models.esmfold2": _package(f"{root_name}.models.esmfold2"),
        f"{root_name}.configuration_utils": configuration_utils,
    }
    return _load_source(
        f"{root_name}.models.esmfold2.configuration_esmfold2",
        OFFICIAL_CONFIG,
        aliases,
    )


def _assert_exact(actual: Any, expected: Any) -> None:
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
        return
    if isinstance(expected, float):
        assert isinstance(actual, float)
        assert struct.pack("!d", actual) == struct.pack("!d", expected)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert tuple(actual) == tuple(expected)
        for key in expected:
            _assert_exact(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_exact(actual_item, expected_item)
        return
    assert type(actual) is type(expected)
    assert actual == expected


def _public_data(module: types.ModuleType) -> dict[str, Any]:
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_")
        and name != "annotations"
        and not isinstance(value, types.ModuleType)
        and not callable(value)
    }


def test_all_public_residue_tables_match_exactly(
    official_residues: types.ModuleType,
) -> None:
    expected = _public_data(official_residues)
    actual = _public_data(local_residues)
    assert actual.keys() == expected.keys(), (
        f"local-only={sorted(actual.keys() - expected.keys())}, "
        f"official-only={sorted(expected.keys() - actual.keys())}"
    )
    for name in expected:
        _assert_exact(actual[name], expected[name])
    assert local_residues.Bond._fields == official_residues.Bond._fields
    assert local_residues.BondAngle._fields == official_residues.BondAngle._fields


@pytest.mark.parametrize("atom_index", range(4))
def test_chi_selectors_match_exactly(official_residues: types.ModuleType, atom_index: int) -> None:
    _assert_exact(
        local_residues.chi_angle_atom(atom_index),
        official_residues.chi_angle_atom(atom_index),
    )


@pytest.mark.parametrize(
    ("sequence", "mapping", "map_unknown_to_x"),
    [
        ("", {"A": 0, "X": 1}, False),
        ("AXA", {"A": 0, "X": 1}, False),
        ("AZ", {"A": 0, "X": 1}, True),
        ("ACDEFGHIKLMNPQRSTVWY", local_residues.restype_order_with_x, False),
    ],
)
def test_one_hot_encoding_matches_exactly(
    official_residues: types.ModuleType,
    sequence: str,
    mapping: dict[str, int],
    map_unknown_to_x: bool,
) -> None:
    actual = local_residues.sequence_to_onehot(sequence, mapping, map_unknown_to_x)
    expected = official_residues.sequence_to_onehot(sequence, mapping, map_unknown_to_x)
    _assert_exact(actual, expected)


@pytest.mark.parametrize(
    ("sequence", "mapping", "map_unknown_to_x"),
    [
        ("a", {"A": 0, "X": 1}, True),
        ("?", {"A": 0, "X": 1}, True),
        ("B", {"A": 0, "X": 2}, True),
        ("B", {"A": 0}, False),
    ],
)
def test_one_hot_failures_match_pinned_behavior(
    official_residues: types.ModuleType,
    sequence: str,
    mapping: dict[str, int],
    map_unknown_to_x: bool,
) -> None:
    with pytest.raises(Exception) as local_error:
        local_residues.sequence_to_onehot(sequence, mapping, map_unknown_to_x)
    with pytest.raises(Exception) as official_error:
        official_residues.sequence_to_onehot(sequence, mapping, map_unknown_to_x)
    assert type(local_error.value) is type(official_error.value)
    assert str(local_error.value) == str(official_error.value)


def test_rigid_transform_and_mapping_builders_match_exactly(
    official_residues: types.ModuleType,
) -> None:
    ex = np.asarray((1.25, -0.5, 0.75), dtype=np.float64)
    ey = np.asarray((-0.25, 1.5, 0.125), dtype=np.float64)
    translation = np.asarray((7.0, -3.0, 2.0), dtype=np.float64)
    _assert_exact(
        local_residues._make_rigid_transformation_4x4(ex, ey, translation),
        official_residues._make_rigid_transformation_4x4(ex, ey, translation),
    )
    for function_name in (
        "_make_standard_atom_mask",
        "_make_restype_atom14_to_atom37",
        "_make_restype_atom37_to_atom14",
    ):
        _assert_exact(
            getattr(local_residues, function_name)(),
            getattr(official_residues, function_name)(),
        )
    indices = np.asarray((0, 4, 7, 20, 2), dtype=np.int64)
    assert local_residues.aatype_to_str_sequence(indices) == (
        official_residues.aatype_to_str_sequence(indices)
    )


def test_stereo_chemical_derivation_matches_exactly(
    monkeypatch: pytest.MonkeyPatch,
    official_residues: types.ModuleType,
) -> None:
    table = """bond residue length stddev
N-CA ALA 1.458 0.020
CA-C ALA 1.525 0.021
-

angle residue degrees stddev
N-CA-C ALA 111.2 2.1
-
"""

    class _Table:
        @staticmethod
        def read_text() -> str:
            return table

    monkeypatch.setattr(local_residues, "_STEREO_CHEMICAL_PROPS_PATH", _Table())
    monkeypatch.setattr(official_residues, "Path", lambda _path: _Table())
    local_residues.load_stereo_chemical_props.cache_clear()
    official_residues.load_stereo_chemical_props.cache_clear()
    actual = local_residues.load_stereo_chemical_props()
    expected = official_residues.load_stereo_chemical_props()
    for actual_group, expected_group in zip(actual, expected, strict=True):
        assert tuple(actual_group) == tuple(expected_group)
        for residue_name in expected_group:
            assert [tuple(record) for record in actual_group[residue_name]] == [
                tuple(record) for record in expected_group[residue_name]
            ]
            for actual_record, expected_record in zip(
                actual_group[residue_name], expected_group[residue_name], strict=True
            ):
                for actual_value, expected_value in zip(
                    actual_record, expected_record, strict=True
                ):
                    _assert_exact(actual_value, expected_value)
    local_residues.load_stereo_chemical_props.cache_clear()
    official_residues.load_stereo_chemical_props.cache_clear()


@pytest.mark.parametrize(
    ("overlap_tolerance", "bond_length_tolerance_factor"),
    [(1.5, 15.0), (0.75, 4.0)],
)
def test_atom14_distance_bounds_match_exactly(
    monkeypatch: pytest.MonkeyPatch,
    official_residues: types.ModuleType,
    overlap_tolerance: float,
    bond_length_tolerance_factor: float,
) -> None:
    def tables(module: types.ModuleType) -> tuple[dict, dict, dict]:
        bonds = {name: [] for name in module.resnames}
        virtual = {name: [] for name in module.resnames}
        angles = {name: [] for name in module.resnames}
        bonds["ALA"] = [module.Bond("N", "CA", 1.458, 0.02)]
        return bonds, virtual, angles

    monkeypatch.setattr(
        local_residues, "load_stereo_chemical_props", lambda: tables(local_residues)
    )
    monkeypatch.setattr(
        official_residues,
        "load_stereo_chemical_props",
        lambda: tables(official_residues),
    )
    actual = local_residues.make_atom14_dists_bounds(
        overlap_tolerance, bond_length_tolerance_factor
    )
    expected = official_residues.make_atom14_dists_bounds(
        overlap_tolerance, bond_length_tolerance_factor
    )
    assert actual.keys() == expected.keys()
    for name in expected:
        _assert_exact(actual[name], expected[name])


_NESTED_NAMES = (
    "AtomAttentionConfig",
    "DiffusionModuleConfig",
    "FoldingTrunkConfig",
    "InputsEmbedderConfig",
    "DiffusionStructureHeadConfig",
    "ConfidenceHeadConfig",
    "MSAEncoderConfig",
    "LMEncoderConfig",
    "ParcaeConfig",
)


def test_nested_configuration_defaults_match_exactly(
    official_config: types.ModuleType,
) -> None:
    for class_name in _NESTED_NAMES:
        actual = getattr(local_config, class_name)()
        expected = getattr(official_config, class_name)()
        assert asdict(actual) == asdict(expected)


def test_full_configuration_schema_matches_pinned_biohub(
    official_config: types.ModuleType,
) -> None:
    kwargs = {
        "type": "experimental",
        "d_single": 320,
        "d_pair": 192,
        "n_relative_residx_bins": 21,
        "n_relative_chain_bins": 3,
        "num_loops": 7,
        "num_diffusion_samples": 3,
        "disable_msa_features": True,
        "lm_dropout": 0.125,
        "force_lm_dropout_during_inference": True,
        "lm_d_model": 1280,
        "lm_num_layers": 40,
        "esmc_id": "Synthyra/ESMplusplus_6B",
        "inputs": {"d_inputs": 777, "atom_encoder": {"n_blocks": 5}},
        "folding_trunk": {"n_layers": 8, "n_heads": 4},
        "structure_head": {"diffusion_module": {"token_num_blocks": 3}},
        "confidence_head": {"folding_trunk": {"n_layers": 2}},
        "msa_encoder": {"enabled": True, "d_msa": 64},
        "parcae": {"max_steps": None},
        "lm_encoder": {"per_loop_lm_dropout": False},
        "msa_encoder_overwrite": False,
    }
    actual = local_config.ESMFold2Config(**kwargs)
    expected = official_config.ESMFold2Config(**kwargs)
    shared_scalars = (
        "type",
        "d_single",
        "d_pair",
        "n_relative_residx_bins",
        "n_relative_chain_bins",
        "num_loops",
        "num_diffusion_samples",
        "disable_msa_features",
        "lm_dropout",
        "force_lm_dropout_during_inference",
        "lm_d_model",
        "lm_num_layers",
        "esmc_id",
        "msa_encoder_overwrite",
    )
    assert {name: getattr(actual, name) for name in shared_scalars} == {
        name: getattr(expected, name) for name in shared_scalars
    }
    for name in (
        "inputs",
        "folding_trunk",
        "structure_head",
        "confidence_head",
        "msa_encoder",
        "parcae",
        "lm_encoder",
    ):
        assert asdict(getattr(actual, name)) == asdict(getattr(expected, name))


def test_fastplms_configuration_extensions_are_strict(tmp_path: Path) -> None:
    config = local_config.ESMFold2Config(
        type="release",
        esmc_id="biohub/ESMC-6B",
        attn_implementation={"": "flex"},
        esmc_precision="fp8",
    )
    assert config.esmc_id == "Synthyra/ESMplusplus_6B"
    assert config.esmc_attn_backend == "flex_attention"
    assert config.esmc_precision == "fp8"
    config.save_pretrained(tmp_path)
    restored = local_config.ESMFold2Config.from_pretrained(tmp_path)
    assert restored.to_dict() == config.to_dict()
    with pytest.raises(ValueError, match="Unsupported ESMFold2 attention"):
        local_config.ESMFold2Config(type="release", attn_implementation="unknown")
    with pytest.raises(ValueError, match="esmc_precision"):
        local_config.ESMFold2Config(type="release", esmc_precision="int8")


def test_modules_have_standalone_artifact_import_closure() -> None:
    residues = _load_source(
        "_fastplms_artifact_residue_constants",
        LOCAL_ROOT / "esmfold2_residue_constants.py",
    )
    configuration = _load_source(
        "_fastplms_artifact_configuration_esmfold2",
        LOCAL_ROOT / "configuration_esmfold2.py",
    )
    assert residues.STANDARD_ATOM_MASK.shape == (21, 37)
    config = configuration.ESMFold2Config(type="release")
    assert config.model_type == "esmfold2"
    assert config.esmc_id == "Synthyra/ESMplusplus_6B"


def _meaningful_lines(text: str) -> list[str]:
    return [
        " ".join(line.strip().split())
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


@pytest.mark.parametrize(
    ("local_name", "official_path"),
    [
        ("esmfold2_residue_constants.py", OFFICIAL_RESIDUES),
        ("configuration_esmfold2.py", OFFICIAL_CONFIG),
    ],
)
def test_runtime_source_is_independently_organized(local_name: str, official_path: Path) -> None:
    local_path = LOCAL_ROOT / local_name
    similarity = SequenceMatcher(
        None,
        _meaningful_lines(local_path.read_text(encoding="utf-8")),
        _meaningful_lines(official_path.read_text(encoding="utf-8")),
        autojunk=False,
    ).ratio()
    assert similarity < 0.75, f"{local_name} has line similarity {similarity:.3f}"


def test_comments_and_docstrings_follow_shape_notation() -> None:
    square_shape = re.compile(r"\[\s*[A-Z](?:\s*,\s*[A-Z])+(?:\s*,[^]]*)?\]")
    upper_dimensions = re.compile(r"\(\s*[BLDNH](?:\s*,\s*[BLDNH])+(?:\s*,[^)]*)?\)")
    for path in (
        LOCAL_ROOT / "esmfold2_residue_constants.py",
        LOCAL_ROOT / "configuration_esmfold2.py",
    ):
        source = path.read_text(encoding="utf-8")
        prose = "\n".join(
            token.string
            for token in tokenize.generate_tokens(StringIO(source).readline)
            if token.type in {tokenize.COMMENT, tokenize.STRING}
        )
        assert square_shape.search(prose) is None
        assert upper_dimensions.search(prose) is None
