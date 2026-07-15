"""Manifest-derived benchmark matrix tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.run import (
    _benchmark_load_dtype,
    _resolve_bf16_execution,
    _uses_bf16_autocast,
    run_case,
)
from benchmarks.suite import (
    ESMFOLD2_DEDICATED_MODE,
    ESMFOLD2_REPRESENTATION_PROFILE,
    FIXED_SHAPES,
    SEQUENCE_FORWARD_PROFILE,
    STRUCTURE_DEDICATED_MODE,
    STRUCTURE_STARTUP_PROFILE,
    benchmark_auto_class,
    benchmark_cases,
    benchmark_model_key,
    exhaustive_benchmark_cases,
)
from fastplms.registry import get_model_registry


def _shape(case: SimpleNamespace) -> tuple[int, int, tuple[int, ...]]:
    return case.batch_size, case.sequence_length, tuple(case.lengths)


def test_full_matrix_covers_fixed_shapes_for_each_sequence_backend() -> None:
    cases = list(benchmark_cases(family=None, quick=False, local_files_only=True))
    registry = get_model_registry()
    representatives = [
        spec
        for spec in registry.values()
        if spec.is_deep_reference
        and spec.family.tokenizer_mode != "structure"
        and "benchmark" in spec.family.test_tiers
    ]
    assert {spec.family.id for spec in representatives} == {
        "ankh",
        "dplm",
        "dplm2",
        "e1",
        "esm2",
        "esm3",
        "esm_plusplus",
    }
    for spec in representatives:
        for backend in spec.family.attention:
            matching = [
                case
                for case in cases
                if case.model == spec.fast.repo_id
                and case.backend == backend
                and case.mode == "steady"
            ]
            assert {_shape(case) for case in matching} == set(FIXED_SHAPES)
            assert all(case.bf16_execution == spec.family.bf16_execution for case in matching)
            compile_cases = [
                case
                for case in cases
                if case.model == spec.fast.repo_id
                and case.backend == backend
                and case.mode == "compile"
            ]
            assert len(compile_cases) == 1
            assert _shape(compile_cases[0]) == (1, 512, ())
        startup = [
            case for case in cases if case.model == spec.fast.repo_id and case.mode == "startup"
        ]
        embedding = [
            case for case in cases if case.model == spec.fast.repo_id and case.mode == "embed"
        ]
        assert len(startup) == len(embedding) == 1
        assert startup[0].suite_profile == SEQUENCE_FORWARD_PROFILE
        assert embedding[0].suite_profile == SEQUENCE_FORWARD_PROFILE


def test_esmfold2_matrix_separates_projection_from_esmc_precision() -> None:
    cases = list(benchmark_cases(family="esmfold2", quick=False, local_files_only=True))
    expected = {spec.fast.repo_id for spec in get_model_registry().by_family("esmfold2")}
    assert {case.model for case in cases} == expected
    for model_id in expected:
        model_cases = [case for case in cases if case.model == model_id]
        projection = [case for case in model_cases if case.mode == "projection"]
        assert {case.precision for case in projection} == {"bf16"}
        assert {_shape(case) for case in projection} == set(FIXED_SHAPES)
        assert all(case.backend == "sdpa" for case in projection)

        esmc_projection = [case for case in model_cases if case.mode == "esmc_projection"]
        for precision in ("bf16", "fp8"):
            for backend in ("eager", "sdpa", "flex_attention"):
                matching = [
                    case
                    for case in esmc_projection
                    if case.precision == precision and case.backend == backend
                ]
                assert {_shape(case) for case in matching} == set(FIXED_SHAPES)
                assert all(case.claim_eligible for case in matching)

        full_embedding = [case for case in model_cases if case.mode == "esmfold2_embed"]
        assert {case.precision for case in full_embedding} == {"bf16", "fp8"}
        assert all(_shape(case) == (1, 512, ()) for case in full_embedding)
        assert all(not case.claim_eligible for case in full_embedding)
        assert all(
            case.suite_profile == ESMFOLD2_REPRESENTATION_PROFILE
            and case.dedicated_mode == ESMFOLD2_DEDICATED_MODE
            for case in model_cases
        )


def test_structure_models_use_dedicated_startup_records() -> None:
    for family in ("esmfold", "boltz2"):
        cases = list(benchmark_cases(family=family, quick=False, local_files_only=True))
        assert len(cases) == 1
        case = cases[0]
        assert case.mode == "startup"
        assert case.suite_profile == STRUCTURE_STARTUP_PROFILE
        assert case.dedicated_mode == STRUCTURE_DEDICATED_MODE
        assert not case.claim_eligible


def test_every_manifest_family_declares_a_benchmark_tier() -> None:
    registry = get_model_registry()
    assert all("benchmark" in family.test_tiers for family in registry.families.values())


def test_exhaustive_matrix_is_all_checkpoint_and_descriptive() -> None:
    cases = list(
        exhaustive_benchmark_cases(
            family=None,
            batch_sizes=(1, 2),
            sequence_lengths=(128, 256),
            local_files_only=True,
        )
    )
    registry = get_model_registry()
    expected = {
        spec.fast.repo_id for spec in registry.values() if "benchmark" in spec.family.test_tiers
    }
    assert {case.model for case in cases} == expected
    assert all(case.matrix_kind == "exhaustive" for case in cases)
    assert all(not case.claim_eligible for case in cases)

    for spec in registry.values():
        if spec.family.tokenizer_mode == "structure":
            continue
        model_cases = [case for case in cases if case.model == spec.fast.repo_id]
        expected_axes = {
            (backend, batch_size, sequence_length)
            for backend in spec.family.attention
            for batch_size in (1, 2)
            for sequence_length in (128, 256)
        }
        assert {
            (case.backend, case.batch_size, case.sequence_length) for case in model_cases
        } == expected_axes

    for family in ("esmfold", "boltz2"):
        model_id = registry[registry.families[family].representative].fast.repo_id
        matching = [case for case in cases if case.model == model_id]
        assert len(matching) == 1
        assert matching[0].mode == "startup"

    for spec in registry.by_family("esmfold2"):
        model_cases = [case for case in cases if case.model == spec.fast.repo_id]
        projections = [case for case in model_cases if case.mode == "projection"]
        assert {
            (case.batch_size, case.sequence_length, case.precision) for case in projections
        } == {
            (batch_size, sequence_length, "bf16")
            for batch_size in (1, 2)
            for sequence_length in (128, 256)
        }
        esmc_projections = [case for case in model_cases if case.mode == "esmc_projection"]
        assert {
            (
                case.backend,
                case.batch_size,
                case.sequence_length,
                case.precision,
            )
            for case in esmc_projections
        } == {
            (backend, batch_size, sequence_length, precision)
            for backend in spec.family.attention
            for batch_size in (1, 2)
            for sequence_length in (128, 256)
            for precision in ("bf16", "fp8")
        }


def test_projection_mode_rejects_a_misleading_fp8_label_before_gpu_setup() -> None:
    arguments = SimpleNamespace(mode="projection", precision="fp8")
    with pytest.raises(ValueError, match="esmc_projection"):
        run_case(arguments)


def test_quick_matrix_is_one_short_case() -> None:
    cases = list(benchmark_cases(family=None, quick=True, local_files_only=True))
    assert len(cases) == 1
    assert cases[0].sequence_length <= 128


def test_benchmark_load_class_is_manifest_advertised() -> None:
    registry = get_model_registry()
    for spec in registry.values():
        if "benchmark" not in spec.family.test_tiers:
            continue
        selected = benchmark_auto_class(spec)
        assert selected in spec.auto_map


def test_benchmark_parameter_dtype_follows_manifest_bf16_execution() -> None:
    import torch

    static = SimpleNamespace(
        precision="bf16",
        bf16_execution="static_parameters",
    )
    autocast = SimpleNamespace(
        precision="bf16",
        bf16_execution="fp32_parameters_autocast",
    )
    assert _benchmark_load_dtype(static, torch) == torch.bfloat16
    assert not _uses_bf16_autocast(static)
    assert _benchmark_load_dtype(autocast, torch) == torch.float32
    assert _uses_bf16_autocast(autocast)


def test_benchmark_derives_registered_bf16_execution_from_manifest() -> None:
    registry = get_model_registry()
    for model_id in (
        "esm2_8m",
        "dplm_150m",
        "dplm2_150m",
        "boltz2",
        "esmfold",
        "esmfold2",
    ):
        spec = registry[model_id]
        arguments = SimpleNamespace(model=spec.fast.repo_id, bf16_execution=None)
        assert _resolve_bf16_execution(arguments) == spec.family.bf16_execution


def test_benchmark_rejects_precision_override_that_conflicts_with_manifest() -> None:
    spec = get_model_registry()["dplm_150m"]
    arguments = SimpleNamespace(
        model=spec.fast.repo_id,
        bf16_execution="static_parameters",
    )
    with pytest.raises(ValueError, match="conflicts with the manifest policy"):
        _resolve_bf16_execution(arguments)


def test_architecture_specific_benchmark_heads() -> None:
    registry = get_model_registry()
    assert benchmark_auto_class(registry["ankh_base"]) == "AutoModel"
    assert benchmark_auto_class(registry["esm3_small"]) == "AutoModel"
    assert benchmark_auto_class(registry["e1_150m"]) == "AutoModelForMaskedLM"
    assert benchmark_auto_class(registry["esmfold"]) == "AutoModel"
    assert benchmark_auto_class(registry["boltz2"]) == "AutoModel"
    assert benchmark_auto_class(registry["esmfold2"]) == "AutoModel"


def test_model_cache_key_reuses_backends_and_shapes_but_not_precision() -> None:
    cases = list(benchmark_cases(family="esm2", quick=False, local_files_only=True))
    assert len({benchmark_model_key(case) for case in cases}) == 1

    structure_cases = list(benchmark_cases(family="esmfold2", quick=False, local_files_only=True))
    by_model = {
        model_id: {benchmark_model_key(case) for case in structure_cases if case.model == model_id}
        for model_id in {case.model for case in structure_cases}
    }
    assert all(len(keys) == 2 for keys in by_model.values())
