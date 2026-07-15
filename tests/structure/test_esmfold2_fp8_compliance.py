"""Experimental H100 smoke coverage for explicit ESMFold2 FP8 reloads."""

from __future__ import annotations

import gc
import importlib

import pytest
import torch

from fastplms.registry import ModelSpec, get_model_registry

SEQUENCES = (
    "MSTNPKPQRKTKRNTNR",
    "ACDEFGHIK",
)
EXPECTED_FP8_PROJECTIONS = 80


def _esmfold2_specs() -> tuple[ModelSpec, ...]:
    return get_model_registry().by_family("esmfold2")


def _parameter(spec: ModelSpec) -> object:
    return pytest.param(spec, id=spec.id, marks=pytest.mark.large)


def _base_spec() -> ModelSpec:
    return get_model_registry()["esmfold2"]


def _load_current_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    module_name, class_name = spec.auto_map["AutoModel"].rsplit(".", maxsplit=1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    model = model_class.from_pretrained(
        spec.fast.repo_id,
        revision=spec.fast.revision,
        load_esmc=False,
        dtype=torch.bfloat16,
    )
    return model.eval().to(device=device)


def _assert_fp8_smoke(model: torch.nn.Module) -> None:
    status = model.esmc_precision_status
    assert status.requested == "fp8"
    assert status.resolved == "fp8"
    assert str(status.device).startswith("cuda")
    assert status.transformer_engine_version
    paths = model._esmc_fp8_module_paths
    assert len(paths) == len(set(paths)) == EXPECTED_FP8_PROJECTIONS
    assert all(path.endswith(".attn.out_proj") for path in paths)

    result = model.embed_dataset(
        list(SEQUENCES),
        batch_size=2,
        full_embeddings=True,
        dtype=torch.float32,
    )
    embeddings = tuple(record.load_tensor() for record in result)
    assert tuple(tensor.shape for tensor in embeddings) == ((17, 256), (9, 256))
    assert all(torch.isfinite(tensor).all() for tensor in embeddings)
    assert not any(key.startswith("_esmc.") for key in model.state_dict())


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("spec", [_parameter(spec) for spec in _esmfold2_specs()])
def test_explicit_fp8_smoke_on_each_esmfold2_variant(spec: ModelSpec) -> None:
    """Exercise the experimental FP8 opt-in once on every supported variant."""

    device = torch.device("cuda")
    model = _load_current_model(spec, device)
    try:
        model.reload_esmc(precision="fp8", device=device)
        _assert_fp8_smoke(model)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.large
def test_standard_esmfold2_rebuilds_fp8_from_bf16_three_times() -> None:
    """Rebuild transient FP8 modules from canonical BF16 state on every reload."""

    device = torch.device("cuda")
    model = _load_current_model(_base_spec(), device)
    try:
        for _cycle in range(3):
            model.reload_esmc(precision="bf16", device=device)
            assert model.esmc_precision_status.resolved == "bf16"
            assert model._esmc_fp8_module_paths == ()

            model.reload_esmc(precision="fp8", device=device)
            _assert_fp8_smoke(model)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.large
def test_auto_precision_selects_runtime_bf16_on_the_locked_h100() -> None:
    """Exercise the stable BF16 auto policy on a directly loaded CUDA model."""

    device = torch.device("cuda")
    model = _load_current_model(_base_spec(), device)
    try:
        model.reload_esmc(precision="auto", device=device)
        status = model.esmc_precision_status
        assert status.requested == "auto"
        assert status.resolved == "bf16"
        assert str(status.device).startswith("cuda")
        assert model._esmc_fp8_module_paths == ()
        result = model.embed_dataset(
            list(SEQUENCES),
            batch_size=2,
            full_embeddings=True,
            dtype=torch.float32,
        )
        assert all(torch.isfinite(record.load_tensor()).all() for record in result)
        assert not any(key.startswith("_esmc.") for key in model.state_dict())
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
