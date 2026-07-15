"""H100 compliance for ESMFold2 learned summaries under runtime FP8 reloads."""

from __future__ import annotations

import gc
import importlib
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
import torch.nn.functional as F

from fastplms.registry import ModelSpec, get_model_registry

SEQUENCES = (
    "MSTNPKPQRKTKRNTNR",
    "ACDEFGHIK",
)
FP8_RELATIVE_L2_TARGET = 0.04
FP8_RELATIVE_L2_HARD = 0.08
FP8_RESIDUE_COSINE_TARGET = 0.995
FP8_RESIDUE_COSINE_HARD = 0.99
FP8_POOLED_COSINE_TARGET = 0.999
FP8_POOLED_COSINE_HARD = 0.995


def _esmfold2_specs() -> tuple[ModelSpec, ...]:
    return get_model_registry().by_family("esmfold2")


def _parameter(spec: ModelSpec) -> object:
    return pytest.param(spec, id=spec.id, marks=pytest.mark.large)


def _base_spec() -> ModelSpec:
    return get_model_registry()["esmfold2"]


@pytest.mark.structure
@pytest.mark.compliance
@pytest.mark.gpu
def test_fp8_validation_stack_uses_the_cuda13_transformer_engine_core() -> None:
    assert version("transformer-engine") == "2.12.0"
    assert version("transformer-engine-cu13") == "2.12.0"
    assert version("transformer-engine-torch") == "2.12.0"
    with pytest.raises(PackageNotFoundError):
        version("transformer-engine-cu12")


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


def _full_embeddings(model: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    result = model.embed_dataset(
        list(SEQUENCES),
        batch_size=2,
        full_embeddings=True,
        dtype=torch.float32,
    )
    X = tuple(record.load_tensor() for record in result)
    assert tuple(tensor.shape for tensor in X) == ((17, 256), (9, 256))
    assert all(torch.isfinite(tensor).all() for tensor in X)
    return X


def _projection_metrics(
    bf16: tuple[torch.Tensor, ...],
    fp8: tuple[torch.Tensor, ...],
) -> tuple[float, float, float]:
    H_bf16 = torch.cat(bf16, dim=0).float()
    H_fp8 = torch.cat(fp8, dim=0).float()
    relative_l2 = (
        torch.linalg.vector_norm(H_fp8 - H_bf16)
        / torch.linalg.vector_norm(H_bf16).clamp_min(torch.finfo(torch.float32).tiny)
    ).item()
    residue_cosines = F.cosine_similarity(H_fp8, H_bf16, dim=-1)
    residue_cosine_p01 = torch.quantile(residue_cosines, 0.01).item()
    pooled_cosines = torch.stack(
        [
            F.cosine_similarity(X_fp8.mean(0), X_bf16.mean(0), dim=0)
            for X_bf16, X_fp8 in zip(bf16, fp8, strict=True)
        ]
    )
    return relative_l2, residue_cosine_p01, pooled_cosines.min().item()


def _assert_engineering_targets(
    spec: ModelSpec,
    cycle: int,
    metrics: tuple[float, float, float],
) -> None:
    relative_l2, residue_cosine_p01, pooled_cosine_min = metrics
    context = f"{spec.id} reload cycle {cycle}"
    assert relative_l2 <= FP8_RELATIVE_L2_HARD, (
        f"{context}: FP8 relative L2 {relative_l2:.6g} exceeds hard limit "
        f"{FP8_RELATIVE_L2_HARD:.6g}"
    )
    assert residue_cosine_p01 >= FP8_RESIDUE_COSINE_HARD, (
        f"{context}: first-percentile residue cosine {residue_cosine_p01:.6g} "
        f"is below hard limit {FP8_RESIDUE_COSINE_HARD:.6g}"
    )
    assert pooled_cosine_min >= FP8_POOLED_COSINE_HARD, (
        f"{context}: pooled cosine {pooled_cosine_min:.6g} is below hard limit "
        f"{FP8_POOLED_COSINE_HARD:.6g}"
    )
    assert relative_l2 <= FP8_RELATIVE_L2_TARGET, (
        f"{context}: FP8 relative L2 {relative_l2:.6g} misses engineering target "
        f"{FP8_RELATIVE_L2_TARGET:.6g}"
    )
    assert residue_cosine_p01 >= FP8_RESIDUE_COSINE_TARGET, (
        f"{context}: first-percentile residue cosine {residue_cosine_p01:.6g} "
        f"misses engineering target {FP8_RESIDUE_COSINE_TARGET:.6g}"
    )
    assert pooled_cosine_min >= FP8_POOLED_COSINE_TARGET, (
        f"{context}: pooled cosine {pooled_cosine_min:.6g} misses engineering target "
        f"{FP8_POOLED_COSINE_TARGET:.6g}"
    )


@pytest.mark.structure
@pytest.mark.compliance
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("spec", [_parameter(spec) for spec in _esmfold2_specs()])
def test_three_fresh_bf16_to_fp8_embedding_reload_cycles(spec: ModelSpec) -> None:
    """Requantize canonical BF16 weights and compare residue summaries three times."""

    device = torch.device("cuda")
    model = _load_current_model(spec, device)
    try:
        for cycle in range(1, 4):
            model.reload_esmc(precision="bf16", device=device)
            bf16 = _full_embeddings(model)
            assert model.esmc_precision_status.requested == "bf16"
            assert model.esmc_precision_status.resolved == "bf16"

            model.reload_esmc(precision="fp8", device=device)
            status = model.esmc_precision_status
            assert status.requested == "fp8"
            assert status.resolved == "fp8"
            assert str(status.device).startswith("cuda")
            assert status.transformer_engine_version
            fp8 = _full_embeddings(model)
            _assert_engineering_targets(spec, cycle, _projection_metrics(bf16, fp8))
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.compliance
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.large
def test_auto_precision_selects_runtime_fp8_on_the_locked_h100() -> None:
    """Exercise the public auto policy on a directly loaded CUDA model."""

    device = torch.device("cuda")
    model = _load_current_model(_base_spec(), device)
    try:
        model.reload_esmc(precision="auto", device=device)
        status = model.esmc_precision_status
        assert status.requested == "auto"
        assert status.resolved == "fp8"
        assert str(status.device).startswith("cuda")
        assert status.transformer_engine_version
        assert len(model._esmc_fp8_module_paths) == 80
        assert all(path.endswith(".attn.out_proj") for path in model._esmc_fp8_module_paths)
        _full_embeddings(model)
        assert not any(key.startswith("_esmc.") for key in model.state_dict())
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
