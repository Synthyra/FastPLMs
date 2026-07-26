"""Execution contracts for precompiled Hugging Face FlashAttention kernels."""

from __future__ import annotations

import importlib.util
import pytest
import torch
from collections.abc import Callable
from pathlib import Path
from typing import Any
from torch.nn import functional as F

from fastplms.attention import _core
from fastplms.models.dplm.modeling_dplm import DPLMConfig, DPLMModel
from fastplms.models.esm2.modeling_fastesm import (
    FastEsmConfig,
    FastEsmForMaskedLM,
    FastEsmModel,
)
from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
    ESMplusplusConfig,
    ESMplusplusForMaskedLM,
    ESMplusplusModel,
)
from fastplms.registry import get_model_registry
from tools.debug.probe_flash_attention_forward import _model_results, _shared_results
from tools.debug.probe_flash_checkpoint_forward import _run_checkpoint


_FLASH_BACKENDS = ("flash_attention_2", "flash_attention_3")
_ESMC_FLASH_BACKENDS = _FLASH_BACKENDS


def _tiny_model_spec(family_id: str) -> tuple[type[torch.nn.Module], Any]:
    if family_id == "esm2":
        return FastEsmModel, FastEsmConfig(
            vocab_size=33,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=64,
            pad_token_id=1,
            mask_token_id=32,
            position_embedding_type="rotary",
            token_dropout=False,
            attn_backend="sdpa",
        )
    if family_id == "esm_plusplus":
        return ESMplusplusModel, ESMplusplusConfig(
            vocab_size=33,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            dropout=0.0,
            pad_token_id=1,
            attn_backend="sdpa",
        )
    raise AssertionError(f"Unexpected FlashAttention family: {family_id}")


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # actual: (...), expected: (...)
    relative_l2 = torch.linalg.vector_norm(
        actual.float() - expected.float()
    ) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    # minimum_cosine: ()
    minimum_cosine = F.cosine_similarity(actual.float(), expected.float(), dim=-1).min()
    assert relative_l2.item() <= 1e-2
    assert minimum_cosine.item() >= 0.999


def test_manifest_flash_inventory_is_exact() -> None:
    registry = get_model_registry()
    advertised = {
        family.id: tuple(name for name in family.attention if name in _FLASH_BACKENDS)
        for family in registry.families.values()
        if set(family.attention).intersection(_FLASH_BACKENDS)
    }
    assert advertised == {
        "esm2": _FLASH_BACKENDS,
        "esm_plusplus": _ESMC_FLASH_BACKENDS,
        "dplm": ("flash_attention_3",),
    }
    assert importlib.util.find_spec("flash_attn") is None


@pytest.mark.gpu
@pytest.mark.parametrize("family_id", ("esm2", "esm_plusplus"))
def test_explicit_flash_from_pretrained_uses_only_pinned_kernels(
    family_id: str,
    tmp_path: Path,
) -> None:
    assert torch.cuda.is_available(), "FlashAttention integration requires CUDA."
    model_class, config = _tiny_model_spec(family_id)
    model_path = tmp_path / family_id
    torch.manual_seed(29)
    model_class(config).save_pretrained(model_path)

    # input_ids: (2, 17)
    input_ids = torch.tensor(
        (
            (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 1, 1, 1, 1),
            (0, 20, 19, 18, 17, 16, 15, 14, 2, 1, 1, 1, 1, 1, 1, 1, 1),
        ),
        device="cuda",
    )
    # attention_mask: (b, l)
    attention_mask = input_ids.ne(1)
    # reference: (...)
    reference = (
        model_class.from_pretrained(
            model_path,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        )
        .eval()
        .to("cuda")
    )
    with torch.inference_mode():
        expected = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    advertised = tuple(
        name
        for name in get_model_registry().families[family_id].attention
        if name in _FLASH_BACKENDS
    )
    expected_backends = _FLASH_BACKENDS if family_id == "esm2" else _ESMC_FLASH_BACKENDS
    assert advertised == expected_backends
    for backend in advertised:
        _core._FLASH_KERNELS.pop(backend, None)
        model = (
            model_class.from_pretrained(
                model_path,
                attn_implementation=backend,
                dtype=torch.bfloat16,
            )
            .eval()
            .to("cuda")
        )
        assert model.config._attn_implementation == backend
        assert backend not in _core._FLASH_KERNELS
        with torch.inference_mode():
            actual = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
        assert backend in _core._FLASH_KERNELS
        assert torch.isfinite(actual).all()
        _assert_close(actual[attention_mask], expected[attention_mask])


@pytest.mark.gpu
def test_precompiled_flash_kernels_match_sdpa_for_dense_and_mixed_padding() -> None:
    assert torch.cuda.is_available(), "FlashAttention integration requires CUDA."
    results = _shared_results()
    for backend in _FLASH_BACKENDS:
        assert results[backend]["dense"]["relative_l2"] <= 1e-2
        assert results[backend]["dense"]["minimum_cosine"] >= 0.999
        assert results[backend]["mixed_padding"]["relative_l2"] <= 1e-2
        assert results[backend]["mixed_padding"]["minimum_cosine"] >= 0.999
        assert results[backend]["padding_is_zero"] is True


@pytest.mark.gpu
@pytest.mark.parametrize("mixed_padding", (False, True), ids=("dense", "mixed-padding"))
def test_precompiled_flash_attention_2_dense_and_varlen_backward(
    mixed_padding: bool,
) -> None:
    """The pinned FA2 autograd wrappers propagate gradients through Q, K, and V."""

    assert torch.cuda.is_available(), "FlashAttention integration requires CUDA."
    torch.manual_seed(31)
    # query: (2, 17, 4, 16)
    query = torch.randn(
        (2, 17, 4, 16),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    attention_mask = None
    if mixed_padding:
        # attention_mask: (2, 17)
        attention_mask = torch.tensor(
            (
                (1,) * 13 + (0,) * 4,
                (1,) * 9 + (0,) * 8,
            ),
            device="cuda",
            dtype=torch.bool,
        )

    output = _core.kernels_flash_attention_func(
        query,
        key,
        value,
        attention_mask_2d=attention_mask,
        implementation="flash_attention_2",
    )
    selected = output if attention_mask is None else output[attention_mask]
    selected.float().square().mean().backward()

    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.gpu
def test_flash_attention_2_mixed_padding_lora_step_and_reload(
    tmp_path: Path,
) -> None:
    """A real pinned FA2 kernel preserves a complete PEFT training graph."""

    peft = pytest.importorskip("peft")
    assert torch.cuda.is_available(), "FlashAttention PEFT integration requires CUDA."

    def make_base() -> FastEsmForMaskedLM:
        config = FastEsmConfig(
            vocab_size=33,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=128,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=64,
            pad_token_id=1,
            mask_token_id=32,
            position_embedding_type="rotary",
            token_dropout=False,
            attn_backend="flash_attention_2",
        )
        return FastEsmForMaskedLM(config)

    torch.manual_seed(41)
    base = make_base()
    base_state = {
        name: tensor.detach().clone()
        for name, tensor in base.state_dict().items()
    }
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            task_type=peft.TaskType.TOKEN_CLS,
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["query", "value"],
        ),
    ).to("cuda").train()
    model.base_model.model.set_attn_implementation("flash_attention_2")

    # input_ids: (2, 17)
    input_ids = torch.tensor(
        (
            (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 1, 1, 1, 1),
            (0, 20, 19, 18, 17, 16, 15, 14, 2, 1, 1, 1, 1, 1, 1, 1, 1),
        ),
        device="cuda",
    )
    # attention_mask: (b, l)
    attention_mask = input_ids.ne(1)
    # labels: (b, l)
    labels = input_ids.masked_fill(~attention_mask, -100)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-2,
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
    assert loss.is_cuda
    assert torch.isfinite(loss)
    loss.backward()

    trainable_gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_gradients
    assert all(gradient is not None for gradient in trainable_gradients.values())
    assert all(
        torch.isfinite(gradient).all()
        for gradient in trainable_gradients.values()
        if gradient is not None
    )
    optimizer.step()

    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    }
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert changed
    assert changed <= trainable
    assert any("lora_" in name for name in changed)
    assert all(
        torch.equal(parameter.detach(), before[name])
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )

    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        expected = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
    model.save_pretrained(tmp_path)

    restored_base = make_base()
    restored_base.load_state_dict(base_state)
    restored = peft.PeftModel.from_pretrained(restored_base, tmp_path).to("cuda").eval()
    restored.base_model.model.set_attn_implementation("flash_attention_2")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        observed = restored(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", _FLASH_BACKENDS)
def test_precompiled_flash_accepts_fp32_storage_only_under_cuda_bf16_autocast(
    backend: str,
) -> None:
    assert torch.cuda.is_available(), "FlashAttention integration requires CUDA."
    # X: (2, 17, 4, 16)
    X = torch.randn((2, 17, 4, 16), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match=r"bfloat16.*received float32"):
        _core.kernels_flash_attention_func(X, X, X, implementation=backend)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        Y = _core.kernels_flash_attention_func(X, X, X, implementation=backend)

    assert Y.dtype == torch.bfloat16
    assert torch.isfinite(Y).all()


@pytest.mark.gpu
def test_advertised_small_models_run_finite_flash_forwards() -> None:
    assert torch.cuda.is_available(), "FlashAttention integration requires CUDA."
    results = _model_results()
    assert set(results) == {"esm2", "esm_plusplus"}
    for family_id, family_results in results.items():
        expected_backends = _FLASH_BACKENDS if family_id == "esm2" else _ESMC_FLASH_BACKENDS
        assert set(family_results) == set(expected_backends)
        for backend in expected_backends:
            metrics = family_results[backend]
            assert metrics["finite"] is True
            assert metrics["vs_sdpa"]["relative_l2"] <= 1e-2
            assert metrics["vs_sdpa"]["minimum_cosine"] >= 0.999


@pytest.mark.gpu
@pytest.mark.parametrize("mixed_padding", (False, True), ids=("dense", "mixed-padding"))
def test_dplm_tiny_flash_attention_3_forward_parity_and_backward(
    mixed_padding: bool,
) -> None:
    """DPLM's advertised FA3 path executes under its official BF16 policy."""

    assert torch.cuda.is_available(), "DPLM FlashAttention integration requires CUDA."
    config = DPLMConfig(
        vocab_size=33,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=64,
        pad_token_id=1,
        mask_token_id=32,
        position_embedding_type="rotary",
        token_dropout=False,
        attn_backend="sdpa",
    )
    torch.manual_seed(37)
    model = DPLMModel(config).to(device="cuda", dtype=torch.float32)
    # input_ids: (2, 17)
    input_ids = torch.tensor(
        (
            (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2, 1, 1, 1, 1),
            (0, 20, 19, 18, 17, 16, 15, 14, 2, 1, 1, 1, 1, 1, 1, 1, 1),
        ),
        device="cuda",
    )
    attention_mask = input_ids.ne(1) if mixed_padding else torch.ones_like(input_ids).bool()

    model.eval()
    outputs: dict[str, torch.Tensor] = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for backend in ("sdpa", "flash_attention_3"):
            model.set_attn_implementation(backend)
            outputs[backend] = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
    assert torch.isfinite(outputs["flash_attention_3"]).all()
    _assert_close(
        outputs["flash_attention_3"][attention_mask],
        outputs["sdpa"][attention_mask],
    )

    # Use a fresh training instance so rotary factors created under
    # `inference_mode` cannot leak into the autograd contract.
    training_model = DPLMModel(config).to(device="cuda", dtype=torch.float32).train()
    training_model.set_attn_implementation("flash_attention_3")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = training_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        # loss: ()
        loss = output[attention_mask].float().square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in training_model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    ("model_id", "model_class", "backends"),
    (
        ("esm2_8m", FastEsmForMaskedLM, _FLASH_BACKENDS),
        ("esmc_small", ESMplusplusForMaskedLM, _ESMC_FLASH_BACKENDS),
        ("dplm_150m", DPLMModel, ("flash_attention_3",)),
    ),
)
def test_manifest_checkpoint_flash_mixed_padding_parity(
    model_id: str,
    model_class: type[torch.nn.Module],
    backends: tuple[str, ...],
    record_property: Callable[[str, object], None],
) -> None:
    assert torch.cuda.is_available(), "Checkpoint FlashAttention parity requires CUDA."
    spec = get_model_registry()[model_id]
    result = _run_checkpoint(model_id, model_class)
    assert result["checkpoint"] == spec.fast.repo_id
    assert result["revision"] == spec.fast.revision
    assert set(result["backends"]) == set(backends)
    for backend in backends:
        metrics = result["backends"][backend]
        record_property(f"{backend}_relative_l2", metrics["relative_l2"])
        record_property(f"{backend}_minimum_cosine", metrics["minimum_cosine"])
        assert metrics["finite"] is True
        assert metrics["relative_l2"] <= 1e-2
        assert metrics["minimum_cosine"] >= 0.999
