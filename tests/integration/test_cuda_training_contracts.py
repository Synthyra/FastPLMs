"""Small real-CUDA contracts that must not be satisfied by CPU-only probes."""

from __future__ import annotations

import pytest
import torch
from pathlib import Path

from fastplms.attention import _core
from fastplms.models.dplm.modeling_dplm import DPLMConfig, DPLMForMaskedLM
from fastplms.models.dplm2.modeling_dplm2 import DPLM2Config, DPLM2ForMaskedLM
from fastplms.models.esm2.modeling_fastesm import FastEsmConfig, FastEsmModel
from fastplms.models.esm3.modeling_esm3 import (
    FastESM3Config,
    FastESM3GenerationConfig,
    FastESM3Model,
)
from fastplms.models.esm_plusplus.modeling_esm_plusplus import TransformerStack
from tests.conftest import strict_fp32_matmul
from tests.integration.test_ttt import (
    DummyPretrainedTTTConfig,
    DummyPretrainedTTTModel,
)


pytestmark = pytest.mark.gpu


def _diffusion_config(vocab_size: int) -> dict[str, object]:
    return {
        "vocab_size": vocab_size,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "max_position_embeddings": 64,
        "pad_token_id": 1,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "mask_token_id": 32,
        "position_embedding_type": "rotary",
        "attn_backend": "sdpa",
    }


@pytest.mark.parametrize("family", ("dplm", "dplm2", "esm3"))
def test_seeded_generation_trace_executes_deterministically_on_cuda(
    family: str,
) -> None:
    assert torch.cuda.is_available(), "Generation CUDA contract requires CUDA."
    torch.manual_seed(53)

    if family == "dplm":
        model = DPLMForMaskedLM(
            DPLMConfig(**_diffusion_config(33)),
            dropout=0.0,
        ).train().cuda()
        # inputs: (1, 5)
        inputs = torch.tensor([[0, 6, 32, 8, 2]], device="cuda")

        def run() -> torch.Tensor:
            return model.generate(inputs, max_iter=2)

    elif family == "dplm2":
        model = DPLM2ForMaskedLM(
            DPLM2Config(**_diffusion_config(64)),
            dropout=0.0,
        ).train().cuda()
        # inputs: (1, 8)
        inputs = torch.tensor(
            [[33, 50, 50, 34, 0, 6, 6, 2]],
            device="cuda",
        )

        def run() -> torch.Tensor:
            generated = model.generate(inputs, max_iter=2)
            return generated["output_tokens"]

    else:
        model = FastESM3Model(
            FastESM3Config(
                hidden_size=64,
                num_attention_heads=4,
                num_vector_heads=8,
                num_hidden_layers=2,
            )
        ).train().cuda()
        generation_config = FastESM3GenerationConfig(
            num_steps=2,
            temperature=1.0,
            seed=73,
        )

        def run() -> str:
            return model.generate("MK__A", generation_config)

    first_child = next(model.children())
    first_child.eval()
    expected_training_states = tuple(module.training for module in model.modules())

    traces: list[torch.Tensor | str] = []
    for _ in range(2):
        torch.manual_seed(59)
        torch.cuda.manual_seed_all(59)
        traces.append(run())
        assert tuple(module.training for module in model.modules()) == expected_training_states

    if isinstance(traces[0], str):
        assert traces[0] == traces[1]
    else:
        assert torch.is_tensor(traces[0])
        assert traces[0].is_cuda
        assert torch.equal(traces[0], traces[1])


@pytest.mark.parametrize("family", ("dplm", "dplm2", "esm3"))
def test_generation_restores_mixed_training_state_after_cuda_forward_failure(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert torch.cuda.is_available(), "Generation CUDA contract requires CUDA."
    if family == "dplm":
        model = DPLMForMaskedLM(
            DPLMConfig(**_diffusion_config(33)),
            dropout=0.2,
        ).train().cuda()
        # inputs: (1, 5)
        inputs = torch.tensor([[0, 6, 32, 8, 2]], device="cuda")

        def run():
            return model.generate(inputs, max_iter=2)

    elif family == "dplm2":
        model = DPLM2ForMaskedLM(
            DPLM2Config(**_diffusion_config(64)),
            dropout=0.2,
        ).train().cuda()
        # inputs: (1, 8)
        inputs = torch.tensor([[33, 50, 50, 34, 0, 6, 6, 2]], device="cuda")

        def run():
            return model.generate(inputs, max_iter=2)

    else:
        model = FastESM3Model(
            FastESM3Config(
                hidden_size=64,
                num_attention_heads=4,
                num_vector_heads=8,
                num_hidden_layers=1,
            )
        ).train().cuda()

        def run():
            return model.generate(
                "MK__A",
                FastESM3GenerationConfig(num_steps=2, seed=73),
            )

    next(model.children()).eval()
    expected_training_states = tuple(module.training for module in model.modules())

    def fail_forward(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic generation forward failure")

    monkeypatch.setattr(model, "forward", fail_forward)
    with pytest.raises(RuntimeError, match="synthetic generation forward failure"):
        run()
    assert tuple(module.training for module in model.modules()) == expected_training_states


def test_esmplusplus_flex_sequence_masks_reuse_on_cuda_and_match_sdpa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert torch.cuda.is_available(), "ESM++ Flex CUDA contract requires CUDA."
    torch.manual_seed(79)
    sdpa = TransformerStack(32, 2, 1, attn_backend="sdpa").cuda().to(torch.bfloat16)
    flex = TransformerStack(32, 2, 1, attn_backend="flex_attention").cuda().to(torch.bfloat16)
    flex.load_state_dict(sdpa.state_dict())
    # pattern: (2, 5)
    pattern = torch.tensor(
        ((True, True, True, False, False), (True, True, False, True, False)),
        device="cuda",
    )
    created = 0
    original_create_block_mask = _core.create_block_mask

    def count_create_block_mask(*args, **kwargs):
        nonlocal created
        created += 1
        return original_create_block_mask(*args, **kwargs)

    monkeypatch.setattr(_core, "create_block_mask", count_create_block_mask)
    _core.clear_flex_attention_caches()
    flex_input = torch.randn(2, 5, 32, device="cuda", dtype=torch.bfloat16).requires_grad_()
    sdpa_input = flex_input.detach().clone().requires_grad_()

    flex_output = flex(flex_input, sequence_id=pattern).last_hidden_state
    repeated = flex(flex_input.detach(), sequence_id=pattern.clone()).last_hidden_state
    assert created == 1
    torch.testing.assert_close(repeated, flex_output.detach(), rtol=0.0, atol=0.0)
    flex_output.float().square().mean().backward()
    sdpa_output = sdpa(sdpa_input, sequence_id=pattern).last_hidden_state
    sdpa_output.float().square().mean().backward()
    torch.testing.assert_close(flex_output, sdpa_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(flex_input.grad, sdpa_input.grad, rtol=3e-2, atol=3e-2)
    assert torch.isfinite(flex_input.grad).all()

    # chain_pattern: (2, 5)
    chain_pattern = torch.tensor(
        ((0, 0, 1, -1, -1), (0, 1, 1, 2, -1)),
        device="cuda",
    )
    chain_output = flex(flex_input.detach(), sequence_id=chain_pattern).last_hidden_state
    assert created == 2
    assert torch.isfinite(chain_output).all()
    assert len(_core._flex_block_masks) == 2
    _core.clear_flex_attention_caches()
    assert not _core._flex_block_masks


def test_eager_sdpa_and_flex_match_forward_and_backward_on_cuda() -> None:
    assert torch.cuda.is_available(), "Attention CUDA contract requires CUDA."
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
        attn_backend="sdpa",
    )
    torch.manual_seed(61)
    initial = FastEsmModel(config)
    state = {
        name: tensor.detach().clone()
        for name, tensor in initial.state_dict().items()
    }
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
    outputs: dict[str, torch.Tensor] = {}
    parameter_gradients: dict[str, dict[str, torch.Tensor]] = {}

    with strict_fp32_matmul():
        for backend in ("eager", "sdpa", "flex_attention"):
            model = FastEsmModel(config).cuda().train()
            model.load_state_dict(state)
            model.set_attn_implementation(backend)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            # loss: ()
            loss = output[attention_mask].float().square().mean()
            assert loss.is_cuda
            assert torch.isfinite(loss)
            loss.backward()
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            assert gradients
            assert all(
                torch.isfinite(gradient).all() for gradient in gradients.values()
            )
            # outputs[backend]: (...)
            outputs[backend] = output.detach()
            parameter_gradients[backend] = gradients

    expected_attention_projections = (
        "query",
        "key",
        "value",
        "attention.output.dense",
    )
    assert all(
        any(projection in name for name in parameter_gradients["eager"])
        for projection in expected_attention_projections
    )

    torch.testing.assert_close(outputs["sdpa"], outputs["eager"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        outputs["flex_attention"],
        outputs["sdpa"],
        rtol=1e-5,
        atol=1e-5,
    )
    assert parameter_gradients["sdpa"].keys() == parameter_gradients["eager"].keys()
    for name, eager_gradient in parameter_gradients["eager"].items():
        torch.testing.assert_close(
            parameter_gradients["sdpa"][name],
            eager_gradient,
            rtol=1e-5,
            atol=1e-6,
        )
    assert (
        parameter_gradients["flex_attention"].keys()
        == parameter_gradients["sdpa"].keys()
    )
    for name, sdpa_gradient in parameter_gradients["sdpa"].items():
        torch.testing.assert_close(
            parameter_gradients["flex_attention"][name],
            sdpa_gradient,
            rtol=1e-5,
            atol=1e-5,
        )


def test_ttt_step_reset_and_save_reload_execute_on_cuda(tmp_path: Path) -> None:
    assert torch.cuda.is_available(), "TTT CUDA contract requires CUDA."
    model = DummyPretrainedTTTModel(DummyPretrainedTTTConfig()).train().cuda()
    model._ttt_ensure_initialized()
    initial_state = model._ttt_snapshot_lora_state()
    assert initial_state
    assert all(
        tensor.is_cuda
        for module_state in initial_state
        for tensor in module_state.values()
    )
    ttt_config = {
        "steps": 1,
        "ags": 1,
        "batch_size": 2,
        "mask_ratio": 0.5,
        "bert_leave_prob": 0.0,
        "bert_replace_prob": 0.0,
        "seed": 67,
        "initial_state_reset": True,
    }

    first_metrics = model.ttt(seq="ACDE", ttt_config=ttt_config)
    first_adapted = model._ttt_snapshot_lora_state()
    assert len(first_metrics["losses"]) == 1
    assert all(torch.isfinite(torch.tensor(first_metrics["losses"])))
    assert any(
        not torch.equal(initial[name], adapted[name])
        for initial, adapted in zip(initial_state, first_adapted, strict=True)
        for name in initial
    )

    model.ttt_reset()
    reset_state = model._ttt_snapshot_lora_state()
    for initial, reset in zip(initial_state, reset_state, strict=True):
        for name in initial:
            torch.testing.assert_close(reset[name], initial[name])

    second_metrics = model.ttt(seq="ACDE", ttt_config=ttt_config)
    second_adapted = model._ttt_snapshot_lora_state()
    assert second_metrics["losses"] == pytest.approx(first_metrics["losses"])
    for first, second in zip(first_adapted, second_adapted, strict=True):
        for name in first:
            torch.testing.assert_close(second[name], first[name])

    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = DummyPretrainedTTTModel.from_pretrained(
        tmp_path,
        local_files_only=True,
    ).cuda()
    restored_state = restored._ttt_snapshot_lora_state()
    assert restored._ttt_initialized is True
    for expected, observed in zip(second_adapted, restored_state, strict=True):
        for name in expected:
            assert observed[name].is_cuda
            torch.testing.assert_close(observed[name], expected[name])

    restored.ttt_reset()
    restored_reset = restored._ttt_snapshot_lora_state()
    for initial, observed in zip(initial_state, restored_reset, strict=True):
        for name in initial:
            torch.testing.assert_close(observed[name], initial[name])
