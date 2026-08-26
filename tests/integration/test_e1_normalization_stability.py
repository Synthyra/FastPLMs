"""Evidence that the E1 forward pass is fixed by its revision, not its environment.

Two properties are gated here. A stored ``"flex"`` backend, the spelling the
official Profluent source serializes, must reach the compiled Flex
implementation on a real checkpoint. And E1 normalization must stay within a
recorded bound of the fused Triton RMSNorm that the official source selects
whenever ``kernels`` happens to resolve it, so that installing or removing an
optional package cannot move published numbers.

Measured on NVIDIA GH200 480GB, CUDA 13.0, torch 2.13.0+cu130,
transformers 5.13.0, BF16 parameters, four 64-residue sequences.
"""

from __future__ import annotations

import importlib
import importlib.util
import random
import sys
import pytest
import torch
import torch.nn.functional as F
from collections.abc import Callable, Iterator
from huggingface_hub import snapshot_download
from pathlib import Path

from fastplms.models.e1 import modeling_e1
from fastplms.registry import get_model_registry
from tests.conftest import CANONICAL_AAS, SEED


# The kernel repository is a plain model repository without the kernel status
# metadata that `kernels` 0.15 requires, so the pinned snapshot is imported
# directly. This is test evidence about an external implementation and never a
# FastPLMs runtime dependency.
TRITON_LAYER_NORM_REPO = "kernels-community/triton-layer-norm"
TRITON_LAYER_NORM_REVISION = "5ebc83aa387c282ff3f233bc3022c3a8be33a013"
MODEL_ID = "e1_150m"
NUM_SEQUENCES = 4
SEQUENCE_LENGTH = 64
# Bounds recorded from the measurement described in the module docstring.
# Elementwise BF16 normalization agrees to a relative L2 near 2e-5, and the
# difference accumulated across every layer reaches a relative L2 near 3e-3.
MAX_ELEMENTWISE_RELATIVE_L2 = 1e-4
MAX_LOGIT_RELATIVE_L2 = 1e-2
MIN_RESIDUE_COSINE = 0.999


@pytest.fixture(scope="module")
def triton_rms_norm() -> Iterator[Callable[..., torch.Tensor]]:
    """Import the pinned fused Triton RMSNorm used by the official E1 source."""

    snapshot = Path(
        snapshot_download(
            TRITON_LAYER_NORM_REPO,
            revision=TRITON_LAYER_NORM_REVISION,
            allow_patterns=["build/torch-universal/**"],
        )
    )
    package_root = str(snapshot / "build" / "torch-universal")
    sys.path.insert(0, package_root)
    try:
        assert importlib.util.find_spec("triton_layer_norm") is not None
        yield importlib.import_module("triton_layer_norm").rms_norm_fn
    finally:
        sys.path.remove(package_root)
        sys.modules.pop("triton_layer_norm", None)


def _sequences() -> list[str]:
    generator = random.Random(SEED)
    return [
        "M" + "".join(generator.choices(CANONICAL_AAS, k=SEQUENCE_LENGTH - 1))
        for _ in range(NUM_SEQUENCES)
    ]


def _load_model(attn_backend: str, device: torch.device) -> torch.nn.Module:
    spec = get_model_registry()[MODEL_ID]
    model = modeling_e1.E1ForMaskedLM.from_pretrained(
        spec.fast.repo_id,
        revision=spec.fast.revision,
        attn_backend=attn_backend,
        dtype=torch.bfloat16,
        device_map=device,
    )
    return model.eval()


def _encoder_inputs(model: torch.nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    batch = model.model.prep_tokens.get_batch_kwargs(_sequences(), device=device)
    return {
        name: batch[name]
        for name in (
            "input_ids",
            "within_seq_position_ids",
            "global_position_ids",
            "sequence_ids",
        )
    }


def _relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    # candidate, reference: identical shapes, compared in float32
    tiny = torch.finfo(torch.float32).tiny
    difference = torch.linalg.vector_norm(candidate.float() - reference.float())
    return float(difference / torch.linalg.vector_norm(reference.float()).clamp_min(tiny))


@pytest.mark.gpu
@pytest.mark.checkpoint
def test_stored_flex_spelling_loads_and_matches_the_canonical_backend() -> None:
    """The official ``"flex"`` backend name must select compiled Flex attention."""

    device = torch.device("cuda")
    legacy = _load_model("flex", device)
    assert legacy.config.attn_backend == "flex_attention"
    assert legacy.config._attn_implementation == "flex_attention"

    inputs = _encoder_inputs(legacy, device)
    with torch.inference_mode():
        legacy_logits = legacy(**inputs).logits.detach().clone()
    del legacy
    torch.cuda.empty_cache()

    canonical = _load_model("flex_attention", device)
    with torch.inference_mode():
        canonical_logits = canonical(**inputs).logits.detach().clone()
    del canonical
    torch.cuda.empty_cache()

    # The two names select one implementation, so this is an identity rather
    # than a numerical tolerance.
    assert torch.equal(legacy_logits, canonical_logits)


@pytest.mark.gpu
@pytest.mark.network
def test_elementwise_normalization_matches_the_fused_triton_kernel(
    triton_rms_norm: Callable[..., torch.Tensor],
) -> None:
    """Record how far the two RMSNorm implementations drift elementwise."""

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(SEED)
    for hidden_size in (960, 1152, 1536):
        # hidden_states: (b=2, l=512, d=hidden_size)
        hidden_states = torch.randn(
            2,
            512,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        # weight: (d,) held away from zero so the relative comparison is meaningful
        weight = (
            torch.rand(hidden_size, device=device, dtype=torch.bfloat16, generator=generator) * 0.5
            + 0.75
        )
        candidate = F.rms_norm(hidden_states, (hidden_size,), weight, 1e-5)  # (b, l, d)
        reference = triton_rms_norm(  # (b, l, d)
            x=hidden_states,
            weight=weight,
            bias=None,
            residual=None,
            eps=1e-5,
            dropout_p=0.0,
            prenorm=False,
            residual_in_fp32=False,
        ).to(hidden_states.dtype)

        relative_l2 = _relative_l2(candidate, reference)
        assert relative_l2 <= MAX_ELEMENTWISE_RELATIVE_L2, (
            f"d={hidden_size}: relative L2={relative_l2}"
        )


@pytest.mark.gpu
@pytest.mark.checkpoint
@pytest.mark.network
def test_full_forward_pass_survives_swapping_the_normalization_kernel(
    triton_rms_norm: Callable[..., torch.Tensor],
) -> None:
    """Bound the end-to-end effect of the environment-selected fused kernel.

    The official E1 source uses whichever normalization its environment can
    resolve. Running one FastPLMs model both ways isolates that single variable,
    so a residual difference here is attributable to the kernel and nothing else.
    """

    device = torch.device("cuda")
    model = _load_model("flex_attention", device)
    inputs = _encoder_inputs(model, device)

    with torch.inference_mode():
        unfused_logits = model(**inputs).logits.detach().clone()  # (b, l, vocab)

    unfused_forward = modeling_e1.RMSNorm.forward

    def fused_forward(self: modeling_e1.RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (b, l, d)
        return triton_rms_norm(
            x=hidden_states,
            weight=self.weight,
            bias=None,
            residual=None,
            eps=self.variance_epsilon,
            dropout_p=0.0,
            prenorm=False,
            residual_in_fp32=False,
        ).to(hidden_states.dtype)

    modeling_e1.RMSNorm.forward = fused_forward
    try:
        with torch.inference_mode():
            fused_logits = model(**inputs).logits.detach().clone()  # (b, l, vocab)
    finally:
        modeling_e1.RMSNorm.forward = unfused_forward

    residue_mask = inputs["sequence_ids"] != -1  # (b, l)
    fused_residues = fused_logits[residue_mask].float()  # (residues, vocab)
    unfused_residues = unfused_logits[residue_mask].float()  # (residues, vocab)

    relative_l2 = _relative_l2(fused_residues, unfused_residues)
    residue_cosine = F.cosine_similarity(fused_residues, unfused_residues, dim=-1)  # (residues,)

    confidence, expected_top1 = unfused_residues.softmax(-1).max(-1)  # (residues,), (residues,)
    confident = confidence.ge(0.5)  # (residues,)
    assert bool(confident.any()), "the sequence panel produced no confident biological positions"
    observed_top1 = fused_residues.argmax(-1)  # (residues,)
    top1_agreement = (observed_top1[confident] == expected_top1[confident]).float().mean()

    assert relative_l2 <= MAX_LOGIT_RELATIVE_L2, f"relative L2={relative_l2}"
    assert float(residue_cosine.min()) >= MIN_RESIDUE_COSINE, (
        f"minimum residue cosine={float(residue_cosine.min())}"
    )
    assert float(top1_agreement) == 1.0, f"confident top-1 agreement={float(top1_agreement)}"

    del model
    torch.cuda.empty_cache()
