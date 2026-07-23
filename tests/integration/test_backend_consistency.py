"""Global BF16 equivalence gates across every advertised attention backend."""

from __future__ import annotations

import contextlib
import importlib
import random
from collections.abc import Sequence
from typing import Any

import pytest
import torch
import torch.nn.functional as F
import transformers

from fastplms.registry import ModelSpec, get_model_registry
from tests.conftest import CANONICAL_AAS, SEED

REGISTRY = get_model_registry()
SEQUENCE_SPECS = tuple(
    spec for spec in REGISTRY.values() if spec.family.tokenizer_mode != "structure"
)
NUM_SEQUENCES = 4
SEQUENCE_LENGTH = 64
GH200_MEASURED_BACKENDS = ("eager", "sdpa", "flex_attention")


def _parameter(spec: ModelSpec) -> Any:
    marks: list[Any] = [pytest.mark.gpu]
    if spec.size_category in {"large", "xlarge"}:
        marks.append(pytest.mark.slow)
    if spec.size_category == "xlarge":
        marks.append(pytest.mark.large)
    return pytest.param(spec, id=spec.id, marks=marks)


def _model_class(spec: ModelSpec) -> type[torch.nn.Module]:
    """Resolve the installed package class declared by the model manifest.

    Backend consistency is a package-source contract. Remote-code loading is
    covered separately by the artifact suite, where the generated artifact
    contains the same source revision as the candidate package.
    """

    advertised = set(spec.auto_map)
    if spec.family.id == "ankh":
        name = "AutoModel"
    elif "AutoModelForMaskedLM" in advertised:
        name = "AutoModelForMaskedLM"
    else:
        name = "AutoModel"
    assert name in advertised, f"{spec.id} does not advertise {name}"
    qualified_name = spec.auto_map[name]
    module_name, class_name = qualified_name.rsplit(".", maxsplit=1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    assert issubclass(model_class, torch.nn.Module)
    return model_class


def _sequences() -> list[str]:
    generator = random.Random(SEED)
    return [
        "M" + "".join(generator.choices(CANONICAL_AAS, k=SEQUENCE_LENGTH - 1))
        for _ in range(NUM_SEQUENCES)
    ]


def _prepare_inputs(
    spec: ModelSpec,
    model: torch.nn.Module,
    sequences: Sequence[str],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if spec.family.tokenizer_mode == "sequence":
        batch = model.model.prep_tokens.get_batch_kwargs(sequences, device=device)
        inputs = {
            "input_ids": batch["input_ids"],
            "within_seq_position_ids": batch["within_seq_position_ids"],
            "global_position_ids": batch["global_position_ids"],
            "sequence_ids": batch["sequence_ids"],
            "attention_mask": batch["sequence_ids"].ne(-1).long(),
        }
        return inputs, batch["sequence_ids"].ge(0)

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            spec.fast.repo_id,
            revision=spec.fast.revision,
            trust_remote_code=True,
        )
    tokenize_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
    }
    sequence_tokenizer = getattr(model, "_tokenize_sequence_batch", None)
    if callable(sequence_tokenizer):
        encoded = sequence_tokenizer(
            list(sequences),
            tokenizer=tokenizer,
            **tokenize_kwargs,
        )
    else:
        encoded = tokenizer(list(sequences), **tokenize_kwargs)
    inputs = {name: value.to(device) for name, value in encoded.items() if torch.is_tensor(value)}
    input_ids = inputs["input_ids"]
    residue_mask = inputs["attention_mask"].bool()
    for token_id in getattr(tokenizer, "all_special_ids", ()):
        residue_mask &= input_ids.ne(token_id)
    if spec.family.architecture == "ESMC":
        inputs["sequence_id"] = inputs["attention_mask"].bool()
    if getattr(model.config, "is_encoder_decoder", False):
        inputs["decoder_input_ids"] = input_ids
        inputs["decoder_attention_mask"] = inputs["attention_mask"]
    return inputs, residue_mask


def _sequence_output(output: object) -> tuple[torch.Tensor, bool]:
    for name in ("logits", "sequence_logits"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value, True
    value = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(value):
        return value, False
    raise AssertionError("Advertised sequence model output omitted a residue tensor")


def _assert_global_bf16_contract(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    residue_mask: torch.Tensor,
    context: str,
    *,
    has_logits: bool,
) -> None:
    assert candidate.shape == reference.shape
    assert candidate.ndim == 3
    candidate_f = candidate.float()
    reference_f = reference.float()
    valid_candidate = candidate_f[residue_mask]
    valid_reference = reference_f[residue_mask]
    difference = valid_candidate - valid_reference
    tiny = torch.finfo(torch.float32).tiny
    relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        valid_reference
    ).clamp_min(tiny)
    relative_q999 = torch.quantile(difference.abs().reshape(-1), 0.999) / torch.quantile(
        valid_reference.abs().reshape(-1), 0.999
    ).clamp_min(tiny)
    residue_cosine_p01 = torch.quantile(
        F.cosine_similarity(valid_candidate, valid_reference, dim=-1),
        0.01,
    )

    M = residue_mask.unsqueeze(-1).float()
    candidate_pooled = (candidate_f * M).sum(1) / M.sum(1).clamp_min(1)
    reference_pooled = (reference_f * M).sum(1) / M.sum(1).clamp_min(1)
    pooled_cosine = F.cosine_similarity(candidate_pooled, reference_pooled, dim=-1)

    assert float(relative_l2) <= 1e-2, f"{context}: relative L2={relative_l2}"
    assert float(relative_q999) <= 2.5e-2, f"{context}: relative Q99.9={relative_q999}"
    assert float(residue_cosine_p01) >= 0.999, f"{context}: residue cosine p01={residue_cosine_p01}"
    assert bool((pooled_cosine >= 0.9995).all()), (
        f"{context}: per-sequence pooled cosine={pooled_cosine.tolist()}"
    )
    if has_logits:
        reference_probabilities = reference_f.softmax(-1)
        confidence, reference_top1 = reference_probabilities.max(-1)
        confident_mask = residue_mask & confidence.ge(0.5)
        assert bool(confident_mask.any()), f"{context}: no confident biological positions"
        candidate_top1 = candidate_f.argmax(-1)
        top1_agreement = (
            (candidate_top1[confident_mask] == reference_top1[confident_mask]).float().mean()
        )
        assert float(top1_agreement) >= 0.995, (
            f"{context}: confident top-1 agreement={top1_agreement}"
        )


@pytest.mark.parametrize("spec", [_parameter(spec) for spec in SEQUENCE_SPECS])
def test_gh200_backends_meet_global_bf16_contract(spec: ModelSpec) -> None:
    """Measure only the no-download GH200 eager, SDPA, and Flex matrix."""

    device = torch.device("cuda")
    model_class = _model_class(spec)
    use_bf16_autocast = spec.family.bf16_execution == "fp32_parameters_autocast"
    load_dtype = torch.float32 if use_bf16_autocast else torch.bfloat16
    model = model_class.from_pretrained(
        spec.fast.repo_id,
        revision=spec.fast.revision,
        dtype=load_dtype,
        device_map=device,
    ).eval()
    inputs, residue_mask = _prepare_inputs(spec, model, _sequences(), device)

    outputs: dict[str, tuple[torch.Tensor, bool]] = {}
    measured_backends = tuple(
        backend for backend in spec.family.attention if backend in GH200_MEASURED_BACKENDS
    )
    assert measured_backends, f"{spec.id}: no GH200 backend is declared"
    for backend in measured_backends:
        assert hasattr(model, "set_attn_implementation")
        model.set_attn_implementation(backend)
        resolved = getattr(model.config, "_attn_implementation", None)
        if resolved is None:
            resolved = getattr(model.config, "attn_implementation", None)
        assert resolved == backend, f"{spec.id}: requested {backend}, resolved {resolved}"
        numeric_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16_autocast
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), numeric_context:
            output_tensor, has_logits = _sequence_output(model(**inputs))
            outputs[backend] = output_tensor.detach(), has_logits

    assert "sdpa" in outputs
    reference, reference_has_logits = outputs["sdpa"]
    for backend, (candidate, has_logits) in outputs.items():
        if backend != "sdpa":
            assert has_logits is reference_has_logits
            _assert_global_bf16_contract(
                candidate,
                reference,
                residue_mask,
                f"{spec.id}:sdpa-vs-{backend}",
                has_logits=has_logits,
            )

    del model, outputs
    torch.cuda.empty_cache()
