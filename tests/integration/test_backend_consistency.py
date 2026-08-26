"""Global BF16 equivalence gates across every advertised attention backend."""

from __future__ import annotations

import contextlib
import importlib
import inspect
import random
import pytest
import torch
import torch.nn.functional as F
import transformers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from fastplms.registry import ModelSpec, get_model_registry
from tests.conftest import CANONICAL_AAS, SEED


REGISTRY = get_model_registry()
SEQUENCE_SPECS = tuple(
    spec for spec in REGISTRY.values() if spec.family.tokenizer_mode != "structure"
)
NUM_SEQUENCES = 4
SEQUENCE_LENGTH = 64
GH200_MEASURED_BACKENDS = ("eager", "sdpa", "flex_attention")


@dataclass(frozen=True)
class BF16Contract:
    """Agreement limits between two advertised backends of one family."""

    relative_l2: float
    relative_q999: float
    residue_cosine_p01: float
    pooled_cosine: float
    confident_top1: float


GLOBAL_BF16_CONTRACT = BF16Contract(
    relative_l2=1e-2,
    relative_q999=2.5e-2,
    residue_cosine_p01=0.999,
    pooled_cosine=0.9995,
    confident_top1=0.995,
)

# ANKH stores parameters in BF16 instead of autocasting from FP32, and its
# encoder activations peak near |h| = 0.2 to 0.8. Relative L2 and residue cosine
# both scale with that magnitude, so BF16 quantization alone puts a floor under
# the two scale-sensitive metrics that no implementation can clear. Eager and
# SDPA are bitwise identical in FP32, which
# ``test_relaxed_families_agree_exactly_in_fp32`` gates, so the residual is
# accumulation order rather than attention semantics.
#
# Measured on NVIDIA GH200 480GB, CUDA 13.0, torch 2.13.0+cu130, transformers
# 5.13.0, four 64-residue sequences, eager against SDPA, across all five ANKH
# checkpoints. Worst value, from ankh_base: relative L2 2.303e-02, relative
# Q99.9 2.045e-02, residue cosine p01 0.9986925, pooled cosine 0.9996330. The
# two metrics that already sit inside the global limits keep their global
# values, so this widens exactly what the measurement requires.
ANKH_BF16_CONTRACT = BF16Contract(
    relative_l2=3.0e-2,
    relative_q999=GLOBAL_BF16_CONTRACT.relative_q999,
    residue_cosine_p01=0.998,
    pooled_cosine=GLOBAL_BF16_CONTRACT.pooled_cosine,
    confident_top1=GLOBAL_BF16_CONTRACT.confident_top1,
)
# A family earns an entry here only by passing the FP32 identity gate below.
FAMILY_BF16_CONTRACTS: Mapping[str, BF16Contract] = MappingProxyType(
    {"ankh": ANKH_BF16_CONTRACT}
)
RELAXED_BF16_SPECS = tuple(
    spec for spec in SEQUENCE_SPECS if spec.family.id in FAMILY_BF16_CONTRACTS
)


def _parameter(spec: ModelSpec) -> Any:
    marks: list[Any] = [pytest.mark.gpu]
    if spec.size_category in {"large", "xlarge"}:
        marks.append(pytest.mark.slow)
    if spec.size_category == "xlarge":
        marks.append(pytest.mark.large)
    return pytest.param(spec, id=spec.id, marks=marks)


def _model_class(spec: ModelSpec) -> type[torch.nn.Module]:
    """Resolve the repository source class declared by the model manifest.

    Backend consistency is a source contract. Remote-code loading is
    covered separately by the artifact suite, where the generated artifact
    contains the same source revision as the candidate checkout.
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
        # The sequence-mode family carries padding in ``sequence_ids``, where -1
        # marks a padded position. It has no ``attention_mask`` parameter, and
        # its public forward rejects arguments it does not declare.
        inputs = {
            "input_ids": batch["input_ids"],
            "within_seq_position_ids": batch["within_seq_position_ids"],
            "global_position_ids": batch["global_position_ids"],
            "sequence_ids": batch["sequence_ids"],
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
    # input_ids: (b, l)
    input_ids = inputs["input_ids"]
    # residue_mask: (b, l)
    residue_mask = inputs["attention_mask"].bool()
    for token_id in getattr(tokenizer, "all_special_ids", ()):
        residue_mask &= input_ids.ne(token_id)
    if spec.family.architecture == "ESMC":
        # inputs['sequence_id']: (b, l)
        inputs["sequence_id"] = inputs["attention_mask"].bool()
    if _consumes_decoder_inputs(model):
        inputs["decoder_input_ids"] = input_ids
        # inputs['decoder_attention_mask']: (b, l)
        inputs["decoder_attention_mask"] = inputs["attention_mask"]
    return inputs, residue_mask


def _consumes_decoder_inputs(model: torch.nn.Module) -> bool:
    """Report whether this model both declares and acts on decoder arguments.

    Neither signal is sufficient alone. ``config.is_encoder_decoder`` describes
    the published checkpoint schema, so ANKH keeps it set to match the official
    T5 state even though its advertised ``AutoModel`` exposes only the encoder
    view. The forward signature is not sufficient either, because DPLM declares
    the decoder parameters solely to reject them with a named error. Backend
    measurement therefore feeds decoder inputs only where both hold.
    """

    if not getattr(model.config, "is_encoder_decoder", False):
        return False
    return "decoder_input_ids" in inspect.signature(model.forward).parameters


def _sequence_output(output: object) -> tuple[torch.Tensor, bool]:
    for name in ("logits", "sequence_logits"):
        value = getattr(output, name, None)
        if torch.is_tensor(value):
            return value, True
    value = getattr(output, "last_hidden_state", None)
    if torch.is_tensor(value):
        return value, False
    raise AssertionError("Advertised sequence model output omitted a residue tensor")


def _assert_bf16_contract(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    residue_mask: torch.Tensor,
    context: str,
    *,
    has_logits: bool,
    contract: BF16Contract,
) -> None:
    # candidate: (...), reference: (...), residue_mask: (b, l)
    assert candidate.shape == reference.shape
    assert candidate.ndim == 3
    # candidate_f: (...)
    candidate_f = candidate.float()
    # reference_f: (...)
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
    # residue_cosine_p01: ()
    residue_cosine_p01 = torch.quantile(
        F.cosine_similarity(valid_candidate, valid_reference, dim=-1),
        0.01,
    )

    # M: (...)
    M = residue_mask.unsqueeze(-1).float()
    candidate_pooled = (candidate_f * M).sum(1) / M.sum(1).clamp_min(1)
    reference_pooled = (reference_f * M).sum(1) / M.sum(1).clamp_min(1)
    pooled_cosine = F.cosine_similarity(candidate_pooled, reference_pooled, dim=-1)

    assert float(relative_l2) <= contract.relative_l2, f"{context}: relative L2={relative_l2}"
    assert float(relative_q999) <= contract.relative_q999, (
        f"{context}: relative Q99.9={relative_q999}"
    )
    assert float(residue_cosine_p01) >= contract.residue_cosine_p01, (
        f"{context}: residue cosine p01={residue_cosine_p01}"
    )
    assert bool((pooled_cosine >= contract.pooled_cosine).all()), (
        f"{context}: per-sequence pooled cosine={pooled_cosine.tolist()}"
    )
    if has_logits:
        # reference_probabilities: (...)
        reference_probabilities = reference_f.softmax(-1)
        # confidence: (...), reference_top1: (...)
        confidence, reference_top1 = reference_probabilities.max(-1)
        # confident_mask: (...)
        confident_mask = residue_mask & confidence.ge(0.5)
        assert bool(confident_mask.any()), f"{context}: no confident biological positions"
        # candidate_top1: (...)
        candidate_top1 = candidate_f.argmax(-1)
        # top1_agreement: ()
        top1_agreement = (
            (candidate_top1[confident_mask] == reference_top1[confident_mask]).float().mean()
        )
        assert float(top1_agreement) >= contract.confident_top1, (
            f"{context}: confident top-1 agreement={top1_agreement}"
        )


def _measure_backends(
    spec: ModelSpec,
    device: torch.device,
    dtype: torch.dtype,
    *,
    autocast: bool,
) -> tuple[dict[str, tuple[torch.Tensor, bool]], torch.Tensor]:
    """Run every GH200-measurable backend of one model on one shared batch.

    Each backend maps to its residue tensor and whether that tensor is logits.
    The tensor is (b, l, v) for a family that returns logits and (b, l, d)
    otherwise, so the caller compares like with like rather than assuming one
    trailing width. The returned mask is (b, l).
    """

    model = _model_class(spec).from_pretrained(
        spec.fast.repo_id,
        revision=spec.fast.revision,
        dtype=dtype,
        device_map=device,
    )
    model.eval()
    inputs, residue_mask = _prepare_inputs(spec, model, _sequences(), device)
    # A prepared argument the forward does not declare raises a bare TypeError
    # that reads like any other failure, which is how the E1 and ANKH rows of
    # this gate went unexecuted. Name the offending key instead.
    unexpected = sorted(set(inputs) - set(inspect.signature(model.forward).parameters))
    assert not unexpected, (
        f"{spec.id}: prepared inputs {unexpected} are not parameters of "
        f"{type(model).__name__}.forward, so this family would not be measured"
    )

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
            if autocast
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), numeric_context:
            # output_tensor: (b, l, v) with logits, otherwise (b, l, d)
            output_tensor, has_logits = _sequence_output(model(**inputs))
            outputs[backend] = output_tensor.detach().clone(), has_logits

    del model
    torch.cuda.empty_cache()
    return outputs, residue_mask


@pytest.mark.parametrize("spec", [_parameter(spec) for spec in SEQUENCE_SPECS])
def test_gh200_backends_meet_global_bf16_contract(spec: ModelSpec) -> None:
    """Measure only the no-download GH200 eager, SDPA, and Flex matrix."""

    device = torch.device("cuda")
    use_bf16_autocast = spec.family.bf16_execution == "fp32_parameters_autocast"
    outputs, residue_mask = _measure_backends(
        spec,
        device,
        torch.float32 if use_bf16_autocast else torch.bfloat16,
        autocast=use_bf16_autocast,
    )
    contract = FAMILY_BF16_CONTRACTS.get(spec.family.id, GLOBAL_BF16_CONTRACT)

    assert "sdpa" in outputs
    reference, reference_has_logits = outputs["sdpa"]
    for backend, (candidate, has_logits) in outputs.items():
        if backend != "sdpa":
            assert has_logits is reference_has_logits
            _assert_bf16_contract(
                candidate,
                reference,
                residue_mask,
                f"{spec.id}:sdpa-vs-{backend}",
                has_logits=has_logits,
                contract=contract,
            )

    del outputs
    torch.cuda.empty_cache()


@pytest.mark.parametrize("spec", [_parameter(spec) for spec in RELAXED_BF16_SPECS])
def test_relaxed_families_agree_exactly_in_fp32(spec: ModelSpec) -> None:
    """A relaxed BF16 band is only defensible when FP32 execution is identical.

    Widening a numerical limit hides an implementation bug unless the
    implementations are first shown to compute the same function. Running the
    same batch through every advertised backend in FP32 makes that explicit: any
    mask, bias, or dispatch difference survives the wider dtype, while pure BF16
    accumulation order does not.
    """

    device = torch.device("cuda")
    outputs, _ = _measure_backends(spec, device, torch.float32, autocast=False)

    reference = outputs["sdpa"][0]
    for backend, (candidate, _) in outputs.items():
        if backend == "sdpa":
            continue
        assert torch.equal(candidate, reference), (
            f"{spec.id}: {backend} differs from sdpa in FP32; the relaxed BF16 band in "
            f"FAMILY_BF16_CONTRACTS[{spec.family.id!r}] is not justified"
        )

    del outputs
    torch.cuda.empty_cache()
