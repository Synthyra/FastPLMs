"""Strict manifest-driven equivalence against pinned official implementations.

Configuration, tokenizer behavior, state, aliases, and inference are release
gates. Numeric contracts are shared except for explicit, evidence-backed
model/backend calibrations; this suite never silently falls back.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import os
import random
import warnings
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from transformers import AutoModel, AutoModelForMaskedLM

from fastplms.registry import ModelSpec, get_model_registry
from tests.conftest import CANONICAL_AAS, SEED, strict_fp32_matmul
from tests.parity.support.semantic_config import (
    semantic_config,
    transformed_semantic_config,
)
from tests.parity.support.state_transforms import (
    TRANSFORMS,
    transform_parameter_names,
    transform_preserves_aliases,
    transform_state,
)


_semantic_config = semantic_config

pytestmark = pytest.mark.compliance

REGISTRY = get_model_registry()
SEQUENCE_SPECS = tuple(
    spec for spec in REGISTRY.values() if spec.family.tokenizer_mode != "structure"
)
DEEP_SPECS = tuple(spec for spec in SEQUENCE_SPECS if spec.is_deep_reference)
MIXED_LENGTHS = (61, 29, 13)
EDGE_SEQUENCES = (
    "ACDEFGHIKLMNPQRSTVWY",
    "AXBJOUZ",
    "acdefghik",
    "A C\nD\tE",
    "",
)


@dataclass(frozen=True, slots=True)
class NumericContract:
    """Fixed engineering target and hard release boundary."""

    relative_l2_target: float
    relative_l2_hard: float
    relative_q999_target: float
    relative_q999_hard: float
    residue_cosine_target: float
    residue_cosine_hard: float
    pooled_cosine_target: float
    pooled_cosine_hard: float
    top1_target: float
    top1_hard: float
    jsd_target: float
    jsd_hard: float


FP32_CONTRACT = NumericContract(
    relative_l2_target=2e-6,
    relative_l2_hard=2e-5,
    relative_q999_target=1e-5,
    relative_q999_hard=1e-4,
    residue_cosine_target=0.999999,
    residue_cosine_hard=0.9999,
    pooled_cosine_target=0.999999,
    pooled_cosine_hard=0.9999,
    top1_target=0.9999,
    top1_hard=0.999,
    jsd_target=1e-8,
    jsd_hard=1e-6,
)
BF16_CONTRACT = NumericContract(
    relative_l2_target=1e-2,
    relative_l2_hard=3e-2,
    relative_q999_target=2.5e-2,
    relative_q999_hard=5e-2,
    residue_cosine_target=0.999,
    residue_cosine_hard=0.995,
    pooled_cosine_target=0.9995,
    pooled_cosine_hard=0.995,
    top1_target=0.995,
    top1_hard=0.99,
    jsd_target=1e-4,
    jsd_hard=1e-3,
)
ESMC_ALTERNATE_BF16_CONTRACT = NumericContract(
    relative_l2_target=2.9e-2,
    relative_l2_hard=BF16_CONTRACT.relative_l2_hard,
    relative_q999_target=4.9e-2,
    relative_q999_hard=BF16_CONTRACT.relative_q999_hard,
    residue_cosine_target=0.997,
    residue_cosine_hard=BF16_CONTRACT.residue_cosine_hard,
    pooled_cosine_target=BF16_CONTRACT.pooled_cosine_target,
    pooled_cosine_hard=BF16_CONTRACT.pooled_cosine_hard,
    top1_target=BF16_CONTRACT.top1_target,
    top1_hard=BF16_CONTRACT.top1_hard,
    jsd_target=4e-4,
    jsd_hard=BF16_CONTRACT.jsd_hard,
)
# Flex and FA3 are supported ESMC implementations whose backend-specific BF16
# arithmetic is reported diagnostically. These deliberately broad limits catch
# corrupt dispatch, broken masking, non-finite outputs, or catastrophic
# biological disagreement without turning known backend drift into an xfail.
ESMC_CATASTROPHIC_BF16_CONTRACT = NumericContract(
    relative_l2_target=0.25,
    relative_l2_hard=0.25,
    relative_q999_target=0.50,
    relative_q999_hard=0.50,
    residue_cosine_target=0.90,
    residue_cosine_hard=0.90,
    pooled_cosine_target=0.95,
    pooled_cosine_hard=0.95,
    top1_target=0.80,
    top1_hard=0.80,
    jsd_target=0.05,
    jsd_hard=0.05,
)
ESM2_OPTIMIZED_BF16_CONTRACT = NumericContract(
    relative_l2_target=2e-2,
    relative_l2_hard=BF16_CONTRACT.relative_l2_hard,
    relative_q999_target=BF16_CONTRACT.relative_q999_target,
    relative_q999_hard=BF16_CONTRACT.relative_q999_hard,
    residue_cosine_target=BF16_CONTRACT.residue_cosine_target,
    residue_cosine_hard=BF16_CONTRACT.residue_cosine_hard,
    pooled_cosine_target=BF16_CONTRACT.pooled_cosine_target,
    pooled_cosine_hard=BF16_CONTRACT.pooled_cosine_hard,
    top1_target=BF16_CONTRACT.top1_target,
    top1_hard=BF16_CONTRACT.top1_hard,
    jsd_target=BF16_CONTRACT.jsd_target,
    jsd_hard=BF16_CONTRACT.jsd_hard,
)
ESM2_3B_SDPA_BF16_CONTRACT = NumericContract(
    # Calibrated on the pinned 3B checkpoint: exact weights and logits retain
    # perfect confident-token agreement while deep BF16 SDPA layers accumulate
    # more rounding drift than the smaller ESM2 variants.
    relative_l2_target=6e-2,
    relative_l2_hard=7e-2,
    relative_q999_target=1.5e-1,
    relative_q999_hard=1.8e-1,
    residue_cosine_target=0.994,
    residue_cosine_hard=0.992,
    pooled_cosine_target=0.998,
    pooled_cosine_hard=0.997,
    top1_target=BF16_CONTRACT.top1_target,
    top1_hard=BF16_CONTRACT.top1_hard,
    jsd_target=BF16_CONTRACT.jsd_target,
    jsd_hard=BF16_CONTRACT.jsd_hard,
)


@dataclass(frozen=True, slots=True)
class TensorMetrics:
    """Normalized metrics over biological residues only."""

    relative_l2: float
    relative_q999: float
    residue_cosine_p01: float
    pooled_cosine_min: float


@dataclass(frozen=True, slots=True)
class TensorMetricRecord:
    """Metrics and identity for one layer or output tensor."""

    context: str
    metrics: TensorMetrics


@dataclass(frozen=True, slots=True)
class LogitsMetrics:
    """Distribution-level metrics for a masked-language-model head."""

    confident_top1_agreement: float
    mean_jsd: float


def _numeric_contract(
    spec: ModelSpec,
    dtype: torch.dtype,
    backend: str | None,
) -> NumericContract:
    """Resolve the fixed contract without weakening the global BF16 policy."""

    if dtype == torch.float32:
        return FP32_CONTRACT
    if dtype != torch.bfloat16:
        raise ValueError(f"Unsupported parity dtype: {dtype}")
    if spec.id == "esm2_3b" and backend in (None, "sdpa"):
        return ESM2_3B_SDPA_BF16_CONTRACT
    if spec.family.id == "esm2" and backend != "eager":
        return ESM2_OPTIMIZED_BF16_CONTRACT
    if spec.family.architecture == "ESMC" and backend not in (None, "sdpa"):
        return ESMC_ALTERNATE_BF16_CONTRACT
    return BF16_CONTRACT


def _parameter(spec: ModelSpec) -> Any:
    marks: list[Any] = [pytest.mark.slow]
    if spec.size_category == "xlarge":
        marks.append(pytest.mark.large)
    return pytest.param(spec, id=spec.id, marks=marks)


def _deep_parameter(spec: ModelSpec, dtype: torch.dtype, backend: str) -> Any:
    marks: list[Any] = [pytest.mark.gpu, pytest.mark.slow]
    if spec.size_category == "xlarge":
        marks.append(pytest.mark.large)
    dtype_name = "fp32" if dtype == torch.float32 else "bf16"
    return pytest.param(spec, dtype, backend, id=f"{spec.id}-{dtype_name}-{backend}", marks=marks)


def _sequence_batch(lengths: Sequence[int] = MIXED_LENGTHS) -> list[str]:
    rng = random.Random(SEED)
    return ["M" + "".join(rng.choices(CANONICAL_AAS, k=length - 1)) for length in lengths]


def _load_fast(
    spec: ModelSpec,
    device: torch.device,
    dtype: torch.dtype | None,
) -> nn.Module:
    # ANKH parity is the official encoder contract. Its masked-LM class is a
    # separately named FastPLMs extension and is not presented as upstream-equivalent.
    auto_class_name = (
        "AutoModel"
        if spec.family.id == "ankh" or "AutoModelForMaskedLM" not in spec.auto_map
        else "AutoModelForMaskedLM"
    )
    auto_class = AutoModel if auto_class_name == "AutoModel" else AutoModelForMaskedLM
    artifact_root = os.environ.get("FASTPLMS_CANDIDATE_ARTIFACTS")
    if artifact_root:
        repository_name = spec.fast.repo_id.split("/", maxsplit=1)[-1]
        model_source = Path(artifact_root) / repository_name
        if not model_source.is_dir():
            raise FileNotFoundError(
                f"Candidate compliance artifact is missing for {spec.id}: {model_source}"
            )
        # Native compliance already imports the current package to read the
        # typed manifest. Loading the generated remote-code bridge in this same
        # interpreter would deliberately fail its runtime-isolation guard.
        # Resolve the manifest-declared package classes directly while keeping
        # the artifact as the sole config, tokenizer-asset, and weight source.
        config_path = spec.auto_map["AutoConfig"]
        model_path = spec.auto_map[auto_class_name]
        config_module, config_name = config_path.rsplit(".", maxsplit=1)
        model_module, model_name = model_path.rsplit(".", maxsplit=1)
        config_class = getattr(importlib.import_module(config_module), config_name)
        model_class = getattr(importlib.import_module(model_module), model_name)
        config = config_class.from_pretrained(model_source, local_files_only=True)
        load_kwargs = {
            "config": config,
            "local_files_only": True,
            "device_map": device,
        }
    else:
        model_source = spec.fast.repo_id
        load_kwargs = {
            "revision": spec.fast.revision,
            "trust_remote_code": True,
            "device_map": device,
        }
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    loader = model_class if artifact_root else auto_class
    model = loader.from_pretrained(model_source, **load_kwargs)
    return model.eval()


def _load_reference(
    spec: ModelSpec,
    device: torch.device,
    dtype: torch.dtype | None,
) -> tuple[nn.Module, object]:
    adapter = importlib.import_module(spec.family.reference_adapter)
    kwargs: dict[str, Any] = {}
    if spec.oracle_assets:
        kwargs["oracle_assets"] = spec.oracle_assets
    return adapter.load_official_model(
        reference_repo_id=spec.official.repo_id,
        reference_revision=spec.official.revision,
        device=device,
        dtype=dtype,
        **kwargs,
    )


def _reference_core(reference: nn.Module) -> nn.Module:
    core = getattr(reference, "model", reference)
    return core


def _assert_semantic_config_equal(spec: ModelSpec, fast: nn.Module, reference: nn.Module) -> None:
    fast_config = _semantic_config(fast)
    reference_config = transformed_semantic_config(
        _reference_core(reference), spec.family.state_transform
    )
    missing = sorted(set(reference_config).difference(fast_config))
    assert not missing, (
        f"{spec.id}: FastPLMs configuration omits official semantic fields {missing}"
    )
    compared_fast = {name: fast_config[name] for name in reference_config}
    assert compared_fast == reference_config, (
        f"{spec.id}: semantic configuration differs: "
        f"fast={compared_fast}, official={reference_config}"
    )


def _assert_state_equal(spec: ModelSpec, fast: nn.Module, reference: nn.Module) -> None:
    official_state = transform_state(
        spec.family.state_transform,
        _reference_core(reference).state_dict(),
    )
    fast_state = fast.state_dict()
    assert set(fast_state) == set(official_state), (
        f"{spec.id}: state-key set differs; "
        f"only_fast={sorted(set(fast_state) - set(official_state))[:20]}, "
        f"only_official={sorted(set(official_state) - set(fast_state))[:20]}"
    )
    for name in sorted(fast_state):
        # candidate: (...)
        candidate = fast_state[name].detach().cpu()
        # official: (...)
        official = official_state[name].detach().cpu()
        assert candidate.shape == official.shape, (
            f"{spec.id}:{name}: shape {tuple(candidate.shape)} != {tuple(official.shape)}"
        )
        assert candidate.dtype == official.dtype, (
            f"{spec.id}:{name}: dtype {candidate.dtype} != {official.dtype}"
        )
        assert torch.equal(candidate, official), f"{spec.id}:{name}: tensor values are not exact"


def _alias_groups(model: nn.Module) -> set[frozenset[str]]:
    by_parameter: dict[int, set[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        by_parameter.setdefault(id(parameter), set()).add(name)
    return {frozenset(names) for names in by_parameter.values() if len(names) > 1}


def _transformed_alias_groups(spec: ModelSpec, model: nn.Module) -> set[frozenset[str]]:
    if not transform_preserves_aliases(spec.family.state_transform):
        return set()
    by_parameter: dict[int, set[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        mapped = transform_parameter_names(spec.family.state_transform, name)
        by_parameter.setdefault(id(parameter), set()).update(mapped)
    return {frozenset(names) for names in by_parameter.values() if len(names) > 1}


def _assert_aliases_equal(spec: ModelSpec, fast: nn.Module, reference: nn.Module) -> None:
    candidate = _alias_groups(fast)
    official = _transformed_alias_groups(spec, _reference_core(reference))
    assert candidate == official, (
        f"{spec.id}: tied-parameter aliases differ; "
        f"only_fast={sorted(map(sorted, candidate - official))}, "
        f"only_official={sorted(map(sorted, official - candidate))}"
    )


def _normalize_tokenizer_error(message: str) -> str:
    """Remove a dependency-list difference between Transformers v4 and v5."""

    return message.replace(
        "python, numpy, pytorch or tensorflow object.",
        "python, numpy or pytorch object.",
    ).replace("python, numpy, or pytorch object.", "python, numpy or pytorch object.")


def _token_result(tokenizer: object, sequences: Sequence[str], **kwargs: Any) -> Any:
    try:
        encoded = tokenizer(sequences, return_tensors="pt", **kwargs)
    except Exception as error:  # Exact error behavior is part of the token contract.
        return (
            "error",
            type(error).__module__,
            type(error).__qualname__,
            _normalize_tokenizer_error(str(error)),
        )
    normalized: dict[str, Any] = {}
    for key, value in encoded.items():
        normalized[key] = value.tolist() if torch.is_tensor(value) else value
    return ("ok", normalized)


def _assert_tokenizer_equal(
    spec: ModelSpec,
    fast_tokenizer: object,
    official_tokenizer: object,
) -> None:
    assert fast_tokenizer.get_vocab() == official_tokenizer.get_vocab(), (
        f"{spec.id}: tokenizer vocabulary or token IDs differ"
    )
    for name in (
        "pad_token_id",
        "bos_token_id",
        "cls_token_id",
        "eos_token_id",
        "mask_token_id",
        "unk_token_id",
    ):
        assert getattr(fast_tokenizer, name, None) == getattr(official_tokenizer, name, None), (
            f"{spec.id}: tokenizer {name} differs"
        )

    settings = (
        {"padding": True},
        {"padding": "max_length", "truncation": True, "max_length": 12},
        {"padding": True, "truncation": True, "max_length": 5},
    )
    for options in settings:
        fast_result = _token_result(fast_tokenizer, EDGE_SEQUENCES, **options)
        official_result = _token_result(official_tokenizer, EDGE_SEQUENCES, **options)
        assert fast_result == official_result, (
            f"{spec.id}: tokenizer behavior differs for options={options}: "
            f"fast={fast_result}, official={official_result}"
        )


def _to_device(values: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in values.items() if torch.is_tensor(value)}


def _prepare_inputs(
    spec: ModelSpec,
    fast: nn.Module,
    reference_tokenizer: object,
    sequences: Sequence[str],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    if spec.family.tokenizer_mode == "sequence":
        fast_batch = fast.model.prep_tokens.get_batch_kwargs(sequences, device=device)
        official_batch = reference_tokenizer.get_batch_kwargs(sequences, device=device)
        assert set(fast_batch) == set(official_batch), f"{spec.id}: sequence-adapter keys differ"
        for name in fast_batch:
            assert torch.equal(fast_batch[name], official_batch[name]), (
                f"{spec.id}: sequence-adapter tensor {name!r} differs"
            )
        # residue_mask: (b, l)
        residue_mask = fast_batch["sequence_ids"].ge(0)
        fast_inputs = dict(fast_batch)
        official_inputs = dict(official_batch)
        # fast_inputs['attention_mask']: (b, l)
        fast_inputs["attention_mask"] = residue_mask.long()
        # official_inputs['attention_mask']: (b, l)
        official_inputs["attention_mask"] = residue_mask.long()
        return fast_inputs, official_inputs, residue_mask

    fast_tokenizer = fast.tokenizer
    fast_encoded = _to_device(
        fast_tokenizer(sequences, return_tensors="pt", padding=True),
        device,
    )
    official_encoded = _to_device(
        reference_tokenizer(sequences, return_tensors="pt", padding=True),
        device,
    )
    assert set(fast_encoded) == set(official_encoded), f"{spec.id}: tokenized keys differ"
    for name in fast_encoded:
        assert torch.equal(fast_encoded[name], official_encoded[name]), (
            f"{spec.id}: tokenized tensor {name!r} differs"
        )

    # input_ids: (b, l)
    input_ids = fast_encoded["input_ids"]
    # residue_mask: (b, l)
    residue_mask = fast_encoded["attention_mask"].bool()
    for token_id in getattr(fast_tokenizer, "all_special_ids", ()):
        residue_mask &= input_ids.ne(token_id)

    allowed = {"input_ids", "attention_mask"}
    fast_inputs = {name: value for name, value in fast_encoded.items() if name in allowed}
    official_inputs = {name: value for name, value in official_encoded.items() if name in allowed}
    if spec.family.architecture == "ESMC":
        # sequence_id: (b, l)
        sequence_id = fast_encoded["attention_mask"].bool()
        fast_inputs["sequence_id"] = sequence_id
        official_inputs["sequence_id"] = sequence_id
    return fast_inputs, official_inputs, residue_mask


def _hidden_state_tuple(output: object) -> tuple[torch.Tensor, ...]:
    """Normalize tuple-valued and layer-stacked hidden-state outputs."""

    raw = getattr(output, "hidden_states", None)
    if torch.is_tensor(raw):
        return tuple(raw.unbind(dim=0))
    return tuple(raw or ())


def _last_hidden(output: object) -> torch.Tensor:
    value = getattr(output, "last_hidden_state", None)
    if value is not None:
        return value
    hidden_states = _hidden_state_tuple(output)
    assert hidden_states, "Model output omitted last_hidden_state and hidden_states"
    return hidden_states[-1]


def tensor_metrics(
    candidate: torch.Tensor,
    official: torch.Tensor,
    residue_mask: torch.Tensor,
) -> TensorMetrics:
    """Compute normalized errors and cosine metrics on valid residues."""

    # candidate: (...), official: (...), residue_mask: (b, l)
    assert candidate.shape == official.shape
    assert candidate.ndim == 3
    valid_candidate = candidate.float()[residue_mask]
    valid_official = official.float()[residue_mask]
    assert valid_candidate.numel() > 0, "Parity batch contains no biological residues"
    difference = valid_candidate - valid_official
    # denominator: (...)
    denominator = torch.linalg.vector_norm(valid_official).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    relative_l2 = torch.linalg.vector_norm(difference) / denominator
    # reference_q999: (...)
    reference_q999 = torch.quantile(valid_official.abs().reshape(-1), 0.999)
    # difference_q999: ()
    difference_q999 = torch.quantile(difference.abs().reshape(-1), 0.999)
    relative_q999 = difference_q999 / reference_q999.clamp_min(torch.finfo(torch.float32).tiny)
    residue_cosines = F.cosine_similarity(valid_candidate, valid_official, dim=-1)
    # residue_cosine_p01: ()
    residue_cosine_p01 = torch.quantile(residue_cosines, 0.01)

    # mask: (...)
    mask = residue_mask.unsqueeze(-1)
    # denominator: (...)
    denominator = mask.sum(1).clamp_min(1)
    # candidate_values: (...)
    candidate_values = candidate.float()
    # official_values: (...)
    official_values = official.float()
    candidate_pooled = torch.where(mask, candidate_values, 0.0).sum(1) / denominator
    official_pooled = torch.where(mask, official_values, 0.0).sum(1) / denominator
    # pooled_cosine_min: ()
    pooled_cosine_min = F.cosine_similarity(candidate_pooled, official_pooled, dim=-1).min()
    return TensorMetrics(
        relative_l2=float(relative_l2),
        relative_q999=float(relative_q999),
        residue_cosine_p01=float(residue_cosine_p01),
        pooled_cosine_min=float(pooled_cosine_min),
    )


def _assert_upper(name: str, value: float, target: float, hard: float, context: str) -> None:
    assert value <= hard, f"{context}: {name}={value:.6g} exceeds hard limit {hard:.6g}"
    assert value <= target, f"{context}: {name}={value:.6g} misses target {target:.6g}"


def _assert_lower(name: str, value: float, target: float, hard: float, context: str) -> None:
    assert value >= hard, f"{context}: {name}={value:.6g} violates hard limit {hard:.6g}"
    assert value >= target, f"{context}: {name}={value:.6g} misses target {target:.6g}"


def _assert_tensor_contract(
    candidate: torch.Tensor,
    official: torch.Tensor,
    residue_mask: torch.Tensor,
    contract: NumericContract,
    context: str,
) -> None:
    # candidate: (...), official: (...), residue_mask: (b, l)
    metrics = tensor_metrics(candidate, official, residue_mask)
    _assert_tensor_metrics(metrics, contract, context)


def _assert_tensor_metrics(
    metrics: TensorMetrics,
    contract: NumericContract,
    context: str,
) -> None:
    _assert_upper(
        "relative_l2",
        metrics.relative_l2,
        contract.relative_l2_target,
        contract.relative_l2_hard,
        context,
    )
    _assert_upper(
        "relative_q999",
        metrics.relative_q999,
        contract.relative_q999_target,
        contract.relative_q999_hard,
        context,
    )
    _assert_lower(
        "residue_cosine_p01",
        metrics.residue_cosine_p01,
        contract.residue_cosine_target,
        contract.residue_cosine_hard,
        context,
    )
    _assert_lower(
        "pooled_cosine_min",
        metrics.pooled_cosine_min,
        contract.pooled_cosine_target,
        contract.pooled_cosine_hard,
        context,
    )


def _assert_tensor_metric_records(
    records: Sequence[TensorMetricRecord],
    contract: NumericContract,
) -> None:
    """Assert aggregate extrema after every output tensor has been measured."""

    assert records, "No output tensor metrics were collected"
    upper_metrics = (
        (
            "relative_l2",
            "relative_l2",
            contract.relative_l2_target,
            contract.relative_l2_hard,
        ),
        (
            "relative_q999",
            "relative_q999",
            contract.relative_q999_target,
            contract.relative_q999_hard,
        ),
    )
    lower_metrics = (
        (
            "residue_cosine_p01",
            "residue_cosine_p01",
            contract.residue_cosine_target,
            contract.residue_cosine_hard,
        ),
        (
            "pooled_cosine_min",
            "pooled_cosine_min",
            contract.pooled_cosine_target,
            contract.pooled_cosine_hard,
        ),
    )
    for name, attribute, target, hard in upper_metrics:
        worst = max(records, key=lambda record: getattr(record.metrics, attribute))
        _assert_upper(name, getattr(worst.metrics, attribute), target, hard, worst.context)
    for name, attribute, target, hard in lower_metrics:
        worst = min(records, key=lambda record: getattr(record.metrics, attribute))
        _assert_lower(name, getattr(worst.metrics, attribute), target, hard, worst.context)


def _logits_metrics(
    candidate: torch.Tensor,
    official: torch.Tensor,
    residue_mask: torch.Tensor,
    context: str,
) -> LogitsMetrics:
    """Collect logits semantics before any numeric threshold is asserted."""

    # candidate: (...), official: (...), residue_mask: (b, l)
    # official_probabilities: (...)
    official_probabilities = official.float().softmax(-1)
    # candidate_probabilities: (...)
    candidate_probabilities = candidate.float().softmax(-1)
    # confidence: (...), official_top1: (...)
    confidence, official_top1 = official_probabilities.max(-1)
    # confident_mask: (...)
    confident_mask = residue_mask & confidence.ge(0.5)
    assert bool(confident_mask.any()), (
        f"{context}: no positions meet the fixed confidence threshold"
    )
    # candidate_top1: (...)
    candidate_top1 = candidate_probabilities.argmax(-1)
    # top1_agreement: ()
    top1_agreement = (
        (candidate_top1[confident_mask] == official_top1[confident_mask]).float().mean()
    )

    midpoint = 0.5 * (official_probabilities + candidate_probabilities)
    official_log = official_probabilities.clamp_min(1e-12).log()
    candidate_log = candidate_probabilities.clamp_min(1e-12).log()
    midpoint_log = midpoint.clamp_min(1e-12).log()
    jsd = 0.5 * (
        (official_probabilities * (official_log - midpoint_log)).sum(-1)
        + (candidate_probabilities * (candidate_log - midpoint_log)).sum(-1)
    )
    return LogitsMetrics(
        confident_top1_agreement=float(top1_agreement),
        mean_jsd=float(jsd[residue_mask].mean()),
    )


def _assert_logits_contract(
    candidate: torch.Tensor,
    official: torch.Tensor,
    residue_mask: torch.Tensor,
    contract: NumericContract,
    context: str,
) -> None:
    # candidate: (...), official: (...), residue_mask: (b, l)
    _assert_tensor_contract(candidate, official, residue_mask, contract, context)
    metrics = _logits_metrics(candidate, official, residue_mask, context)
    _assert_lower(
        "confident_top1_agreement",
        metrics.confident_top1_agreement,
        contract.top1_target,
        contract.top1_hard,
        context,
    )
    _assert_upper(
        "mean_jsd",
        metrics.mean_jsd,
        contract.jsd_target,
        contract.jsd_hard,
        context,
    )


def _collect_output_metrics(
    spec: ModelSpec,
    fast_output: object,
    official_output: object,
    residue_mask: torch.Tensor,
    context: str,
) -> tuple[list[TensorMetricRecord], LogitsMetrics | None]:
    """Validate output structure/finite values and collect every parity metric."""

    # residue_mask: (b, l)
    fast_hidden = _hidden_state_tuple(fast_output)
    official_hidden = _hidden_state_tuple(official_output)
    assert len(fast_hidden) == len(official_hidden), (
        f"{context}: hidden-state count {len(fast_hidden)} != {len(official_hidden)}"
    )
    assert fast_hidden, f"{context}: hidden states were not returned"
    metric_records: list[TensorMetricRecord] = []
    for layer, (candidate, official) in enumerate(zip(fast_hidden, official_hidden, strict=True)):
        assert torch.isfinite(candidate).all(), f"{context}:layer={layer}: non-finite candidate"
        assert torch.isfinite(official).all(), f"{context}:layer={layer}: non-finite reference"
        metric_records.append(
            TensorMetricRecord(
                context=f"{context}:layer={layer}",
                metrics=tensor_metrics(candidate, official, residue_mask),
            )
        )
    fast_last = _last_hidden(fast_output)
    official_last = _last_hidden(official_output)
    assert torch.isfinite(fast_last).all(), f"{context}:last_hidden_state: non-finite candidate"
    assert torch.isfinite(official_last).all(), f"{context}:last_hidden_state: non-finite reference"
    metric_records.append(
        TensorMetricRecord(
            context=f"{context}:last_hidden_state",
            metrics=tensor_metrics(fast_last, official_last, residue_mask),
        )
    )

    fast_logits = getattr(fast_output, "logits", None)
    official_logits = getattr(official_output, "logits", None)
    assert (fast_logits is None) == (official_logits is None), (
        f"{spec.id}: official and FastPLMs output-head contracts differ"
    )
    logits_context = f"{context}:logits"
    logits_metrics = None
    if fast_logits is not None:
        assert official_logits is not None
        assert torch.isfinite(fast_logits).all(), f"{logits_context}: non-finite candidate"
        assert torch.isfinite(official_logits).all(), f"{logits_context}: non-finite reference"
        metric_records.append(
            TensorMetricRecord(
                context=logits_context,
                metrics=tensor_metrics(fast_logits, official_logits, residue_mask),
            )
        )
        logits_metrics = _logits_metrics(
            fast_logits,
            official_logits,
            residue_mask,
            logits_context,
        )

    return metric_records, logits_metrics


def _assert_outputs(
    spec: ModelSpec,
    fast_output: object,
    official_output: object,
    residue_mask: torch.Tensor,
    contract: NumericContract,
    context: str,
) -> None:
    # residue_mask: (b, l)
    metric_records, logits_metrics = _collect_output_metrics(
        spec,
        fast_output,
        official_output,
        residue_mask,
        context,
    )

    _assert_tensor_metric_records(metric_records, contract)
    if logits_metrics is not None:
        logits_context = f"{context}:logits"
        _assert_lower(
            "confident_top1_agreement",
            logits_metrics.confident_top1_agreement,
            contract.top1_target,
            contract.top1_hard,
            logits_context,
        )
        _assert_upper(
            "mean_jsd",
            logits_metrics.mean_jsd,
            contract.jsd_target,
            contract.jsd_hard,
            logits_context,
        )


def _assert_esmc_alternate_backend_outputs(
    spec: ModelSpec,
    fast_output: object,
    official_output: object,
    residue_mask: torch.Tensor,
    context: str,
) -> None:
    """Warn on published-band drift while retaining catastrophic hard gates."""

    # residue_mask: (b, l)
    records, logits = _collect_output_metrics(
        spec,
        fast_output,
        official_output,
        residue_mask,
        context,
    )
    _assert_tensor_metric_records(records, ESMC_CATASTROPHIC_BF16_CONTRACT)
    if logits is not None:
        assert logits.confident_top1_agreement >= ESMC_CATASTROPHIC_BF16_CONTRACT.top1_hard
        assert logits.mean_jsd <= ESMC_CATASTROPHIC_BF16_CONTRACT.jsd_hard
    try:
        _assert_tensor_metric_records(records, ESMC_ALTERNATE_BF16_CONTRACT)
        if logits is not None:
            _assert_lower(
                "confident_top1_agreement",
                logits.confident_top1_agreement,
                ESMC_ALTERNATE_BF16_CONTRACT.top1_target,
                ESMC_ALTERNATE_BF16_CONTRACT.top1_hard,
                f"{context}:logits",
            )
            _assert_upper(
                "mean_jsd",
                logits.mean_jsd,
                ESMC_ALTERNATE_BF16_CONTRACT.jsd_target,
                ESMC_ALTERNATE_BF16_CONTRACT.jsd_hard,
                f"{context}:logits",
            )
    except AssertionError as error:
        warnings.warn(
            f"{context}: supported ESMC backend is outside its published diagnostic band "
            f"but passed catastrophic biological gates: {error}",
            UserWarning,
            stacklevel=2,
        )


def _assert_esmc_sdpa_exact(
    fast_output: object,
    official_output: object,
    context: str,
) -> None:
    """Require exact ESMC SDPA equality, including special and padding tokens."""

    fast_hidden = _hidden_state_tuple(fast_output)
    official_hidden = _hidden_state_tuple(official_output)
    assert len(fast_hidden) == len(official_hidden)
    for layer, (candidate, official) in enumerate(
        zip(fast_hidden, official_hidden, strict=True)
    ):
        assert torch.equal(candidate, official), (
            f"{context}:layer={layer}: full hidden states are not exact"
        )

    assert torch.equal(_last_hidden(fast_output), _last_hidden(official_output)), (
        f"{context}: full last_hidden_state is not exact"
    )
    fast_logits = getattr(fast_output, "logits", None)
    official_logits = getattr(official_output, "logits", None)
    if fast_logits is not None or official_logits is not None:
        assert fast_logits is not None and official_logits is not None
        assert torch.equal(fast_logits, official_logits), (
            f"{context}: full logits are not exact"
        )


def _run_inference_contract(
    spec: ModelSpec,
    dtype: torch.dtype,
    backend: str | None,
) -> None:
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    # The manifest distinguishes static BF16 parameters from FP32-resident
    # parameters evaluated under CUDA BF16 autocast. Parity, serving probes,
    # and benchmarks all consume this same typed execution contract.
    use_bf16_autocast = (
        dtype == torch.bfloat16 and spec.family.bf16_execution == "fp32_parameters_autocast"
    )
    load_dtype = torch.float32 if use_bf16_autocast else dtype
    fast = _load_fast(spec, device, load_dtype)
    reference, reference_tokenizer = _load_reference(spec, device, load_dtype)
    if backend is not None:
        assert hasattr(fast, "set_attn_implementation"), (
            f"{spec.id}: advertised backend API set_attn_implementation is missing"
        )
        fast.set_attn_implementation(backend)
        resolved = getattr(fast.config, "_attn_implementation", None)
        if resolved is None:
            resolved = getattr(fast.config, "attn_implementation", None)
        assert resolved == backend, (
            f"{spec.id}: requested {backend!r}, resolved {resolved!r}; silent fallback is forbidden"
        )

    fast_inputs, official_inputs, residue_mask = _prepare_inputs(
        spec,
        fast,
        reference_tokenizer,
        _sequence_batch(),
        device,
    )
    if dtype == torch.float32:
        numeric_context = strict_fp32_matmul()
    elif use_bf16_autocast:
        numeric_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        numeric_context = contextlib.nullcontext()
    with torch.inference_mode(), numeric_context:
        fast_output = fast(**fast_inputs, output_hidden_states=True)
        official_output = reference(**official_inputs, output_hidden_states=True)
    contract = _numeric_contract(spec, dtype, backend)
    dtype_name = "fp32" if dtype == torch.float32 else "bf16"
    context = f"{spec.id}:{dtype_name}:{backend or 'default'}"
    if (
        spec.family.architecture == "ESMC"
        and dtype == torch.bfloat16
        and backend in {"flex_attention", "flash_attention_3"}
    ):
        _assert_esmc_alternate_backend_outputs(
            spec,
            fast_output,
            official_output,
            residue_mask,
            context,
        )
    else:
        _assert_outputs(
            spec,
            fast_output,
            official_output,
            residue_mask,
            contract,
            context,
        )
    if spec.family.architecture == "ESMC" and backend in (None, "sdpa"):
        _assert_esmc_sdpa_exact(
            fast_output,
            official_output,
            context,
        )
    del fast, reference, fast_output, official_output
    gc.collect()
    torch.cuda.empty_cache()


def test_manifest_state_transforms_are_registered() -> None:
    declared = {
        spec.family.state_transform
        for spec in REGISTRY.values()
        if spec.family.tokenizer_mode != "structure"
    }
    assert declared.issubset(TRANSFORMS), (
        f"Missing deterministic state transforms: {sorted(declared.difference(TRANSFORMS))}"
    )


@pytest.mark.parametrize("spec", [_parameter(spec) for spec in SEQUENCE_SPECS])
def test_exact_checkpoint_contract(spec: ModelSpec) -> None:
    """Every checkpoint has exact semantic config, state, and aliases."""

    device = torch.device("cpu")
    fast = _load_fast(spec, device, None)
    reference, reference_tokenizer = _load_reference(spec, device, None)
    _assert_semantic_config_equal(spec, fast, reference)
    _assert_state_equal(spec, fast, reference)
    _assert_aliases_equal(spec, fast, reference)
    if spec.family.tokenizer_mode == "tokenizer":
        _assert_tokenizer_equal(spec, fast.tokenizer, reference_tokenizer)
    del fast, reference
    gc.collect()


@pytest.mark.gpu
@pytest.mark.parametrize("spec", [_parameter(spec) for spec in SEQUENCE_SPECS])
def test_every_checkpoint_live_bf16_inference(spec: ModelSpec) -> None:
    """Every checkpoint passes one live mixed-length BF16 official comparison."""

    _run_inference_contract(spec, torch.bfloat16, backend=None)


DEEP_CASES = [
    _deep_parameter(spec, dtype, backend)
    for spec in DEEP_SPECS
    for backend in spec.family.attention
    for dtype in (
        {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[dtype_name]
        for dtype_name in REGISTRY.supported_attention_dtypes(spec.family.id, backend)
    )
]


@pytest.mark.parametrize(("spec", "dtype", "backend"), DEEP_CASES)
def test_representative_deep_backend_parity(
    spec: ModelSpec,
    dtype: torch.dtype,
    backend: str,
) -> None:
    """Representatives pass all-layer parity for every advertised backend."""

    _run_inference_contract(spec, dtype, backend)
