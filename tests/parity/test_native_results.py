"""Consume native-container results without importing an official package."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import tempfile
import warnings
import pytest
import torch
import transformers
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from safetensors.torch import load_file

from fastplms.registry import ModelSpec, get_model_registry
from tests.conftest import strict_fp32_matmul
from tests.parity.support.esmc_calibration import (
    ESMC_BOUNDARY_LENGTHS,
    esmc_calibration_batches,
    load_esmc_biological_holdout,
    validate_esmc_calibration_batch,
)
from tests.parity.support.native_reference import _tensor_digest, _token_result
from tests.parity.support.reference_adapters.biohub_source import (
    BIOHUB_ESM_REVISION,
    BIOHUB_ESM_TREE_SHA256,
    BIOHUB_REFERENCE_SOURCE_NAMES,
    BIOHUB_TRANSFORMERS_REVISION,
    BIOHUB_TRANSFORMERS_TREE_SHA256,
)
from tests.parity.support.reference_adapters.dplm2 import (
    DPLM2_3B_GENERATION_LIMITATION,
    DPLM2_150M_OFFICIAL_HEAD_CONTRACT,
)
from tests.parity.test_model_parity import (
    BF16_CONTRACT,
    EDGE_SEQUENCES,
    ESMC_ALTERNATE_BF16_CONTRACT,
    ESMC_CATASTROPHIC_BF16_CONTRACT,
    LogitsMetrics,
    TensorMetricRecord,
    _alias_groups,
    _assert_esmc_alternate_backend_outputs,
    _assert_esmc_sdpa_exact,
    _assert_outputs,
    _assert_tensor_metric_records,
    _collect_output_metrics,
    _hidden_state_tuple,
    _last_hidden,
    _load_fast,
    _numeric_contract,
    _semantic_config,
)
from tools.remote.biohub_reference_environment import (
    BiohubReferenceEnvironmentError,
    validate_biohub_reference_environment_evidence,
)
from tools.remote.reference_source_attestation import validate_reference_sources_evidence


pytestmark = [pytest.mark.compliance, pytest.mark.gpu, pytest.mark.slow]
REGISTRY = get_model_registry()
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BIOHUB_REFERENCE_LOCK = REPOSITORY_ROOT / "docker/constraints/biohub-reference-lock.json"
SEQUENCE_SPECS = tuple(
    spec for spec in REGISTRY.values() if spec.family.tokenizer_mode != "structure"
)


def _parameter(spec: ModelSpec) -> Any:
    marks = [pytest.mark.large] if spec.size_category == "xlarge" else []
    return pytest.param(spec, id=spec.id, marks=marks)


def _validated_biohub_reference_sources(
    value: object,
) -> dict[str, dict[str, object]]:
    sources = validate_reference_sources_evidence(
        value,
        required_sources=BIOHUB_REFERENCE_SOURCE_NAMES,
    )
    expected = {
        "biohub-esm": {
            "source_revision": BIOHUB_ESM_REVISION,
            "tree_sha256": BIOHUB_ESM_TREE_SHA256,
            "import_name": "esm",
            "import_root": "esm",
            "import_file": "esm/__init__.py",
            "package_version": "3.3.0",
        },
        "biohub-transformers": {
            "source_revision": BIOHUB_TRANSFORMERS_REVISION,
            "tree_sha256": BIOHUB_TRANSFORMERS_TREE_SHA256,
            "import_name": "transformers",
            "import_root": "src/transformers",
            "import_file": "src/transformers/__init__.py",
            "package_version": "4.57.6",
        },
    }
    for source_name, source_expected in expected.items():
        source = sources[source_name]
        for field, expected_value in source_expected.items():
            if source[field] != expected_value:
                raise ValueError(
                    f"{source_name} reference source evidence {field} differs from "
                    f"{expected_value!r}: {source[field]!r}"
                )
    return sources


def _validated_biohub_reference_environment(value: object) -> dict[str, object]:
    try:
        return validate_biohub_reference_environment_evidence(
            value,
            repository_root=REPOSITORY_ROOT,
            contract_path=BIOHUB_REFERENCE_LOCK,
        )
    except BiohubReferenceEnvironmentError as error:
        raise ValueError(f"Biohub reference environment is invalid: {error}") from error


def _result(spec: ModelSpec) -> tuple[dict[str, Any], Path]:
    root = os.environ.get("FASTPLMS_REFERENCE_RESULTS")
    if not root:
        raise RuntimeError("FASTPLMS_REFERENCE_RESULTS is required for native compliance")
    directory = Path(root) / spec.id
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Native reference result is missing for {spec.id}: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Native reference metadata for {spec.id} is not an object")
    expected_identity = {
        "reference_repo_id": spec.official.repo_id,
        "reference_revision": spec.official.revision,
        "state_transform": spec.family.state_transform,
    }
    for name, expected in expected_identity.items():
        if metadata.get(name) != expected:
            raise ValueError(
                f"Native reference {name} for {spec.id} differs from {expected!r}: "
                f"{metadata.get(name)!r}"
            )
    if spec.family.id in {"esm_plusplus", "esm3", "esmfold2"}:
        _validated_biohub_reference_sources(metadata.get("reference_sources"))
        _validated_biohub_reference_environment(metadata.get("reference_environment"))
    return metadata, directory


def _load_package_generation_model(
    spec: ModelSpec,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    """Load pinned mirror weights into the current package implementation."""

    if spec.family.id == "dplm":
        from fastplms.models.dplm.modeling_dplm import DPLMConfig, DPLMForMaskedLM

        config_class = DPLMConfig
        model_class = DPLMForMaskedLM
    elif spec.family.id == "dplm2":
        from fastplms.models.dplm2.modeling_dplm2 import DPLM2Config, DPLM2ForMaskedLM

        config_class = DPLM2Config
        model_class = DPLM2ForMaskedLM
    elif spec.family.id == "ankh":
        from fastplms.models.ankh.modeling_ankh import (
            FastAnkhConfig,
            FastAnkhForConditionalGeneration,
        )

        config_class = FastAnkhConfig
        model_class = FastAnkhForConditionalGeneration
    else:
        raise ValueError(f"Generation loading is unsupported for {spec.family.id!r}")
    config = config_class.from_pretrained(
        spec.fast.repo_id,
        revision=spec.fast.revision,
    )
    model_kwargs: dict[str, Any] = {}
    if spec.family.id == "ankh":
        model_kwargs["attn_implementation"] = "eager"
    model = model_class.from_pretrained(
        spec.fast.repo_id,
        revision=spec.fast.revision,
        config=config,
        dtype=dtype,
        **model_kwargs,
    )
    return model.to(device).eval()


def _normalized_token_result(tokenizer: object, options: dict[str, Any]) -> Any:
    result = _token_result(tokenizer, EDGE_SEQUENCES, options)
    return json.loads(json.dumps(result))


@pytest.mark.parametrize("spec", [_parameter(spec) for spec in SEQUENCE_SPECS])
def test_native_exact_checkpoint_contract(spec: ModelSpec) -> None:
    """Candidate config, weight bytes, aliases, and tokenizer match native output."""

    metadata, _ = _result(spec)
    fast = _load_fast(spec, torch.device("cpu"), None)

    official_config = metadata["semantic_config"]
    candidate_config = _semantic_config(fast)
    assert {name: candidate_config[name] for name in official_config} == official_config

    candidate_state = {
        name: _tensor_digest(tensor) for name, tensor in sorted(fast.state_dict().items())
    }
    assert candidate_state == metadata["state"]["tensors"], (
        f"{spec.id}: candidate state differs from the native official state"
    )
    candidate_aliases = sorted(sorted(group) for group in _alias_groups(fast))
    assert candidate_aliases == metadata["state"]["aliases"]

    if spec.family.tokenizer_mode == "tokenizer":
        contract = metadata["tokenizer"]
        tokenizer = fast.tokenizer
        assert tokenizer.get_vocab() == contract["vocab"]
        assert {
            name: getattr(tokenizer, name, None) for name in contract["special_ids"]
        } == contract["special_ids"]
        for case in contract["behavior"]:
            assert _normalized_token_result(tokenizer, case["options"]) == case["result"]
        artifact_root = os.environ.get("FASTPLMS_CANDIDATE_ARTIFACTS")
        if not artifact_root:
            raise RuntimeError("FASTPLMS_CANDIDATE_ARTIFACTS is required for tokenizer assets")
        repository_name = spec.fast.repo_id.split("/", maxsplit=1)[-1]
        artifact = Path(artifact_root) / repository_name
        for relative_name, expected in metadata["tokenizer_assets"].items():
            path = artifact.joinpath(*Path(relative_name).parts)
            assert path.is_file(), f"{spec.id}: missing tokenizer asset {relative_name}"
            content = path.read_bytes()
            assert len(content) == expected["size"]
            assert hashlib.sha256(content).hexdigest() == expected["sha256"], (
                f"{spec.id}: tokenizer asset bytes differ for {relative_name}"
            )
    del fast
    gc.collect()


def test_native_dplm2_150m_exact_head_contract() -> None:
    """Pinned trained heads remain exact, complete, and independent."""

    metadata, _ = _result(REGISTRY["dplm2_150m"])
    state = metadata["state"]["tensors"]
    observed = {name: state[name] for name in DPLM2_150M_OFFICIAL_HEAD_CONTRACT}
    assert observed == DPLM2_150M_OFFICIAL_HEAD_CONTRACT
    assert metadata["state"]["aliases"] == []


def _official_output(tensors: dict[str, torch.Tensor], device: torch.device) -> object:
    hidden_names = sorted(name for name in tensors if name.startswith("output__hidden_"))
    hidden_states = tuple(tensors[name].to(device) for name in hidden_names)
    values: dict[str, Any] = {
        "hidden_states": hidden_states,
        "last_hidden_state": tensors["output__last_hidden_state"].to(device),
    }
    if "output__logits" in tensors:
        # values['logits']: (..., c)
        values["logits"] = tensors["output__logits"].to(device)
    return SimpleNamespace(**values)


ESMC_DIAGNOSTIC_SCHEMA_VERSION = 3
ESMC_MEASURED_BACKENDS = ("eager", "sdpa", "flex_attention")
ESMC_UNAVAILABLE_BACKENDS = ("flash_attention_2", "flash_attention_3")
_SHA256_HEX_LENGTH = 64
_CANDIDATE_IDENTITY_FIELDS = (
    "fastplms_model_id",
    "fastplms_checkpoint_repo_id",
    "fastplms_checkpoint_revision",
    "fastplms_weights_revision",
    "fastplms_runtime_revision",
    "fastplms_source_tree_sha256",
    "fastplms_runtime_bundle_sha256",
)


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH or value != value.lower():
        raise ValueError(f"{context} must be a 64-character SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{context} must be hexadecimal") from error
    return value


def _require_runtime_revision(
    value: object,
    source_tree_sha256: str,
    context: str,
) -> str:
    """Accept the two immutable identities emitted by the artifact builder."""

    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None:
        return value
    content_addressed = f"source-tree-sha256:{source_tree_sha256}"
    if value == content_addressed:
        return content_addressed
    raise ValueError(f"{context} must be a clean Git revision or the exact source-tree digest")


def _esmc_unavailability_identity(
    backend: str,
    reference_environment: Mapping[str, object],
) -> dict[str, str]:
    runtime = reference_environment.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("ESMC unavailability requires the locked runtime identity")
    operating_system = runtime.get("operating_system")
    architecture = runtime.get("architecture")
    gpu = runtime.get("gpu")
    if (
        not isinstance(operating_system, str)
        or not operating_system.strip()
        or not isinstance(architecture, str)
        or not architecture.strip()
        or not isinstance(gpu, Mapping)
    ):
        raise ValueError("ESMC unavailability runtime platform identity is malformed")
    gpu_name = gpu.get("name")
    capability = gpu.get("capability")
    if (
        not isinstance(gpu_name, str)
        or not gpu_name.strip()
        or not isinstance(capability, list)
        or len(capability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in capability)
    ):
        raise ValueError("ESMC unavailability accelerator identity is malformed")
    platform_identity = f"{operating_system.lower()}/{architecture.lower()}"
    accelerator_identity = f"{gpu_name}/SM{capability[0]}{capability[1]}"
    if backend == "flash_attention_2":
        historical_evidence = "separate_historical_focused_evidence_only"
        reason = (
            f"The locked {platform_identity} {accelerator_identity} release environment "
            "has no validated FlashAttention 2 "
            "kernel. Prior focused execution evidence is historical and is not part of "
            "the current ESMC release distribution."
        )
    elif backend == "flash_attention_3":
        historical_evidence = "none"
        reason = (
            "The manifest-pinned FlashAttention 3 kernel has no validated artifact for "
            f"the locked {platform_identity} {accelerator_identity} release environment."
        )
    else:
        raise ValueError(f"ESMC backend {backend!r} is not a structured unavailable backend")
    return {
        "code": "locked_platform_kernel_unavailable",
        "platform": platform_identity,
        "accelerator": accelerator_identity,
        "dispatch_contract": "fail_closed_without_dispatch",
        "historical_evidence": historical_evidence,
        "reason": reason,
    }


def _optional_package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _cuda_driver_version() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise RuntimeError("ESMC diagnostics require the exact NVIDIA driver version") from error
    versions = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if not versions:
        raise RuntimeError("nvidia-smi did not report an NVIDIA driver version")
    return versions[0]


def _candidate_environment_identity() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("ESMC diagnostic evidence requires a CUDA device")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": str(torch.version.cuda or "unavailable"),
        "cuda_driver": _cuda_driver_version(),
        "gpu": {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(properties.total_memory),
        },
        "packages": {
            distribution: _optional_package_version(distribution)
            for distribution in (
                "fastplms",
                "huggingface-hub",
                "kernels",
                "tokenizers",
                "transformer-engine",
                "transformer-engine-torch",
            )
        },
    }


def _candidate_identity(spec: ModelSpec, model: torch.nn.Module) -> dict[str, object]:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(f"{spec.id}: candidate model has no configuration identity")
    observed = {name: getattr(config, name, None) for name in _CANDIDATE_IDENTITY_FIELDS}
    expected = {
        "fastplms_model_id": spec.id,
        "fastplms_checkpoint_repo_id": spec.artifact_checkpoint.repo_id,
        "fastplms_checkpoint_revision": spec.artifact_checkpoint.revision,
        "fastplms_weights_revision": spec.artifact_checkpoint.revision,
    }
    for name, expected_value in expected.items():
        value = observed[name]
        if value != expected_value:
            raise ValueError(
                f"{spec.id}: candidate {name}={value!r} differs from {expected_value!r}"
            )
    source_tree_sha256 = _require_sha256(
        observed["fastplms_source_tree_sha256"],
        f"{spec.id} candidate fastplms_source_tree_sha256",
    )
    runtime_bundle_sha256 = _require_sha256(
        observed["fastplms_runtime_bundle_sha256"],
        f"{spec.id} candidate fastplms_runtime_bundle_sha256",
    )
    runtime_revision = _require_runtime_revision(
        observed["fastplms_runtime_revision"],
        source_tree_sha256,
        f"{spec.id} candidate runtime revision",
    )
    resolved_commit = getattr(config, "_commit_hash", None)
    if resolved_commit is not None and (
        not isinstance(resolved_commit, str) or not resolved_commit.strip()
    ):
        raise ValueError(f"{spec.id}: candidate resolved commit is invalid")
    return {
        "repo_id": spec.fast.repo_id,
        "manifest_revision": spec.fast.revision,
        "resolved_commit": resolved_commit,
        "checkpoint_repo_id": observed["fastplms_checkpoint_repo_id"],
        "checkpoint_revision": observed["fastplms_checkpoint_revision"],
        "weights_revision": observed["fastplms_weights_revision"],
        "runtime_revision": runtime_revision,
        "source_tree_sha256": source_tree_sha256,
        "runtime_bundle_sha256": runtime_bundle_sha256,
    }


def _reference_identity(
    spec: ModelSpec,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    expected = {
        "reference_repo_id": spec.official.repo_id,
        "reference_revision": spec.official.revision,
        "state_transform": spec.family.state_transform,
    }
    for name, expected_value in expected.items():
        if metadata.get(name) != expected_value:
            raise ValueError(
                f"{spec.id}: native {name}={metadata.get(name)!r} differs from {expected_value!r}"
            )
    environment = metadata.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError(f"{spec.id}: native result omits its environment identity")
    required_environment = {
        "cuda_device",
        "cuda_device_capability",
        "cuda_total_memory",
        "cuda_runtime",
        "packages",
        "python",
        "torch",
    }
    if not required_environment.issubset(environment):
        missing = sorted(required_environment.difference(environment))
        raise ValueError(f"{spec.id}: native environment omits {missing!r}")
    reference_sources = _validated_biohub_reference_sources(metadata.get("reference_sources"))
    reference_environment = _validated_biohub_reference_environment(
        metadata.get("reference_environment")
    )
    return {
        "repo_id": spec.official.repo_id,
        "revision": spec.official.revision,
        "state_transform": spec.family.state_transform,
        "environment": dict(environment),
        "reference_environment": reference_environment,
        "reference_sources": reference_sources,
    }


def _kernel_identity(backend: str) -> dict[str, object]:
    kernel = REGISTRY.attention_kernels.get(backend)
    if kernel is None:
        return {
            "implementation": backend,
            "provider": "torch",
            "torch_version": torch.__version__,
        }
    return {
        "implementation": backend,
        "provider": "huggingface_kernels",
        "repository": kernel.repository,
        "revision": kernel.revision,
        "version": kernel.version,
        "expected_variant": kernel.expected_variant,
        "supported_dtypes": list(kernel.dtypes),
        "kernels_package_version": _optional_package_version("kernels"),
    }


def _metric_payload(
    record: TensorMetricRecord,
    *,
    output: str,
    layer_index: int | None,
) -> dict[str, object]:
    metrics = record.metrics
    return {
        "context": record.context,
        "output": output,
        "layer_index": layer_index,
        "relative_l2": metrics.relative_l2,
        "relative_q999": metrics.relative_q999,
        "residue_cosine_p01": metrics.residue_cosine_p01,
        "pooled_cosine_min": metrics.pooled_cosine_min,
    }


def _structured_metric_payloads(
    output: object,
    records: list[TensorMetricRecord],
) -> list[dict[str, object]]:
    hidden_count = len(_hidden_state_tuple(output))
    has_logits = getattr(output, "logits", None) is not None
    expected_count = hidden_count + 1 + int(has_logits)
    if len(records) != expected_count:
        raise ValueError(
            f"ESMC metric record count {len(records)} differs from expected {expected_count}"
        )
    result = [
        _metric_payload(record, output="hidden_state", layer_index=layer)
        for layer, record in enumerate(records[:hidden_count])
    ]
    result.append(
        _metric_payload(
            records[hidden_count],
            output="last_hidden_state",
            layer_index=None,
        )
    )
    if has_logits:
        result.append(_metric_payload(records[-1], output="logits", layer_index=None))
    return result


def _logits_metric_payload(metrics: LogitsMetrics | None) -> dict[str, float] | None:
    if metrics is None:
        return None
    return {
        "confident_top1_agreement": metrics.confident_top1_agreement,
        "mean_jsd": metrics.mean_jsd,
    }


def _case_identity(case: Mapping[str, object]) -> dict[str, object]:
    return {
        "case_id": case["case_id"],
        "sequence_length": case["sequence_length"],
        "sequence_sha256": case["sequence_sha256"],
        "source": case.get("source"),
        "source_sha256": case.get("source_sha256"),
    }


def _public_panel_identity(panel: Mapping[str, object]) -> dict[str, object]:
    cases = panel.get("cases")
    if not isinstance(cases, list):
        raise ValueError("ESMC panel identity omits its ordered cases")
    return {
        "schema_version": panel["schema_version"],
        "kind": panel["kind"],
        "seed": panel["seed"],
        "definition_sha256": panel["definition_sha256"],
        "cases": [_case_identity(case) for case in cases],
    }


def _slice_output(output: object, index: int) -> SimpleNamespace:
    values: dict[str, object] = {
        "hidden_states": tuple(value[index : index + 1] for value in _hidden_state_tuple(output)),
        "last_hidden_state": _last_hidden(output)[index : index + 1],
    }
    logits = getattr(output, "logits", None)
    if logits is not None:
        # values['logits']: (..., c)
        values["logits"] = logits[index : index + 1]
    return SimpleNamespace(**values)


def _case_metric_distributions(
    spec: ModelSpec,
    candidate: object,
    official: object,
    residue_mask: torch.Tensor,
    panel: Mapping[str, object],
    context: str,
) -> tuple[list[dict[str, object]], list[str]]:
    # residue_mask: (b, l)
    cases = panel.get("cases")
    if not isinstance(cases, list) or residue_mask.ndim != 2:
        raise ValueError("ESMC panel cases and residue mask must be batch aligned")
    if len(cases) != residue_mask.shape[0]:
        raise ValueError(
            f"ESMC panel has {len(cases)} cases for batch size {residue_mask.shape[0]}"
        )
    result: list[dict[str, object]] = []
    violations: list[str] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"ESMC panel case {index} is not an object")
        case_id = str(case["case_id"])
        expected_length = int(case["sequence_length"])
        observed_length = int(residue_mask[index].sum().item())
        if observed_length != expected_length:
            raise ValueError(
                f"{case_id}: residue mask length {observed_length} != {expected_length}"
            )
        candidate_case = _slice_output(candidate, index)
        official_case = _slice_output(official, index)
        # case_mask: (...)
        case_mask = residue_mask[index : index + 1]
        records, logits = _collect_output_metrics(
            spec,
            candidate_case,
            official_case,
            case_mask,
            f"{context}:case={case_id}",
        )
        _assert_esmc_catastrophic_metrics(
            records,
            logits,
            f"{context}:case={case_id}",
        )
        violations.extend(_esmc_published_band_violations(records, logits))
        result.append(
            {
                **_case_identity(case),
                "tensor_metrics": _structured_metric_payloads(candidate_case, records),
                "logits_metrics": _logits_metric_payload(logits),
            }
        )
    return result, violations


def _esmc_published_band_violations(
    records: list[TensorMetricRecord],
    logits: LogitsMetrics | None,
) -> list[str]:
    contract = ESMC_ALTERNATE_BF16_CONTRACT
    violations: list[str] = []
    for record in records:
        metrics = record.metrics
        if metrics.relative_l2 > contract.relative_l2_hard:
            violations.append(
                f"{record.context}:relative_l2={metrics.relative_l2:.6g}>"
                f"{contract.relative_l2_hard:.6g}"
            )
        if metrics.relative_q999 > contract.relative_q999_hard:
            violations.append(
                f"{record.context}:relative_q999={metrics.relative_q999:.6g}>"
                f"{contract.relative_q999_hard:.6g}"
            )
        if metrics.residue_cosine_p01 < contract.residue_cosine_hard:
            violations.append(
                f"{record.context}:residue_cosine_p01={metrics.residue_cosine_p01:.6g}<"
                f"{contract.residue_cosine_hard:.6g}"
            )
        if metrics.pooled_cosine_min < contract.pooled_cosine_hard:
            violations.append(
                f"{record.context}:pooled_cosine_min={metrics.pooled_cosine_min:.6g}<"
                f"{contract.pooled_cosine_hard:.6g}"
            )
    if logits is not None:
        if logits.confident_top1_agreement < contract.top1_hard:
            violations.append(
                "logits:confident_top1_agreement="
                f"{logits.confident_top1_agreement:.6g}<{contract.top1_hard:.6g}"
            )
        if logits.mean_jsd > contract.jsd_hard:
            violations.append(f"logits:mean_jsd={logits.mean_jsd:.6g}>{contract.jsd_hard:.6g}")
    return violations


def _assert_esmc_catastrophic_metrics(
    records: list[TensorMetricRecord],
    logits: LogitsMetrics | None,
    context: str,
) -> None:
    _assert_tensor_metric_records(records, ESMC_CATASTROPHIC_BF16_CONTRACT)
    if logits is None:
        return
    assert logits.confident_top1_agreement >= ESMC_CATASTROPHIC_BF16_CONTRACT.top1_hard, (
        f"{context}: catastrophic top-1 disagreement"
    )
    assert logits.mean_jsd <= ESMC_CATASTROPHIC_BF16_CONTRACT.jsd_hard, (
        f"{context}: catastrophic Jensen-Shannon divergence"
    )


def _release_gate_identity(backend: str) -> dict[str, str]:
    if backend == "sdpa":
        mode = "exact"
    elif backend == "eager":
        mode = "strict_numeric"
    elif backend == "flex_attention":
        mode = "diagnostic_with_catastrophe_gate"
    else:
        raise ValueError(f"Unsupported ESMC diagnostic backend: {backend!r}")
    return {"mode": mode, "status": "passed"}


def _validate_metric_payload(payload: Mapping[str, object]) -> None:
    expected = {
        "context",
        "output",
        "layer_index",
        "relative_l2",
        "relative_q999",
        "residue_cosine_p01",
        "pooled_cosine_min",
    }
    if set(payload) != expected:
        raise ValueError("ESMC tensor-metric fields differ from schema v3")
    context = payload.get("context")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("ESMC tensor metric context is invalid")
    output = payload.get("output")
    layer_index = payload.get("layer_index")
    if output == "hidden_state":
        if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
            raise ValueError("ESMC hidden-state metrics require a nonnegative layer index")
    elif output in {"last_hidden_state", "logits"}:
        if layer_index is not None:
            raise ValueError(f"ESMC {output} metrics must not carry a layer index")
    else:
        raise ValueError(f"Unsupported ESMC metric output: {output!r}")
    for name in (
        "relative_l2",
        "relative_q999",
        "residue_cosine_p01",
        "pooled_cosine_min",
    ):
        value = payload.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"ESMC metric {name} must be a finite number")
    upper_bounds = {
        "relative_l2": ESMC_CATASTROPHIC_BF16_CONTRACT.relative_l2_hard,
        "relative_q999": ESMC_CATASTROPHIC_BF16_CONTRACT.relative_q999_hard,
    }
    lower_bounds = {
        "residue_cosine_p01": ESMC_CATASTROPHIC_BF16_CONTRACT.residue_cosine_hard,
        "pooled_cosine_min": ESMC_CATASTROPHIC_BF16_CONTRACT.pooled_cosine_hard,
    }
    for name, limit in upper_bounds.items():
        if float(payload[name]) < 0 or float(payload[name]) > limit:
            raise ValueError(f"ESMC metric {name} fails the catastrophe gate")
    for name, limit in lower_bounds.items():
        if float(payload[name]) < limit or float(payload[name]) > 1.000001:
            raise ValueError(f"ESMC metric {name} fails the catastrophe gate")


def _validate_logits_metric_payload(payload: object) -> None:
    if payload is None:
        return
    if not isinstance(payload, Mapping) or set(payload) != {
        "confident_top1_agreement",
        "mean_jsd",
    }:
        raise ValueError("ESMC logits-metric fields differ from schema v3")
    for name in ("confident_top1_agreement", "mean_jsd"):
        value = payload.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"ESMC logits metric {name} must be a finite number")
    agreement = float(payload["confident_top1_agreement"])
    mean_jsd = float(payload["mean_jsd"])
    if not ESMC_CATASTROPHIC_BF16_CONTRACT.top1_hard <= agreement <= 1.000001:
        raise ValueError("ESMC top-1 agreement fails the catastrophe gate")
    if not -1e-7 <= mean_jsd <= ESMC_CATASTROPHIC_BF16_CONTRACT.jsd_hard:
        raise ValueError("ESMC Jensen-Shannon divergence fails the catastrophe gate")


def _validate_metric_distribution(
    metrics: object,
    *,
    context: str,
) -> None:
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(f"{context} tensor metrics are missing")
    hidden_layers: list[int] = []
    output_counts = {
        "last_hidden_state": 0,
        "logits": 0,
    }
    for metric in metrics:
        if not isinstance(metric, Mapping):
            raise ValueError(f"{context} tensor metric is not an object")
        _validate_metric_payload(metric)
        output = metric["output"]
        if output == "hidden_state":
            layer_index = metric["layer_index"]
            if not isinstance(layer_index, int):
                raise ValueError(f"{context} hidden-state layer index is invalid")
            hidden_layers.append(layer_index)
        else:
            output_counts[str(output)] += 1
    if not hidden_layers:
        raise ValueError(f"{context} contains no hidden-state layer metrics")
    if hidden_layers != list(range(len(hidden_layers))):
        raise ValueError(f"{context} hidden-state layers are incomplete or unordered")
    if output_counts["last_hidden_state"] != 1:
        raise ValueError(f"{context} must contain one last-hidden-state metric")
    if output_counts["logits"] not in {0, 1}:
        raise ValueError(f"{context} contains duplicate logits metrics")


def _validate_candidate_environment(environment: object) -> None:
    if not isinstance(environment, Mapping) or set(environment) != {
        "python",
        "torch",
        "transformers",
        "cuda_runtime",
        "cuda_driver",
        "gpu",
        "packages",
    }:
        raise ValueError("ESMC candidate environment differs from schema v3")
    for name in ("python", "torch", "transformers", "cuda_runtime", "cuda_driver"):
        value = environment.get(name)
        if not isinstance(value, str) or not value.strip() or value == "unavailable":
            raise ValueError(f"ESMC candidate environment has invalid {name}")

    packages = environment.get("packages")
    expected_packages = {
        "fastplms",
        "huggingface-hub",
        "kernels",
        "tokenizers",
        "transformer-engine",
        "transformer-engine-torch",
    }
    if not isinstance(packages, Mapping) or set(packages) != expected_packages:
        raise ValueError("ESMC candidate package versions differ from schema v3")
    for name, value in packages.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"ESMC candidate package version is invalid for {name}")

    gpu = environment.get("gpu")
    if not isinstance(gpu, Mapping) or set(gpu) != {
        "name",
        "capability",
        "total_memory_bytes",
    }:
        raise ValueError("ESMC GPU identity differs from schema v3")
    capability = gpu.get("capability")
    if (
        not isinstance(gpu.get("name"), str)
        or not str(gpu["name"]).strip()
        or not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in capability
        )
        or isinstance(gpu.get("total_memory_bytes"), bool)
        or not isinstance(gpu.get("total_memory_bytes"), int)
        or int(gpu["total_memory_bytes"]) <= 0
    ):
        raise ValueError("ESMC GPU identity is malformed")
    if environment != _candidate_environment_identity():
        raise ValueError("ESMC candidate environment differs from the active runtime")


def _validate_reference_environment(environment: object) -> None:
    if not isinstance(environment, Mapping):
        raise ValueError("ESMC reference environment is missing")
    required = {
        "cuda_device",
        "cuda_device_capability",
        "cuda_total_memory",
        "cuda_runtime",
        "packages",
        "python",
        "torch",
    }
    if not required.issubset(environment):
        raise ValueError("ESMC reference environment fields are incomplete")
    for name in ("cuda_device", "cuda_runtime", "python", "torch"):
        value = environment.get(name)
        if not isinstance(value, str) or not value.strip() or value == "unavailable":
            raise ValueError(f"ESMC reference environment has invalid {name}")
    capability = environment.get("cuda_device_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in capability
        )
    ):
        raise ValueError("ESMC reference CUDA capability is invalid")
    memory = environment.get("cuda_total_memory")
    if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
        raise ValueError("ESMC reference CUDA memory is invalid")
    packages = environment.get("packages")
    if not isinstance(packages, str) or not packages:
        raise ValueError("ESMC reference package inventory is invalid")
    try:
        package_inventory = json.loads(packages)
    except json.JSONDecodeError as error:
        raise ValueError("ESMC reference package inventory is not JSON") from error
    if not isinstance(package_inventory, Mapping):
        raise ValueError("ESMC reference package inventory is not an object")


def _validate_esmc_environment_binding(
    candidate_environment: Mapping[str, object],
    reference_environment: Mapping[str, object],
    locked_reference_environment: Mapping[str, object],
) -> None:
    candidate_gpu = candidate_environment.get("gpu")
    locked_runtime = locked_reference_environment.get("runtime")
    locked_gpu = locked_runtime.get("gpu") if isinstance(locked_runtime, Mapping) else None
    if not isinstance(candidate_gpu, Mapping) or not isinstance(locked_gpu, Mapping):
        raise ValueError("ESMC candidate/reference GPU binding is malformed")
    candidate_identity = {
        "python": candidate_environment.get("python"),
        "torch": candidate_environment.get("torch"),
        "cuda_runtime": candidate_environment.get("cuda_runtime"),
        "cuda_driver": candidate_environment.get("cuda_driver"),
        "gpu": dict(candidate_gpu),
    }
    dynamic_reference_identity = {
        "python": reference_environment.get("python"),
        "torch": reference_environment.get("torch"),
        "cuda_runtime": reference_environment.get("cuda_runtime"),
        "cuda_driver": candidate_environment.get("cuda_driver"),
        "gpu": {
            "name": reference_environment.get("cuda_device"),
            "capability": reference_environment.get("cuda_device_capability"),
            "total_memory_bytes": reference_environment.get("cuda_total_memory"),
        },
    }
    locked_identity = {
        "python": locked_runtime.get("python_version")
        if isinstance(locked_runtime, Mapping)
        else None,
        "torch": locked_runtime.get("torch") if isinstance(locked_runtime, Mapping) else None,
        "cuda_runtime": locked_runtime.get("cuda_runtime")
        if isinstance(locked_runtime, Mapping)
        else None,
        "cuda_driver": locked_runtime.get("cuda_driver")
        if isinstance(locked_runtime, Mapping)
        else None,
        "gpu": dict(locked_gpu),
    }
    if candidate_identity != dynamic_reference_identity:
        raise ValueError("ESMC candidate and native reference environments differ")
    if candidate_identity != locked_identity:
        raise ValueError("ESMC candidate environment differs from the locked reference runtime")


def _validate_kernel_identity(kernel: object, backend: str) -> None:
    if not isinstance(kernel, Mapping):
        raise ValueError("ESMC kernel identity is malformed")
    expected = _kernel_identity(backend)
    if kernel != expected:
        raise ValueError("ESMC kernel identity differs from the manifest or runtime")


def _expected_panel_identity(kind: object) -> dict[str, object]:
    if not isinstance(kind, str):
        raise ValueError("ESMC panel kind is invalid")
    try:
        batch = next(
            candidate for candidate in esmc_calibration_batches() if candidate["kind"] == kind
        )
    except StopIteration as error:
        raise ValueError(f"Unsupported ESMC calibration panel: {kind!r}") from error
    return _public_panel_identity(validate_esmc_calibration_batch(batch))


def _report_sha256(payload: Mapping[str, object]) -> str:
    digest_payload = dict(payload)
    digest_payload.pop("report_sha256", None)
    encoded = json.dumps(
        digest_payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_esmc_diagnostic_report(
    payload: Mapping[str, object],
    spec: ModelSpec,
    *,
    expected_candidate: Mapping[str, object] | None = None,
) -> None:
    expected = {
        "schema_version",
        "model_id",
        "candidate",
        "reference",
        "record_status",
        "unavailability",
        "configured_backend",
        "effective_backend",
        "dtype",
        "panel",
        "environment",
        "kernel",
        "panel_tensor_metrics",
        "panel_logits_metrics",
        "cases",
        "published_band_violations",
        "catastrophic_gate",
        "release_gate",
        "report_sha256",
    }
    if set(payload) != expected or payload.get("schema_version") != ESMC_DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("ESMC diagnostic fields differ from schema v3")
    if payload.get("model_id") != spec.id or payload.get("dtype") != "bfloat16":
        raise ValueError("ESMC diagnostic model or dtype identity differs from the request")

    backend = payload.get("configured_backend")
    record_status = payload.get("record_status")
    if backend not in (*ESMC_MEASURED_BACKENDS, *ESMC_UNAVAILABLE_BACKENDS):
        raise ValueError("ESMC diagnostic backend identity is invalid")
    if record_status not in {"measured", "unavailable"}:
        raise ValueError("ESMC diagnostic record status is invalid")

    candidate = payload.get("candidate")
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "repo_id",
        "manifest_revision",
        "resolved_commit",
        "checkpoint_repo_id",
        "checkpoint_revision",
        "weights_revision",
        "runtime_revision",
        "source_tree_sha256",
        "runtime_bundle_sha256",
    }:
        raise ValueError("ESMC candidate identity differs from schema v3")
    if (
        candidate.get("repo_id") != spec.fast.repo_id
        or candidate.get("manifest_revision") != spec.fast.revision
    ):
        raise ValueError("ESMC candidate repository identity differs from the manifest")
    expected_checkpoint = {
        "checkpoint_repo_id": spec.artifact_checkpoint.repo_id,
        "checkpoint_revision": spec.artifact_checkpoint.revision,
        "weights_revision": spec.artifact_checkpoint.revision,
    }
    for name, expected_value in expected_checkpoint.items():
        value = candidate.get(name)
        if value != expected_value:
            raise ValueError(f"ESMC candidate {name} differs from the manifest")
    resolved_commit = candidate.get("resolved_commit")
    if resolved_commit is not None and resolved_commit != spec.fast.revision:
        raise ValueError("ESMC candidate resolved commit differs from the manifest")
    source_tree_sha256 = candidate.get("source_tree_sha256")
    runtime_bundle_sha256 = candidate.get("runtime_bundle_sha256")
    runtime_revision = candidate.get("runtime_revision")
    source_digest = _require_sha256(
        source_tree_sha256,
        "ESMC candidate source tree",
    )
    _require_sha256(runtime_bundle_sha256, "ESMC candidate runtime bundle")
    _require_runtime_revision(
        runtime_revision,
        source_digest,
        "ESMC candidate runtime revision",
    )
    if expected_candidate is not None and candidate != expected_candidate:
        raise ValueError("ESMC candidate identity differs from the validated artifact identity")

    reference = payload.get("reference")
    if not isinstance(reference, Mapping) or set(reference) != {
        "repo_id",
        "revision",
        "state_transform",
        "environment",
        "reference_environment",
        "reference_sources",
    }:
        raise ValueError("ESMC reference identity differs from schema v3")
    if (
        reference.get("repo_id") != spec.official.repo_id
        or reference.get("revision") != spec.official.revision
        or reference.get("state_transform") != spec.family.state_transform
    ):
        raise ValueError("ESMC reference identity differs from the manifest")
    dynamic_reference_environment = reference.get("environment")
    _validate_reference_environment(dynamic_reference_environment)
    locked_reference_environment = _validated_biohub_reference_environment(
        reference.get("reference_environment")
    )
    _validated_biohub_reference_sources(reference.get("reference_sources"))

    candidate_environment = payload.get("environment")
    _validate_candidate_environment(candidate_environment)
    if not isinstance(candidate_environment, Mapping) or not isinstance(
        dynamic_reference_environment, Mapping
    ):
        raise ValueError("ESMC candidate/reference environment binding is malformed")
    _validate_esmc_environment_binding(
        candidate_environment,
        dynamic_reference_environment,
        locked_reference_environment,
    )

    _validate_kernel_identity(payload.get("kernel"), str(backend))

    panel = payload.get("panel")
    if not isinstance(panel, Mapping) or set(panel) != {
        "schema_version",
        "kind",
        "seed",
        "definition_sha256",
        "cases",
    }:
        raise ValueError("ESMC panel identity differs from schema v3")
    if panel != _expected_panel_identity(panel.get("kind")):
        raise ValueError("ESMC panel identity differs from the immutable definition")
    panel_cases = panel.get("cases")
    cases = payload.get("cases")
    if (
        not isinstance(panel_cases, list)
        or not isinstance(cases, list)
        or len(panel_cases) != len(cases)
    ):
        raise ValueError("ESMC panel and metric cases are not aligned")

    identity_fields = {
        "case_id",
        "sequence_length",
        "sequence_sha256",
        "source",
        "source_sha256",
    }
    violations = payload.get("published_band_violations")
    if not isinstance(violations, list) or any(
        not isinstance(value, str) or not value.strip() for value in violations
    ):
        raise ValueError("ESMC published-band violations must be a string list")
    if record_status == "unavailable":
        if backend not in ESMC_UNAVAILABLE_BACKENDS:
            raise ValueError("Only locked Flash backends may use unavailable records")
        if payload.get("effective_backend") is not None:
            raise ValueError("Unavailable ESMC records must not claim effective dispatch")
        if payload.get("unavailability") != _esmc_unavailability_identity(
            str(backend), locked_reference_environment
        ):
            raise ValueError("ESMC structured unavailability identity is invalid")
        if payload.get("catastrophic_gate") != "not_run":
            raise ValueError("Unavailable ESMC records must mark the catastrophe gate not run")
        if payload.get("release_gate") != {"mode": "availability", "status": "unavailable"}:
            raise ValueError("Unavailable ESMC release-gate identity is invalid")
        if (
            payload.get("panel_tensor_metrics") is not None
            or payload.get("panel_logits_metrics") is not None
            or violations
        ):
            raise ValueError("Unavailable ESMC records must not contain numerical measurements")
        if cases != panel_cases:
            raise ValueError("Unavailable ESMC cases must be immutable panel identities only")
    else:
        if backend not in ESMC_MEASURED_BACKENDS:
            raise ValueError(
                "Current GH200 release measurements are limited to eager, SDPA, and Flex"
            )
        if payload.get("effective_backend") != backend:
            raise ValueError("ESMC diagnostic backend identity indicates fallback")
        if payload.get("unavailability") is not None:
            raise ValueError("Measured ESMC records must not carry unavailability metadata")
        if payload.get("catastrophic_gate") != "passed":
            raise ValueError("Measured ESMC records require a passed catastrophe gate")
        if payload.get("release_gate") != _release_gate_identity(str(backend)):
            raise ValueError("ESMC diagnostic release-gate identity is invalid")

        metrics = payload.get("panel_tensor_metrics")
        _validate_metric_distribution(metrics, context="ESMC panel")
        panel_logits_metrics = payload.get("panel_logits_metrics")
        _validate_logits_metric_payload(panel_logits_metrics)
        if not isinstance(metrics, list):
            raise ValueError("ESMC panel tensor metrics must be an ordered list")
        panel_layout = [
            (metric["output"], metric["layer_index"])
            for metric in metrics
            if isinstance(metric, Mapping)
        ]
        for panel_case, case in zip(panel_cases, cases, strict=True):
            if not isinstance(panel_case, Mapping) or set(panel_case) != identity_fields:
                raise ValueError("ESMC panel case identity differs from schema v3")
            if not isinstance(case, Mapping) or set(case) != identity_fields | {
                "tensor_metrics",
                "logits_metrics",
            }:
                raise ValueError("ESMC case distribution differs from schema v3")
            if any(case.get(name) != panel_case.get(name) for name in identity_fields):
                raise ValueError("ESMC case distribution is misaligned with the panel")
            _require_sha256(case.get("sequence_sha256"), "ESMC case sequence")
            source_sha256 = case.get("source_sha256")
            if source_sha256 is not None:
                _require_sha256(source_sha256, "ESMC case source")
            case_metrics = case.get("tensor_metrics")
            _validate_metric_distribution(
                case_metrics,
                context=f"ESMC case {case.get('case_id')}",
            )
            if not isinstance(case_metrics, list):
                raise ValueError("ESMC case tensor metrics must be an ordered list")
            case_layout = [
                (metric["output"], metric["layer_index"])
                for metric in case_metrics
                if isinstance(metric, Mapping)
            ]
            if case_layout != panel_layout:
                raise ValueError("ESMC case metric layout differs from the panel")
            case_logits_metrics = case.get("logits_metrics")
            _validate_logits_metric_payload(case_logits_metrics)
            if (case_logits_metrics is None) != (panel_logits_metrics is None):
                raise ValueError("ESMC case logits metrics differ from the panel")
    report_digest = _require_sha256(payload.get("report_sha256"), "ESMC report")
    if report_digest != _report_sha256(payload):
        raise ValueError("ESMC report digest does not match its payload")


def _build_esmc_diagnostic_report(
    spec: ModelSpec,
    candidate: object,
    official: object,
    residue_mask: torch.Tensor,
    *,
    backend: str,
    effective_backend: str,
    context: str,
    calibration_batch: Mapping[str, object],
    model: torch.nn.Module,
    reference_metadata: Mapping[str, object],
) -> dict[str, object]:
    # residue_mask: (b, l)
    panel = validate_esmc_calibration_batch(calibration_batch)
    records, logits = _collect_output_metrics(
        spec,
        candidate,
        official,
        residue_mask,
        context,
    )
    _assert_esmc_catastrophic_metrics(records, logits, context)
    violations = _esmc_published_band_violations(records, logits)
    case_metrics, case_violations = _case_metric_distributions(
        spec,
        candidate,
        official,
        residue_mask,
        panel,
        context,
    )
    violations.extend(case_violations)
    candidate_identity = _candidate_identity(spec, model)
    reference_identity = _reference_identity(spec, reference_metadata)
    candidate_environment = _candidate_environment_identity()
    payload: dict[str, object] = {
        "schema_version": ESMC_DIAGNOSTIC_SCHEMA_VERSION,
        "model_id": spec.id,
        "candidate": candidate_identity,
        "reference": reference_identity,
        "record_status": "measured",
        "unavailability": None,
        "configured_backend": backend,
        "effective_backend": effective_backend,
        "dtype": "bfloat16",
        "panel": _public_panel_identity(panel),
        "environment": candidate_environment,
        "kernel": _kernel_identity(backend),
        "panel_tensor_metrics": _structured_metric_payloads(candidate, records),
        "panel_logits_metrics": _logits_metric_payload(logits),
        "cases": case_metrics,
        "published_band_violations": violations,
        "catastrophic_gate": "passed",
        "release_gate": _release_gate_identity(backend),
    }
    payload["report_sha256"] = _report_sha256(payload)
    _validate_esmc_diagnostic_report(
        payload,
        spec,
        expected_candidate=candidate_identity,
    )
    return payload


def _build_esmc_unavailable_report(
    spec: ModelSpec,
    *,
    backend: str,
    calibration_batch: Mapping[str, object],
    model: torch.nn.Module,
    reference_metadata: Mapping[str, object],
) -> dict[str, object]:
    if backend not in ESMC_UNAVAILABLE_BACKENDS:
        raise ValueError(f"ESMC backend {backend!r} is not unavailable on the locked target")
    panel = _public_panel_identity(validate_esmc_calibration_batch(calibration_batch))
    panel_cases = panel["cases"]
    if not isinstance(panel_cases, list):
        raise ValueError("ESMC unavailable report panel cases are not an ordered list")
    candidate_identity = _candidate_identity(spec, model)
    reference_identity = _reference_identity(spec, reference_metadata)
    locked_reference_environment = reference_identity["reference_environment"]
    if not isinstance(locked_reference_environment, Mapping):
        raise ValueError("ESMC unavailable report omits its locked reference environment")
    candidate_environment = _candidate_environment_identity()
    payload: dict[str, object] = {
        "schema_version": ESMC_DIAGNOSTIC_SCHEMA_VERSION,
        "model_id": spec.id,
        "candidate": candidate_identity,
        "reference": reference_identity,
        "record_status": "unavailable",
        "unavailability": _esmc_unavailability_identity(backend, locked_reference_environment),
        "configured_backend": backend,
        "effective_backend": None,
        "dtype": "bfloat16",
        "panel": panel,
        "environment": candidate_environment,
        "kernel": _kernel_identity(backend),
        "panel_tensor_metrics": None,
        "panel_logits_metrics": None,
        "cases": [dict(case) for case in panel_cases],
        "published_band_violations": [],
        "catastrophic_gate": "not_run",
        "release_gate": {"mode": "availability", "status": "unavailable"},
    }
    payload["report_sha256"] = _report_sha256(payload)
    _validate_esmc_diagnostic_report(
        payload,
        spec,
        expected_candidate=candidate_identity,
    )
    return payload


def _write_esmc_diagnostic_report(
    spec: ModelSpec,
    payload: Mapping[str, object],
) -> Path:
    _validate_esmc_diagnostic_report(payload, spec)
    report_root = Path(os.environ.get("FASTPLMS_DIAGNOSTIC_REPORTS", "artifacts/diagnostics/esmc"))
    report_root.mkdir(parents=True, exist_ok=True)
    panel = payload["panel"]
    assert isinstance(panel, Mapping)
    report_name = f"{spec.id}-{payload['configured_backend']}-{panel['kind']}.json"
    report_path = report_root / report_name
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if report_path.exists():
        if report_path.read_text(encoding="utf-8") == encoded:
            return report_path
        raise RuntimeError(f"Refusing to replace different ESMC evidence: {report_path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=report_root,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, report_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return report_path


def _record_esmc_diagnostic(
    spec: ModelSpec,
    candidate: object,
    official: object,
    residue_mask: torch.Tensor,
    *,
    backend: str,
    effective_backend: str,
    context: str,
    calibration_batch: Mapping[str, object],
    model: torch.nn.Module,
    reference_metadata: Mapping[str, object],
    warn_on_published_band: bool,
) -> dict[str, object]:
    # residue_mask: (b, l)
    payload = _build_esmc_diagnostic_report(
        spec,
        candidate,
        official,
        residue_mask,
        backend=backend,
        effective_backend=effective_backend,
        context=context,
        calibration_batch=calibration_batch,
        model=model,
        reference_metadata=reference_metadata,
    )
    report_path = _write_esmc_diagnostic_report(spec, payload)
    violations = payload["published_band_violations"]
    assert isinstance(violations, list)
    if warn_on_published_band and violations:
        warnings.warn(
            f"{spec.id} configured backend={backend}, effective backend={effective_backend}: "
            f"{len(violations)} diagnostic metric(s) are outside the published ESMC "
            f"backend bands; catastrophic biological gates passed. Report: {report_path}",
            UserWarning,
            stacklevel=2,
        )
    return payload


def _assert_and_record_esmc_diagnostic(
    spec: ModelSpec,
    candidate: object,
    official: object,
    residue_mask: torch.Tensor,
    *,
    backend: str,
    effective_backend: str,
    context: str,
    calibration_batch: Mapping[str, object],
    model: torch.nn.Module,
    reference_metadata: Mapping[str, object],
) -> dict[str, object]:
    # residue_mask: (b, l)
    return _record_esmc_diagnostic(
        spec,
        candidate,
        official,
        residue_mask,
        backend=backend,
        effective_backend=effective_backend,
        context=context,
        calibration_batch=calibration_batch,
        model=model,
        reference_metadata=reference_metadata,
        warn_on_published_band=True,
    )


def _run_native_inference(
    spec: ModelSpec,
    result_dir: Path,
    *,
    precision: str,
    backend: str | None,
    package_source: bool = False,
    tensor_path: Path | None = None,
    context_suffix: str = "native",
    calibration_batch: Mapping[str, object] | None = None,
    reference_metadata: Mapping[str, object] | None = None,
) -> None:
    device = torch.device("cuda")
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    tensors = load_file(tensor_path or result_dir / f"{precision}.safetensors", device="cpu")
    use_bf16_autocast = (
        dtype == torch.bfloat16 and spec.family.bf16_execution == "fp32_parameters_autocast"
    )
    model_dtype = torch.float32 if use_bf16_autocast else dtype
    if package_source:
        fast = _load_package_generation_model(spec, device, model_dtype)
    else:
        fast = _load_fast(spec, device, model_dtype)
    effective_backend: str | None = backend
    if backend is not None:
        fast.set_attn_implementation(backend)
        effective_backend = getattr(fast.config, "_attn_implementation", None)
        if effective_backend is None:
            effective_backend = getattr(fast.config, "attn_implementation", None)
        assert effective_backend == backend, (
            f"{spec.id}: requested {backend!r}, resolved {effective_backend!r}"
        )

    inputs = {
        name.removeprefix("input__"): value.to(device)
        for name, value in tensors.items()
        if name.startswith("input__")
    }
    # residue_mask: (b, l)
    residue_mask = tensors["residue_mask"].to(device).bool()
    if dtype == torch.float32:
        numeric_context = strict_fp32_matmul()
    elif use_bf16_autocast:
        numeric_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        numeric_context = contextlib.nullcontext()
    with torch.inference_mode(), numeric_context:
        candidate = fast(**inputs, output_hidden_states=True)
    official = _official_output(tensors, device)
    contract = _numeric_contract(spec, dtype, backend)
    context = f"{spec.id}:{precision}:{backend or 'default'}:{context_suffix}"
    if (
        spec.family.architecture == "ESMC"
        and dtype == torch.bfloat16
        and backend == "flex_attention"
    ):
        assert backend is not None
        if calibration_batch is None:
            _assert_esmc_alternate_backend_outputs(
                spec,
                candidate,
                official,
                residue_mask,
                context,
            )
        else:
            if reference_metadata is None:
                raise ValueError("ESMC calibration requires native reference metadata")
            _assert_and_record_esmc_diagnostic(
                spec,
                candidate,
                official,
                residue_mask,
                backend=backend,
                effective_backend=str(effective_backend),
                context=context,
                calibration_batch=calibration_batch,
                model=fast,
                reference_metadata=reference_metadata,
            )
    else:
        _assert_outputs(
            spec,
            candidate,
            official,
            residue_mask,
            contract,
            context,
        )
    if spec.family.architecture == "ESMC" and backend in (None, "sdpa"):
        _assert_esmc_sdpa_exact(
            candidate,
            official,
            context,
        )
    if (
        spec.family.architecture == "ESMC"
        and dtype == torch.bfloat16
        and backend in {"eager", "sdpa"}
        and calibration_batch is not None
    ):
        if reference_metadata is None:
            raise ValueError("ESMC calibration requires native reference metadata")
        _record_esmc_diagnostic(
            spec,
            candidate,
            official,
            residue_mask,
            backend=backend,
            effective_backend=str(effective_backend),
            context=context,
            calibration_batch=calibration_batch,
            model=fast,
            reference_metadata=reference_metadata,
            warn_on_published_band=False,
        )
        if backend == "sdpa":
            for unavailable_backend in ESMC_UNAVAILABLE_BACKENDS:
                _write_esmc_diagnostic_report(
                    spec,
                    _build_esmc_unavailable_report(
                        spec,
                        backend=unavailable_backend,
                        calibration_batch=calibration_batch,
                        model=fast,
                        reference_metadata=reference_metadata,
                    ),
                )
    del fast, candidate, official, tensors
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("spec", [_parameter(spec) for spec in SEQUENCE_SPECS])
def test_native_every_checkpoint_bf16_inference(spec: ModelSpec) -> None:
    """Every checkpoint matches one native mixed-length BF16 result."""

    _, result_dir = _result(spec)
    _run_native_inference(spec, result_dir, precision="bf16", backend=None)


NATIVE_REPRESENTATIVE_CASES = [
    pytest.param(
        spec,
        precision,
        backend,
        id=f"{spec.id}-{precision}-{backend}",
        marks=[pytest.mark.large] if spec.size_category == "xlarge" else [],
    )
    for spec in SEQUENCE_SPECS
    if spec.is_deep_reference
    for backend in spec.family.attention
    if not (spec.family.id == "esm_plusplus" and backend in ESMC_UNAVAILABLE_BACKENDS)
    for precision, dtype_name in (("fp32", "float32"), ("bf16", "bfloat16"))
    if dtype_name in REGISTRY.supported_attention_dtypes(spec.family.id, backend)
]


@pytest.mark.parametrize(("spec", "precision", "backend"), NATIVE_REPRESENTATIVE_CASES)
def test_native_representatives_all_backends(
    spec: ModelSpec,
    precision: str,
    backend: str,
) -> None:
    """Each representative matches native FP32 and BF16 for advertised backends."""

    _, result_dir = _result(spec)
    _run_native_inference(spec, result_dir, precision=precision, backend=backend)


def _esmc_calibration_marks(spec: ModelSpec, backend: str, kind: str) -> list[Any]:
    del backend, kind
    return [pytest.mark.large] if spec.size_category == "xlarge" else []


@pytest.mark.parametrize(
    ("spec", "backend", "kind"),
    [
        pytest.param(
            spec,
            backend,
            kind,
            id=f"{spec.id}-{backend}-{kind}",
            marks=_esmc_calibration_marks(spec, backend, kind),
        )
        for spec in SEQUENCE_SPECS
        if spec.family.id == "esm_plusplus"
        for backend in ESMC_MEASURED_BACKENDS
        if backend in spec.family.attention
        and "bfloat16" in REGISTRY.supported_attention_dtypes(spec.family.id, backend)
        for kind in ("generated_kernel_boundary", "real_biological_holdout")
    ],
)
def test_esmc_bf16_calibration_and_biological_holdout(
    spec: ModelSpec,
    backend: str,
    kind: str,
) -> None:
    """Calibrate BF16 parity on pinned shape and biological panels."""

    metadata, result_dir = _result(spec)
    batches = metadata.get("calibration_batches")
    assert isinstance(batches, list)
    batch = next(item for item in batches if item["kind"] == kind)
    assert isinstance(batch, Mapping)
    validate_esmc_calibration_batch(batch)
    assert batch["seed"] == 42
    expected_biological = {case["case_id"]: case for case in load_esmc_biological_holdout()}
    if kind == "generated_kernel_boundary":
        observed_lengths = tuple(case["sequence_length"] for case in batch["cases"])
        assert observed_lengths == ESMC_BOUNDARY_LENGTHS
    else:
        assert tuple(case["case_id"] for case in batch["cases"]) == tuple(expected_biological)
    for case in batch["cases"]:
        sequence = case["sequence"]
        assert len(sequence) == case["sequence_length"]
        assert hashlib.sha256(sequence.encode("ascii")).hexdigest() == case["sequence_sha256"]
        if kind == "real_biological_holdout":
            expected = expected_biological[case["case_id"]]
            assert {
                name: case[name]
                for name in (
                    "case_id",
                    "sequence",
                    "sequence_sha256",
                    "source",
                    "source_sha256",
                )
            } == expected
    _run_native_inference(
        spec,
        result_dir,
        precision="bf16",
        backend=backend,
        tensor_path=result_dir / "calibration" / f"{kind}.safetensors",
        context_suffix=kind,
        calibration_batch=batch,
        reference_metadata=metadata,
    )


@pytest.mark.parametrize(
    "spec",
    [_parameter(spec) for spec in SEQUENCE_SPECS if spec.id in {"dplm_150m", "dplm2_150m"}],
)
def test_native_dplm_package_source_fp32(spec: ModelSpec) -> None:
    """Current repository source matches native DPLM-family FP32 inference."""

    _, result_dir = _result(spec)
    _run_native_inference(
        spec,
        result_dir,
        precision="fp32",
        backend=None,
        package_source=True,
    )


@pytest.mark.parametrize(
    ("spec", "backend"),
    [
        pytest.param(spec, backend, id=f"{spec.id}-{backend}")
        for spec in SEQUENCE_SPECS
        if spec.id in {"dplm_150m", "dplm2_150m"}
        for backend in spec.family.attention
        if "bfloat16" in REGISTRY.supported_attention_dtypes(spec.family.id, backend)
    ],
)
def test_native_dplm_package_source_bf16(spec: ModelSpec, backend: str) -> None:
    """Current repository source matches native DPLM BF16 on every supported backend."""

    _, result_dir = _result(spec)
    _run_native_inference(
        spec,
        result_dir,
        precision="bf16",
        backend=backend,
        package_source=True,
    )


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(REGISTRY["dplm_150m"], id="dplm_150m"),
        pytest.param(REGISTRY["dplm2_150m"], id="dplm2_150m"),
    ],
)
def test_native_dplm_sdpa_uses_fp32_storage_and_meets_every_hidden_target(
    spec: ModelSpec,
) -> None:
    """Each manifest-declared DPLM AMP path passes every hidden-state target."""

    assert spec.family.bf16_execution == "fp32_parameters_autocast"
    _, result_dir = _result(spec)
    device = torch.device("cuda")
    tensors = load_file(result_dir / "bf16.safetensors", device="cpu")
    model = _load_package_generation_model(spec, device, torch.float32)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    model.set_attn_implementation("sdpa")
    inputs = {
        name.removeprefix("input__"): value.to(device)
        for name, value in tensors.items()
        if name.startswith("input__")
    }
    # residue_mask: (b, l)
    residue_mask = tensors["residue_mask"].to(device).bool()

    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        candidate = model(**inputs, output_hidden_states=True)
    official = _official_output(tensors, device)
    assert len(candidate.hidden_states) == len(official.hidden_states) == 31
    _assert_outputs(
        spec,
        candidate,
        official,
        residue_mask,
        BF16_CONTRACT,
        f"{spec.id}:bf16-autocast:sdpa:fp32-storage",
    )
    del model, candidate, official, tensors
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "spec",
    [
        _parameter(spec)
        for spec in SEQUENCE_SPECS
        if spec.family.id in {"dplm", "dplm2"} and spec.id != "dplm2_3b"
    ],
)
def test_native_dplm_generation(spec: ModelSpec) -> None:
    """DPLM-family output tokens match the isolated official public sampler."""

    metadata, _ = _result(spec)
    contract = metadata.get("generation")
    assert isinstance(contract, dict), f"{spec.id}: native result omits generation"
    device = torch.device("cuda")
    fast = _load_package_generation_model(spec, device)
    # input_tokens: (...)
    input_tokens = torch.tensor(contract["input_tokens"], device=device)
    torch.manual_seed(int(contract["seed"]))
    torch.cuda.manual_seed_all(int(contract["seed"]))
    with torch.inference_mode(), strict_fp32_matmul():
        generated = fast.generate(input_tokens=input_tokens, **contract["kwargs"])
    if isinstance(generated, dict):
        generated = generated["output_tokens"]
    # expected: (...)
    expected = torch.tensor(contract["output_tokens"], device=device)
    assert torch.equal(generated, expected), f"{spec.id}: generated tokens differ"
    del fast, generated
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "spec",
    [_parameter(spec) for spec in SEQUENCE_SPECS if spec.family.id == "ankh"],
)
def test_native_ankh_explicit_decoder_prompt_generation(spec: ModelSpec) -> None:
    """ANKH tokens match native T5 generation from the recorded decoder prompt."""

    metadata, _ = _result(spec)
    contract = metadata.get("generation")
    assert isinstance(contract, dict), f"{spec.id}: native result omits generation"
    assert contract["interface"] == "T5ForConditionalGeneration.generate"
    assert contract["decoder_prompt_contract"] == "explicit-task-prompt"
    device = torch.device("cuda")
    fast = _load_package_generation_model(spec, device)
    # input_ids: (...)
    input_ids = torch.tensor(contract["input_ids"], device=device)
    # attention_mask: (...)
    attention_mask = torch.tensor(contract["attention_mask"], device=device)
    # decoder_input_ids: (...)
    decoder_input_ids = torch.tensor(contract["decoder_input_ids"], device=device)
    # decoder_attention_mask: (...)
    decoder_attention_mask = torch.tensor(
        contract["decoder_attention_mask"],
        device=device,
    )
    assert _tensor_digest(decoder_input_ids)["sha256"] == (contract["decoder_input_fingerprint"])
    torch.manual_seed(int(contract["seed"]))
    torch.cuda.manual_seed_all(int(contract["seed"]))
    with torch.inference_mode(), strict_fp32_matmul():
        generated = fast.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            **contract["kwargs"],
        )
    # expected: (...)
    expected = torch.tensor(contract["output_tokens"], device=device)
    assert torch.equal(generated, expected), f"{spec.id}: generated tokens differ"
    del fast, generated
    gc.collect()
    torch.cuda.empty_cache()


def test_native_dplm2_3b_official_generation_limitation() -> None:
    """The pinned public 3B sampler is unavailable, not parity-passing."""

    spec = REGISTRY["dplm2_3b"]
    metadata, _ = _result(spec)
    assert "generation" not in metadata
    assert metadata.get("generation_limitation") == DPLM2_3B_GENERATION_LIMITATION
