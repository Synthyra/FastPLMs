"""Consume native-container results without importing an official package."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from fastplms.registry import ModelSpec, get_model_registry
from tests.conftest import strict_fp32_matmul
from tests.parity.support.esmc_calibration import (
    ESMC_BOUNDARY_LENGTHS,
    load_esmc_biological_holdout,
)
from tests.parity.support.native_reference import _tensor_digest, _token_result
from tests.parity.support.reference_adapters.dplm2 import (
    DPLM2_3B_GENERATION_LIMITATION,
    DPLM2_150M_OFFICIAL_HEAD_CONTRACT,
)
from tests.parity.test_model_parity import (
    BF16_CONTRACT,
    EDGE_SEQUENCES,
    FP32_CONTRACT,
    _alias_groups,
    _assert_esmc_sdpa_exact,
    _assert_outputs,
    _load_fast,
    _numeric_contract,
    _semantic_config,
)

pytestmark = [pytest.mark.compliance, pytest.mark.gpu, pytest.mark.slow]
REGISTRY = get_model_registry()
SEQUENCE_SPECS = tuple(
    spec for spec in REGISTRY.values() if spec.family.tokenizer_mode != "structure"
)


def _parameter(spec: ModelSpec) -> Any:
    marks = [pytest.mark.large] if spec.size_category == "xlarge" else []
    return pytest.param(spec, id=spec.id, marks=marks)


def _result(spec: ModelSpec) -> tuple[dict[str, Any], Path]:
    root = os.environ.get("FASTPLMS_REFERENCE_RESULTS")
    if not root:
        raise RuntimeError("FASTPLMS_REFERENCE_RESULTS is required for native compliance")
    directory = Path(root) / spec.id
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Native reference result is missing for {spec.id}: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["reference_repo_id"] == spec.official.repo_id
    assert metadata["reference_revision"] == spec.official.revision
    assert metadata["state_transform"] == spec.family.state_transform
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
    else:
        raise ValueError(f"Generation loading is unsupported for {spec.family.id!r}")
    config = config_class.from_pretrained(
        spec.fast.repo_id,
        revision=spec.fast.revision,
    )
    model = model_class.from_pretrained(
        spec.fast.repo_id,
        revision=spec.fast.revision,
        config=config,
        dtype=dtype,
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
        values["logits"] = tensors["output__logits"].to(device)
    return SimpleNamespace(**values)


def _run_native_inference(
    spec: ModelSpec,
    result_dir: Path,
    *,
    precision: str,
    backend: str | None,
    package_source: bool = False,
    tensor_path: Path | None = None,
    context_suffix: str = "native",
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
    if backend is not None:
        fast.set_attn_implementation(backend)
        resolved = getattr(fast.config, "_attn_implementation", None)
        if resolved is None:
            resolved = getattr(fast.config, "attn_implementation", None)
        assert resolved == backend, f"{spec.id}: requested {backend!r}, resolved {resolved!r}"

    inputs = {
        name.removeprefix("input__"): value.to(device)
        for name, value in tensors.items()
        if name.startswith("input__")
    }
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
    _assert_outputs(
        spec,
        candidate,
        official,
        residue_mask,
        contract,
        f"{spec.id}:{precision}:{backend or 'default'}:{context_suffix}",
    )
    if spec.family.architecture == "ESMC" and backend in (None, "sdpa"):
        _assert_esmc_sdpa_exact(
            candidate,
            official,
            f"{spec.id}:{precision}:{backend or 'default'}:{context_suffix}",
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


@pytest.mark.parametrize(
    ("spec", "backend", "kind"),
    [
        pytest.param(
            spec,
            backend,
            kind,
            id=f"{spec.id}-{backend}-{kind}",
            marks=[pytest.mark.large] if spec.size_category == "xlarge" else [],
        )
        for spec in SEQUENCE_SPECS
        if spec.family.id == "esm_plusplus"
        for backend in spec.family.attention
        if "bfloat16" in REGISTRY.supported_attention_dtypes(spec.family.id, backend)
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
    assert batch["seed"] == 42
    expected_biological = {
        case["case_id"]: case for case in load_esmc_biological_holdout()
    }
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
    )


@pytest.mark.parametrize(
    "spec",
    [_parameter(spec) for spec in SEQUENCE_SPECS if spec.id in {"dplm_150m", "dplm2_150m"}],
)
def test_native_dplm_package_source_fp32(spec: ModelSpec) -> None:
    """Current package source matches native DPLM-family FP32 inference."""

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
    """Current package source matches native DPLM BF16 on every supported backend."""

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
    input_tokens = torch.tensor(contract["input_tokens"], device=device)
    torch.manual_seed(int(contract["seed"]))
    torch.cuda.manual_seed_all(int(contract["seed"]))
    with torch.inference_mode(), strict_fp32_matmul():
        generated = fast.generate(input_tokens=input_tokens, **contract["kwargs"])
    if isinstance(generated, dict):
        generated = generated["output_tokens"]
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
