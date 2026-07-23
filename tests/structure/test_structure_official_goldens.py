"""Candidate-only, hardware-bound regression against immutable structure goldens.

Unlike compliance tests, this module never imports or executes an official
implementation. It consumes the compact, hash-verified outputs already checked
into ``tests/goldens`` and is intended for the conditional structure GPU lane.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from fastplms.registry import ModelSpec, get_model_registry
from tests.structure import test_esmfold2_folding_compliance as esmfold2_metrics
from tests.structure import test_esmfold_folding_compliance as esmfold_metrics
from tests.structure.support import esmfold2_bundle, esmfold_bundle
from tests.structure.support.hardware import assert_recorded_hopper_device_matches
from tools.goldens import validate_golden_bundle

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = get_model_registry()


def _golden(spec: ModelSpec) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    declaration = spec.official_golden
    if declaration is None:
        raise AssertionError(
            f"{spec.id}: no measured immutable structure golden is declared; "
            "candidate-only structure validation must fail closed."
        )
    metadata_path = ROOT / declaration.metadata.path
    tensors_path = ROOT / declaration.tensors.path
    validate_golden_bundle(
        spec,
        REGISTRY,
        metadata_path=metadata_path,
        tensors_path=tensors_path,
        declaration=declaration,
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return load_file(tensors_path, device="cpu"), metadata


def _release_parameter(spec: ModelSpec) -> object:
    return pytest.param(spec, id=spec.id, marks=pytest.mark.large)


def _assert_golden_device_matches_current(metadata: dict[str, object]) -> None:
    environment = metadata["environment"]
    assert isinstance(environment, dict)
    recorded = environment["details"]
    assert isinstance(recorded, dict)
    properties = torch.cuda.get_device_properties(0)
    assert_recorded_hopper_device_matches(
        {
            "cuda_device": properties.name,
            "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_total_memory": int(properties.total_memory),
        },
        recorded,
    )


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.checkpoint
@pytest.mark.network
def test_esmfold_candidate_matches_checked_structure_golden(tmp_path: Path) -> None:
    spec = REGISTRY[esmfold_bundle.model_id]
    golden, metadata = _golden(spec)
    _assert_golden_device_matches_current(metadata)
    request_path = esmfold_bundle.prepare_request(tmp_path)
    request = esmfold_bundle.load_request(request_path)
    assert metadata["input_fingerprint"] == request["request_sha256"]

    model = esmfold_bundle._load_candidate_model(request, torch.device("cuda"))
    candidate = esmfold_bundle._run_infer(model, request, "bf16")

    for name in esmfold_bundle._exact_outputs:
        assert torch.equal(candidate[f"output__{name}"], golden[f"output__{name}"]), name
    esmfold_metrics._assert_valid_bundle(golden, context="checked Meta ESMFold golden")
    esmfold_metrics._assert_valid_bundle(candidate, context="FastPLMs ESMFold candidate")
    for name, value in esmfold_metrics._logit_metrics(candidate, golden).items():
        assert value <= esmfold_metrics.relative_l2_hard_limits["bf16"], (
            f"{name} relative L2 {value:.6g} exceeds the measured BF16 hard limit"
        )
    structure = esmfold_metrics._structure_metrics(candidate, golden)
    for name, hard_limit in esmfold_metrics.structure_hard_limits.items():
        if name == "lddt_ca":
            assert structure[name] >= hard_limit
        else:
            assert structure[name] <= hard_limit

    del model, candidate, golden
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.checkpoint
@pytest.mark.network
@pytest.mark.parametrize(
    "spec",
    [_release_parameter(REGISTRY[model_id]) for model_id in esmfold2_bundle.supported_model_ids],
)
def test_esmfold2_candidate_matches_checked_structure_golden(
    spec: ModelSpec,
    tmp_path: Path,
) -> None:
    golden, metadata = _golden(spec)
    _assert_golden_device_matches_current(metadata)
    request_path = esmfold2_bundle.prepare_requests(tmp_path, model_ids=(spec.id,))[0]
    request = esmfold2_bundle.load_request(request_path)
    assert metadata["input_fingerprint"] == request["request_sha256"]

    model = esmfold2_bundle._load_candidate_model(request, torch.device("cuda"), "bf16")
    candidate = esmfold2_bundle._run_fold(model, request)

    immutable_inputs = [
        name
        for name in golden
        if name.startswith("feature__") or name == "noise__initial_standard_normal"
    ]
    assert immutable_inputs
    for name in immutable_inputs:
        assert name in candidate, f"{spec.id}: candidate omitted golden input {name}"
        assert torch.equal(candidate[name], golden[name]), f"{spec.id}: {name}"
    esmfold2_metrics._assert_valid_geometry(
        golden,
        context=f"{spec.id} checked official golden",
    )
    esmfold2_metrics._assert_valid_geometry(
        candidate,
        context=f"{spec.id} FastPLMs candidate",
    )
    structure = esmfold2_metrics._structure_metrics(candidate, golden)
    esmfold2_metrics._assert_thresholds(
        structure,
        targets=esmfold2_metrics.bf16_targets,
        hard_limits=esmfold2_metrics.bf16_hard_limits,
        context=f"{spec.id} checked BF16 golden",
    )

    del model, candidate, golden
    gc.collect()
    torch.cuda.empty_cache()
