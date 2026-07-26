"""Release gates over isolated Meta and FastPLMs ESMFold v1 bundles."""

from __future__ import annotations

import ast
import inspect
import os
import pytest
import torch
from collections.abc import Mapping
from pathlib import Path

from fastplms.registry import get_model_registry
from tests.structure.support import esmfold_bundle
from tests.structure.support.esmfold_bundle import load_bundle, load_request
from tests.structure.support.hardware import (
    assert_same_hopper_sm90_device,
    hopper_sm90_fingerprint,
)


relative_l2_targets = {"fp32": 2e-6, "bf16": 1e-2}
relative_l2_hard_limits = {"fp32": 2e-5, "bf16": 3e-2}
structure_targets = {
    "ca_rmsd": 0.10,
    "lddt_ca": 0.995,
    "plddt_mae": 0.0012,
    "pae_mae": 0.10,
    "ptm_error": 0.002,
}
structure_hard_limits = {
    "ca_rmsd": 0.25,
    "lddt_ca": 0.99,
    "plddt_mae": 0.005,
    "pae_mae": 0.50,
    "ptm_error": 0.005,
}


def _exchange_root() -> Path:
    return Path(os.environ.get("FASTPLMS_REFERENCE_EXCHANGE", "artifacts/reference"))


def _paths(precision: str) -> tuple[Path, Path, Path]:
    root = _exchange_root()
    request = (
        root
        / "structure"
        / "requests"
        / esmfold_bundle.reference_container
        / f"{esmfold_bundle.model_id}.json"
    )
    results = root / "structure" / "results"
    reference = results / "reference" / esmfold_bundle.model_id / precision
    candidate = results / "candidate" / esmfold_bundle.model_id / precision
    return request, reference, candidate


def _checkpoint_contract(checkpoint: object) -> dict[str, object]:
    return {
        "repo_id": checkpoint.repo_id,
        "revision": checkpoint.revision,
        "files": [
            {
                "path": item.path,
                "algorithm": item.algorithm,
                "digest": item.digest,
            }
            for item in checkpoint.files
        ],
    }


def _upstream_contract(upstream: object) -> dict[str, object]:
    return {
        "id": upstream.id,
        "path": upstream.path,
        "url": upstream.url,
        "revision": upstream.revision,
        "license_expression": upstream.license_expression,
    }


def _output(tensors: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    key = f"output__{name}"
    if key not in tensors:
        raise KeyError(f"ESMFold bundle omits required output {name!r}.")
    return tensors[key]


def _residue_mask(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    # atom37_mask: (...)
    atom37_mask = _output(tensors, "atom37_atom_exists").bool()
    assert atom37_mask.ndim == 3 and atom37_mask.shape[-1] == 37
    return atom37_mask[0, :, 1]


def _ca_coordinates(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    # P is the atom14 position tensor with shape (n_blocks, b, l, 14, 3).
    P = _output(tensors, "positions").float()
    assert P.ndim == 5 and P.shape[-2:] == (14, 3)
    # coordinates: (..., 3)
    coordinates = P[-1, 0, :, 1]
    return coordinates[_residue_mask(tensors)]


def _aligned_ca_rmsd(actual: torch.Tensor, expected: torch.Tensor) -> float:
    # actual: (...), expected: (...)
    actual_centered = actual.float() - actual.float().mean(dim=0, keepdim=True)
    expected_centered = expected.float() - expected.float().mean(dim=0, keepdim=True)
    covariance = actual_centered.T @ expected_centered
    left, _, right = torch.linalg.svd(covariance)
    # correction: (3, 3)
    correction = torch.eye(3, dtype=torch.float32)
    correction[-1, -1] = torch.sign(torch.det(left @ right))
    rotation = left @ correction @ right
    aligned = actual_centered @ rotation
    return torch.sqrt(torch.mean(torch.sum((aligned - expected_centered) ** 2, dim=-1))).item()


def _lddt_ca(actual: torch.Tensor, expected: torch.Tensor) -> float:
    # actual: (...), expected: (...)
    actual_distances = torch.cdist(actual.float(), actual.float())
    expected_distances = torch.cdist(expected.float(), expected.float())
    # pair_mask: (...)
    pair_mask = expected_distances.lt(15.0)
    pair_mask.fill_diagonal_(False)
    assert pair_mask.any(), "No valid C-alpha pairs for ESMFold lDDT."
    errors = (actual_distances - expected_distances).abs()
    # scores: (...)
    scores = torch.stack([errors.lt(threshold).float() for threshold in (0.5, 1.0, 2.0, 4.0)]).mean(
        dim=0
    )
    return scores[pair_mask].mean().item()


def _structure_metrics(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    residue_mask = _residue_mask(actual)
    # pair_mask: (...)
    pair_mask = residue_mask[:, None] & residue_mask[None, :]
    # Meta ESMFold reports pLDDT on (0, 100); compliance uses (0, 1).
    # actual_plddt: (...)
    actual_plddt = _output(actual, "plddt").float()[0, :, 1] / 100.0
    # expected_plddt: (...)
    expected_plddt = _output(expected, "plddt").float()[0, :, 1] / 100.0
    # actual_pae: (...)
    actual_pae = _output(actual, "predicted_aligned_error").float()[0]
    # expected_pae: (...)
    expected_pae = _output(expected, "predicted_aligned_error").float()[0]
    return {
        "ca_rmsd": _aligned_ca_rmsd(
            _ca_coordinates(actual),
            _ca_coordinates(expected),
        ),
        "lddt_ca": _lddt_ca(
            _ca_coordinates(actual),
            _ca_coordinates(expected),
        ),
        "plddt_mae": (actual_plddt[residue_mask] - expected_plddt[residue_mask])
        .abs()
        .mean()
        .item(),
        "pae_mae": (actual_pae[pair_mask] - expected_pae[pair_mask]).abs().mean().item(),
        "ptm_error": (
            _output(actual, "ptm").float().reshape(-1)[0]
            - _output(expected, "ptm").float().reshape(-1)[0]
        )
        .abs()
        .item(),
    }


def _relative_l2(
    actual: torch.Tensor,
    expected: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    # actual: (...), expected: (...), mask: (...)
    while mask.ndim < actual.ndim:
        # mask: (...)
        mask = mask.unsqueeze(-1)
    mask = torch.broadcast_to(mask, actual.shape)
    difference = (actual.float() - expected.float())[mask]
    reference = expected.float()[mask]
    return (
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(reference).clamp_min(torch.finfo(torch.float32).tiny)
    ).item()


def _logit_metrics(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    residue_mask = _residue_mask(actual)
    # pair_mask: (...)
    pair_mask = residue_mask[:, None] & residue_mask[None, :]
    return {
        "distogram_logits": _relative_l2(
            _output(actual, "distogram_logits"),
            _output(expected, "distogram_logits"),
            pair_mask.unsqueeze(0),
        ),
        "ptm_logits": _relative_l2(
            _output(actual, "ptm_logits"),
            _output(expected, "ptm_logits"),
            pair_mask.unsqueeze(0),
        ),
        "lm_logits": _relative_l2(
            _output(actual, "lm_logits"),
            _output(expected, "lm_logits"),
            residue_mask.unsqueeze(0),
        ),
    }


def _assert_valid_bundle(
    tensors: Mapping[str, torch.Tensor],
    *,
    context: str,
) -> None:
    residue_mask = _residue_mask(tensors)
    assert residue_mask.sum().item() == len(esmfold_bundle.fold_sequence)
    for name, tensor in tensors.items():
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), f"{context}: {name} contains NaN or inf"
    ca_coordinates = _ca_coordinates(tensors)
    ca_steps = torch.linalg.vector_norm(ca_coordinates[1:] - ca_coordinates[:-1], dim=-1)
    assert ca_steps.gt(2.0).all() and ca_steps.lt(5.0).all(), (
        f"{context}: invalid consecutive C-alpha distances {ca_steps.tolist()}"
    )


def _assert_bundle_identity(
    metadata: Mapping[str, object],
    request: Mapping[str, object],
    *,
    producer: str,
    precision: str,
) -> None:
    registry = get_model_registry()
    spec = registry[esmfold_bundle.model_id]
    assert metadata["producer"] == producer
    assert metadata["model_id"] == spec.id
    assert metadata["request_sha256"] == request["request_sha256"]
    assert metadata["official"] == _checkpoint_contract(spec.official)
    assert metadata["candidate"] == _checkpoint_contract(spec.fast)
    assert metadata["upstreams"] == [
        _upstream_contract(registry.upstreams[name]) for name in ("fair-esm", "openfold")
    ]
    assert metadata["sequence"] == esmfold_bundle.fold_sequence
    assert metadata["seed"] == esmfold_bundle.fold_seed
    assert metadata["recycles"] == esmfold_bundle.fold_recycles
    assert metadata["attention_backend"] == esmfold_bundle.fold_backend
    assert metadata["deterministic_algorithms"] is True
    assert metadata["parameter_dtype"] == "float32"
    assert metadata["compute_dtype"] == precision
    assert metadata["esm_parameter_dtypes"] == ["float32"]
    expected_execution = (
        "fp32_parameters_cuda_bf16_autocast" if precision == "bf16" else "fp32_parameters"
    )
    assert metadata["execution"] == expected_execution
    environment = metadata["environment"]
    assert isinstance(environment, Mapping)
    hopper_sm90_fingerprint(environment)
    if producer == "candidate":
        assert str(environment["torch"]).split("+", maxsplit=1)[0] == "2.13.0"
        assert str(environment["cuda_runtime"]).startswith("13.0")
        packages = environment["packages"]
        assert isinstance(packages, Mapping)
        assert packages["transformers"] == "5.13.0"


def test_esmfold_reference_path_has_no_fastplms_dependency() -> None:
    """Keep the Meta oracle copyable into a native image without FastPLMs."""

    tree = ast.parse(inspect.getsource(esmfold_bundle))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("fastplms") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("fastplms")
    for function in (
        esmfold_bundle._load_reference_model,
        esmfold_bundle._run_infer,
        esmfold_bundle.produce_reference,
    ):
        assert "fastplms" not in inspect.getsource(function).lower()


def test_prepare_esmfold_request_is_manifest_exact(tmp_path: Path) -> None:
    path = esmfold_bundle.prepare_request(tmp_path)
    request = load_request(path)
    registry = get_model_registry()
    spec = registry[esmfold_bundle.model_id]
    assert request["official"] == _checkpoint_contract(spec.official)
    assert request["candidate"] == _checkpoint_contract(spec.fast)
    assert request["candidate_auto_model"] == spec.auto_map["AutoModel"]
    assert request["adapter"] == spec.family.reference_adapter
    assert request["deterministic_algorithms"] is True


def test_esmfold_metric_helpers_are_exact_for_rigid_identity() -> None:
    # expected: (4, 3)
    expected = torch.tensor([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.2, 1.0, 0.0], [9.0, 4.0, 1.0]])
    # rotation: (3, 3)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    actual = expected @ rotation + torch.tensor([4.0, -2.0, 7.0])
    assert _aligned_ca_rmsd(actual, expected) == pytest.approx(0.0, abs=1e-5)
    assert _lddt_ca(actual, expected) == pytest.approx(1.0)


def test_esmfold_tensor_hash_accepts_scalar_outputs() -> None:
    assert esmfold_bundle.tensor_sha256(torch.tensor(0.5)) == esmfold_bundle.tensor_sha256(
        torch.tensor([0.5])
    )


@pytest.mark.parametrize("precision", esmfold_bundle.supported_precisions)
@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.large
def test_esmfold_live_official_structure_parity(precision: str) -> None:
    """The pinned candidate matches Meta ESMFold v1 under FP32 and BF16 compute."""

    request_path, reference_path, candidate_path = _paths(precision)
    request = load_request(request_path)
    reference_tensors, reference_metadata = load_bundle(reference_path)
    candidate_tensors, candidate_metadata = load_bundle(candidate_path)
    _assert_bundle_identity(
        reference_metadata,
        request,
        producer="reference",
        precision=precision,
    )
    _assert_bundle_identity(
        candidate_metadata,
        request,
        producer="candidate",
        precision=precision,
    )
    reference_environment = reference_metadata["environment"]
    candidate_environment = candidate_metadata["environment"]
    assert isinstance(reference_environment, Mapping)
    assert isinstance(candidate_environment, Mapping)
    assert_same_hopper_sm90_device(candidate_environment, reference_environment)
    assert candidate_metadata["semantic_config"] == reference_metadata["semantic_config"]
    assert candidate_metadata["state"] == reference_metadata["state"]
    assert reference_tensors.keys() == candidate_tensors.keys()
    for name in esmfold_bundle._exact_outputs:
        actual = _output(candidate_tensors, name)
        expected = _output(reference_tensors, name)
        assert actual.dtype == expected.dtype, name
        assert actual.shape == expected.shape, name
        assert torch.equal(actual, expected), name
    _assert_valid_bundle(reference_tensors, context="Meta ESMFold v1")
    _assert_valid_bundle(candidate_tensors, context="FastPLMs ESMFold v1")

    logits = _logit_metrics(candidate_tensors, reference_tensors)
    relative_l2_target = relative_l2_targets[precision]
    relative_l2_hard_limit = relative_l2_hard_limits[precision]
    for name, value in logits.items():
        assert value <= relative_l2_hard_limit, (
            f"ESMFold {name} relative L2 {value:.6g} exceeds hard limit "
            f"{relative_l2_hard_limit:.6g} under {precision} compute."
        )
        assert value <= relative_l2_target, (
            f"ESMFold {name} relative L2 {value:.6g} misses engineering target "
            f"{relative_l2_target:.6g} under {precision} compute."
        )

    metrics = _structure_metrics(candidate_tensors, reference_tensors)
    for name, hard_limit in structure_hard_limits.items():
        value = metrics[name]
        if name == "lddt_ca":
            assert value >= hard_limit, (
                f"ESMFold {name} {value:.6g} is below hard limit {hard_limit:.6g}."
            )
        else:
            assert value <= hard_limit, (
                f"ESMFold {name} {value:.6g} exceeds hard limit {hard_limit:.6g}."
            )
    for name, target in structure_targets.items():
        value = metrics[name]
        if name == "lddt_ca":
            assert value >= target, f"ESMFold {name} {value:.6g} misses target {target:.6g}."
        else:
            assert value <= target, f"ESMFold {name} {value:.6g} misses target {target:.6g}."
