"""Release gates over isolated official and candidate Boltz2 bundles."""

from __future__ import annotations

import ast
import copy
import inspect
import os
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fastplms.models.boltz.modeling_boltz2 import Boltz2Config
from fastplms.registry import get_model_registry
from tests.structure.support import boltz2_bundle
from tests.structure.support.boltz2_bundle import load_bundle, load_request
from tests.structure.support.hardware import (
    assert_same_hopper_sm90_device,
    hopper_sm90_fingerprint,
)
from tests.structure.support.state_contract import semantic_config_contract

bf16_targets = {
    "ca_rmsd": 0.10,
    "lddt_ca": 0.995,
    "plddt_mae": 0.001,
    "pae_mae": 0.10,
    "ptm_error": 0.002,
    "iptm_error": 0.002,
    "mean_probability_jsd": 0.001,
}
bf16_hard_limits = {
    "ca_rmsd": 0.25,
    "lddt_ca": 0.99,
    "plddt_mae": 0.005,
    "pae_mae": 0.50,
    "ptm_error": 0.005,
    "iptm_error": 0.005,
    "mean_probability_jsd": 0.005,
}
bf16_relative_l2_target = 1e-2
bf16_relative_l2_hard_limit = 3e-2


def _exchange_root() -> Path:
    return Path(os.environ.get("FASTPLMS_REFERENCE_EXCHANGE", "artifacts/reference"))


def _paths() -> tuple[Path, Path, Path]:
    root = _exchange_root()
    request = (
        root
        / "structure"
        / "requests"
        / boltz2_bundle.reference_container
        / f"{boltz2_bundle.model_id}.json"
    )
    results = root / "structure" / "results"
    reference = results / "reference" / boltz2_bundle.model_id / boltz2_bundle.fold_dtype
    candidate = results / "candidate" / boltz2_bundle.model_id / boltz2_bundle.fold_dtype
    return request, reference, candidate


def _checkpoint_contract(checkpoint: object) -> dict[str, object]:
    return {
        "repo_id": checkpoint.repo_id,
        "revision": checkpoint.revision,
        "files": [
            {"path": item.path, "algorithm": item.algorithm, "digest": item.digest}
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


def _features(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix("feature__"): tensor
        for name, tensor in tensors.items()
        if name.startswith("feature__")
    }


def _output(tensors: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    key = f"output__{name}"
    if key not in tensors:
        raise KeyError(f"Boltz2 bundle omits required output {name!r}.")
    return tensors[key]


def _assert_bundle_identity(
    metadata: Mapping[str, object],
    request: Mapping[str, object],
    *,
    producer: str,
) -> None:
    registry = get_model_registry()
    spec = registry[boltz2_bundle.model_id]
    assert metadata["producer"] == producer
    assert metadata["model_id"] == spec.id
    assert metadata["request_sha256"] == request["request_sha256"]
    assert metadata["official"] == _checkpoint_contract(spec.official)
    assert metadata["candidate"] == _checkpoint_contract(spec.fast)
    assert metadata["upstream"] == _upstream_contract(registry.upstreams["boltz"])
    for name in (
        "sequence",
        "feature_seed",
        "seed",
        "recycling_steps",
        "sampling_steps",
        "diffusion_samples",
        "diffusion_noise_generator",
        "conformer_policy",
        "steering",
        "dtype",
        "parameter_dtype",
        "compute_dtype",
        "execution",
    ):
        assert metadata[name] == request[name]
    assert metadata["attention_backend"] == "eager"
    environment = metadata["environment"]
    assert isinstance(environment, Mapping)
    hopper_sm90_fingerprint(environment)
    if producer == "candidate":
        assert str(environment["torch"]).split("+", maxsplit=1)[0] == "2.13.0"
        packages = environment["packages"]
        assert isinstance(packages, Mapping)
        assert packages["transformers"] == "5.13.0"
        assert str(environment["cuda_runtime"]).startswith("13.0")


def _assert_exact_features(
    actual_tensors: Mapping[str, torch.Tensor],
    actual_metadata: Mapping[str, object],
    expected_tensors: Mapping[str, torch.Tensor],
    expected_metadata: Mapping[str, object],
) -> None:
    actual = _features(actual_tensors)
    expected = _features(expected_tensors)
    assert actual.keys() == expected.keys() == set(boltz2_bundle._feature_names)
    for name in boltz2_bundle._exact_features:
        X = actual[name]
        X_ref = expected[name]
        assert X.dtype == X_ref.dtype, f"{name}: dtype"
        assert X.shape == X_ref.shape, f"{name}: shape"
        assert torch.equal(X, X_ref), f"{name}: values"
    for name in set(actual).difference(boltz2_bundle._exact_features):
        X = actual[name]
        X_ref = expected[name]
        assert X.dtype == X_ref.dtype, f"{name}: dtype"
        assert X.shape == X_ref.shape, f"{name}: shape"
        if X.is_floating_point():
            assert torch.isfinite(X).all(), f"{name}: candidate finite values"
            assert torch.isfinite(X_ref).all(), f"{name}: reference finite values"
    assert torch.equal(actual["ref_pos"], expected["ref_pos"]), "ref_pos: values"
    actual_hash = actual_metadata["feature_sha256"]
    expected_hash = expected_metadata["feature_sha256"]
    assert isinstance(actual_hash, str)
    assert isinstance(expected_hash, str)
    assert actual_hash == expected_hash, "feature_sha256"


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(actual.float() - expected.float())
    scale = torch.linalg.vector_norm(expected.float()).clamp_min(torch.finfo(torch.float32).tiny)
    return (difference / scale).item()


def _first_coordinates(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    X = _output(tensors, "sample_atom_coords").float()
    return X.reshape(-1, X.shape[-2], 3)[0]


def _ca_mask(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    features = _features(tensors)
    encoded = torch.tensor([ord("C") - 32, ord("A") - 32, 0, 0])
    atom_names = features["ref_atom_name_chars"][0].argmax(dim=-1)
    atom_mask = features["atom_pad_mask"][0].bool()
    ca_mask = atom_names.eq(encoded).all(dim=-1) & atom_mask
    assert ca_mask.sum().item() == len(boltz2_bundle.fold_sequence)
    return ca_mask


def _ca_coordinates(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return _first_coordinates(tensors)[_ca_mask(tensors)]


def _aligned_rmsd(actual: torch.Tensor, expected: torch.Tensor) -> float:
    X = actual.float() - actual.float().mean(dim=0, keepdim=True)
    X_ref = expected.float() - expected.float().mean(dim=0, keepdim=True)
    covariance = X.T @ X_ref
    U, _, Vh = torch.linalg.svd(covariance)
    correction = torch.eye(3)
    correction[-1, -1] = torch.sign(torch.det(U @ Vh))
    rotation = U @ correction @ Vh
    aligned = X @ rotation
    return torch.sqrt(torch.mean(torch.sum((aligned - X_ref) ** 2, dim=-1))).item()


def _lddt_ca(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_distances = torch.cdist(actual.float(), actual.float())
    expected_distances = torch.cdist(expected.float(), expected.float())
    pair_mask = expected_distances.lt(15.0)
    pair_mask.fill_diagonal_(False)
    assert pair_mask.any()
    errors = (actual_distances - expected_distances).abs()
    scores = torch.stack([errors.lt(threshold).float() for threshold in (0.5, 1.0, 2.0, 4.0)]).mean(
        dim=0
    )
    return scores[pair_mask].mean().item()


def _probability_jsd(
    actual_logits: torch.Tensor,
    expected_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    actual_log_prob = F.log_softmax(actual_logits.float(), dim=-1)
    expected_log_prob = F.log_softmax(expected_logits.float(), dim=-1)
    actual_prob = actual_log_prob.exp()
    expected_prob = expected_log_prob.exp()
    mean_prob = 0.5 * (actual_prob + expected_prob)
    log_mean_prob = mean_prob.clamp_min(torch.finfo(torch.float32).tiny).log()
    divergence = 0.5 * (
        (actual_prob * (actual_log_prob - log_mean_prob)).sum(dim=-1)
        + (expected_prob * (expected_log_prob - log_mean_prob)).sum(dim=-1)
    )
    while mask.ndim < divergence.ndim:
        mask = mask.unsqueeze(0)
    mask = torch.broadcast_to(mask, divergence.shape)
    return divergence[mask].mean()


def _metrics(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, float]]:
    features = _features(actual)
    token_mask = features["token_pad_mask"].bool()
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    plddt_actual = _output(actual, "plddt").float().reshape_as(token_mask)
    plddt_expected = _output(expected, "plddt").float().reshape_as(token_mask)
    pae_actual = _output(actual, "pae").float().reshape_as(pair_mask)
    pae_expected = _output(expected, "pae").float().reshape_as(pair_mask)
    probability_values = [
        _probability_jsd(
            _output(actual, "pdistogram").squeeze(-2),
            _output(expected, "pdistogram").squeeze(-2),
            pair_mask,
        ),
        _probability_jsd(
            _output(actual, "pde_logits"),
            _output(expected, "pde_logits"),
            pair_mask,
        ),
        _probability_jsd(
            _output(actual, "pae_logits"),
            _output(expected, "pae_logits"),
            pair_mask,
        ),
        _probability_jsd(
            _output(actual, "plddt_logits"),
            _output(expected, "plddt_logits"),
            token_mask,
        ),
    ]
    structure_metrics = {
        "ca_rmsd": _aligned_rmsd(_ca_coordinates(actual), _ca_coordinates(expected)),
        "lddt_ca": _lddt_ca(_ca_coordinates(actual), _ca_coordinates(expected)),
        "plddt_mae": (plddt_actual[token_mask] - plddt_expected[token_mask]).abs().mean().item(),
        "pae_mae": (pae_actual[pair_mask] - pae_expected[pair_mask]).abs().mean().item(),
        "ptm_error": (
            _output(actual, "ptm").float().reshape(-1)[0]
            - _output(expected, "ptm").float().reshape(-1)[0]
        )
        .abs()
        .item(),
        "iptm_error": (
            _output(actual, "iptm").float().reshape(-1)[0]
            - _output(expected, "iptm").float().reshape(-1)[0]
        )
        .abs()
        .item(),
        "mean_probability_jsd": torch.stack(probability_values).mean().item(),
    }
    relative_l2 = {
        name: _relative_l2(_output(actual, name), _output(expected, name))
        for name in ("pdistogram", "pde_logits", "pae_logits", "plddt_logits")
    }
    return structure_metrics, relative_l2


def _assert_valid_outputs(tensors: Mapping[str, torch.Tensor], *, context: str) -> None:
    features = _features(tensors)
    atom_mask = features["atom_pad_mask"][0].bool()
    coordinates = _first_coordinates(tensors)
    assert torch.isfinite(coordinates[atom_mask]).all(), f"{context}: coordinates"
    for name in boltz2_bundle._required_outputs:
        X = _output(tensors, name)
        if X.is_floating_point():
            assert torch.isfinite(X).all(), f"{context}: {name}"


def test_boltz2_reference_path_has_no_fastplms_dependency() -> None:
    tree = ast.parse(inspect.getsource(boltz2_bundle))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("fastplms") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("fastplms")
    for function in (
        boltz2_bundle._prepare_reference_features,
        boltz2_bundle._load_reference_model,
        boltz2_bundle._run_model,
        boltz2_bundle.produce_reference,
    ):
        assert "fastplms" not in inspect.getsource(function).lower()


def test_boltz2_request_is_manifest_exact(tmp_path: Path) -> None:
    path = boltz2_bundle.prepare_request(tmp_path)
    request = load_request(path)
    registry = get_model_registry()
    spec = registry[boltz2_bundle.model_id]
    assert request["official"] == _checkpoint_contract(spec.official)
    assert request["candidate"] == _checkpoint_contract(spec.fast)
    assert request["upstream"] == _upstream_contract(registry.upstreams["boltz"])
    assert request["sequence"] == boltz2_bundle.fold_sequence
    assert spec.family.bf16_execution == "fp32_parameters_autocast"
    assert request["parameter_dtype"] == boltz2_bundle.fold_parameter_dtype
    assert request["compute_dtype"] == boltz2_bundle.fold_compute_dtype
    assert request["execution"] == boltz2_bundle.fold_execution
    assert request["feature_names"] == list(boltz2_bundle._feature_names)
    assert request["output_names"] == list(boltz2_bundle._required_outputs)


def test_boltz2_parameter_storage_contract_is_strict() -> None:
    model = torch.nn.Linear(3, 2, bias=False)
    boltz2_bundle._require_floating_parameter_dtype(model, torch.float32)

    with pytest.raises(RuntimeError, match=r"requires torch\.float32 parameter storage"):
        boltz2_bundle._require_floating_parameter_dtype(model.to(torch.bfloat16), torch.float32)


def test_boltz2_portable_noise_never_reads_pytorch_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_random_draw(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("Portable Boltz2 noise called PyTorch's RNG.")

    monkeypatch.setattr(torch, "randn", forbidden_random_draw)
    monkeypatch.setattr(torch, "randn_like", forbidden_random_draw)
    streams: list[list[torch.Tensor]] = []
    for global_seed in (1, 987_654_321):
        torch.manual_seed(global_seed)
        with boltz2_bundle._portable_random_draws(17) as captured:
            # N0 and N1 cover both stochastic interfaces used by Boltz2.
            N0 = torch.randn((2, 3), dtype=torch.float32)
            N1 = torch.randn_like(torch.empty(4, dtype=torch.bfloat16))
            out = torch.empty(2, dtype=torch.float64)
            N2 = torch.randn(2, dtype=torch.float64, out=out)
        assert torch.randn is forbidden_random_draw
        assert torch.randn_like is forbidden_random_draw
        assert N2 is out
        assert len(captured) == 3
        for N_recorded, N_actual in zip(captured, (N0, N1, N2), strict=True):
            assert torch.equal(N_recorded, N_actual)
        streams.append([N.clone() for N in captured])

    assert len(streams[0]) == len(streams[1]) == 3
    for N0, N1 in zip(streams[0], streams[1], strict=True):
        assert torch.equal(N0, N1)


def test_boltz2_inference_config_normalization_is_narrow() -> None:
    base = {
        "core_kwargs": {
            "atom_s": 128,
            "pairformer_args": {"v2": True},
            "msa_args": {},
            "diffusion_process_args": {"step_scale": 1.5},
        }
    }
    training_and_backend_variant = {
        "core_kwargs": {
            "atom_s": 128,
            "pairformer_args": {
                "activation_checkpointing": True,
                "dropout": 0.25,
                "post_layer_norm": False,
                "use_trifast": True,
                "v2": True,
            },
            "msa_args": {
                "activation_checkpointing": True,
                "miniformer_blocks": False,
                "msa_dropout": 0.15,
                "subsample_msa": True,
                "z_dropout": 0.25,
            },
            "diffusion_process_args": {
                "mse_rotational_alignment": True,
                "step_scale": 1.5,
                "step_scale_random": [1.0, 1.5],
            },
        },
        "dtype": "float32",
    }
    normalize = boltz2_bundle.normalize_inference_config_contract
    assert normalize(semantic_config_contract(base)) == normalize(
        semantic_config_contract(training_and_backend_variant)
    )

    changed_architecture = copy.deepcopy(training_and_backend_variant)
    changed_architecture["core_kwargs"]["atom_s"] = 256
    assert normalize(semantic_config_contract(base)) != normalize(
        semantic_config_contract(changed_architecture)
    )


@pytest.mark.structure
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.large
def test_boltz2_live_folding_matches_pinned_official() -> None:
    request_path, reference_path, candidate_path = _paths()
    request = load_request(request_path)
    reference_tensors, reference_metadata = load_bundle(reference_path)
    candidate_tensors, candidate_metadata = load_bundle(candidate_path)
    _assert_bundle_identity(reference_metadata, request, producer="reference")
    _assert_bundle_identity(candidate_metadata, request, producer="candidate")
    reference_environment = reference_metadata["environment"]
    candidate_environment = candidate_metadata["environment"]
    assert isinstance(reference_environment, Mapping)
    assert isinstance(candidate_environment, Mapping)
    assert_same_hopper_sm90_device(candidate_environment, reference_environment)
    expected_keys = set(candidate_metadata["state"]["tensors"])
    canonical_reference = boltz2_bundle.canonicalize_reference_state_contract(
        reference_metadata["state"],
        expected_keys=expected_keys,
    )
    assert canonical_reference["tensors"] == candidate_metadata["state"]["tensors"]
    assert canonical_reference["aliases"] == candidate_metadata["state"]["aliases"]

    official_hparams = dict(reference_metadata["semantic_config"]["fields"])
    canonical_config = Boltz2Config.from_hyperparameters(
        official_hparams,
        use_kernels=False,
    )
    canonical_config_contract = boltz2_bundle.normalize_inference_config_contract(
        semantic_config_contract(canonical_config)
    )
    candidate_config_contract = boltz2_bundle.normalize_inference_config_contract(
        candidate_metadata["semantic_config"]
    )
    assert canonical_config_contract == candidate_config_contract
    _assert_exact_features(
        candidate_tensors,
        candidate_metadata,
        reference_tensors,
        reference_metadata,
    )
    assert torch.equal(
        candidate_tensors["noise__initial_standard_normal"],
        reference_tensors["noise__initial_standard_normal"],
    )
    candidate_noise = {
        name: tensor
        for name, tensor in candidate_tensors.items()
        if name.startswith("noise__draw_")
    }
    reference_noise = {
        name: tensor
        for name, tensor in reference_tensors.items()
        if name.startswith("noise__draw_")
    }
    assert candidate_noise.keys() == reference_noise.keys()
    for name in candidate_noise:
        assert torch.equal(candidate_noise[name], reference_noise[name]), name
    assert (
        candidate_metadata["diffusion_noise_draw_count"]
        == reference_metadata["diffusion_noise_draw_count"]
    )
    assert (
        candidate_metadata["diffusion_noise_sha256"] == reference_metadata["diffusion_noise_sha256"]
    )
    _assert_valid_outputs(reference_tensors, context="official Boltz2")
    _assert_valid_outputs(candidate_tensors, context="FastPLMs Boltz2")
    metrics, relative_l2 = _metrics(candidate_tensors, reference_tensors)
    failures = []
    for name, value in relative_l2.items():
        if value > bf16_relative_l2_hard_limit:
            failures.append(
                f"{name} relative L2 {value:.6g} exceeds hard limit "
                f"{bf16_relative_l2_hard_limit:.6g}"
            )
        elif value > bf16_relative_l2_target:
            failures.append(
                f"{name} relative L2 {value:.6g} misses engineering target "
                f"{bf16_relative_l2_target:.6g}"
            )
    for name, hard_limit in bf16_hard_limits.items():
        value = metrics[name]
        if name == "lddt_ca":
            if value < hard_limit:
                failures.append(f"{name} {value:.6g} is below hard limit {hard_limit:.6g}")
        elif value > hard_limit:
            failures.append(f"{name} {value:.6g} exceeds hard limit {hard_limit:.6g}")
    for name, target in bf16_targets.items():
        value = metrics[name]
        if name == "lddt_ca":
            if value < target and value >= bf16_hard_limits[name]:
                failures.append(f"{name} {value:.6g} misses target {target:.6g}")
        elif value > target and value <= bf16_hard_limits[name]:
            failures.append(f"{name} {value:.6g} misses target {target:.6g}")
    assert not failures, "Boltz2 compliance failures:\n- " + "\n- ".join(failures)
