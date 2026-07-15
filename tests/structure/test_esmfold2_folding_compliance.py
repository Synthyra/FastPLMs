"""Release gates over isolated official and candidate ESMFold2 fold bundles."""

from __future__ import annotations

import ast
import inspect
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from fastplms.registry import ModelSpec, get_model_registry
from tests.structure.support import esmfold2_bundle
from tests.structure.support.esmfold2_bundle import load_bundle, load_request

bf16_targets = {
    "ca_rmsd": 0.10,
    "lddt_ca": 0.995,
    "plddt_mae": 0.001,
    "pae_mae": 0.10,
    "ptm_error": 0.002,
    "iptm_error": 0.002,
}
bf16_hard_limits = {
    "ca_rmsd": 0.25,
    "lddt_ca": 0.99,
    "plddt_mae": 0.005,
    "pae_mae": 0.50,
    "ptm_error": 0.005,
    "iptm_error": 0.005,
}
fp8_targets = {
    "ca_rmsd": 0.75,
    "lddt_ca": 0.97,
    "plddt_mae": 0.01,
    "pae_mae": 0.50,
    "ptm_error": 0.01,
    "iptm_error": 0.01,
    "mean_probability_jsd": 0.002,
}
fp8_hard_limits = {
    "ca_rmsd": 1.50,
    "lddt_ca": 0.95,
    "plddt_mae": 0.02,
    "pae_mae": 1.0,
    "ptm_error": 0.02,
    "iptm_error": 0.02,
    "mean_probability_jsd": 0.005,
}


def _exchange_root() -> Path:
    return Path(os.environ.get("FASTPLMS_REFERENCE_EXCHANGE", "artifacts/reference"))


def _bundle_paths(spec: ModelSpec) -> tuple[Path, Path, Path, Path]:
    root = _exchange_root()
    request = (
        root / "structure" / "requests" / esmfold2_bundle.reference_container / f"{spec.id}.json"
    )
    reference = root / "structure" / "results" / "reference" / spec.id
    candidate = root / "structure" / "results" / "candidate" / spec.id
    return request, reference, candidate / "bf16", candidate / "fp8"


def _feature_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name.removeprefix("feature__"): tensor
        for name, tensor in tensors.items()
        if name.startswith("feature__")
    }


def _output(tensors: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    key = f"output__{name}"
    if key not in tensors:
        raise KeyError(f"Structure bundle omits required output {name!r}.")
    return tensors[key]


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


def _assert_bundle_identity(
    metadata: Mapping[str, object],
    request: Mapping[str, object],
    spec: ModelSpec,
    *,
    producer: str,
    precision: str,
) -> None:
    assert metadata["producer"] == producer
    assert metadata["model_id"] == spec.id
    assert metadata["request_sha256"] == request["request_sha256"]
    assert metadata["official"] == _checkpoint_contract(spec.official)
    assert metadata["candidate"] == _checkpoint_contract(spec.fast)
    assert metadata["sequence"] == request["sequence"]
    assert metadata["seed"] == request["seed"]
    assert metadata["sampling_steps"] == request["sampling_steps"]
    assert metadata["attention_backend"] == request["attention_backend"] == "sdpa"
    assert metadata["deterministic_algorithms"] is True
    assert metadata["requested_precision"] == precision
    assert metadata["resolved_precision"] == precision
    status = metadata["precision_status"]
    assert isinstance(status, Mapping)
    assert status["requested"] == precision
    assert status["resolved"] == precision
    environment = metadata["environment"]
    assert isinstance(environment, Mapping)
    assert "H100" in str(environment["cuda_device"])
    if producer == "candidate":
        assert str(environment["torch"]).split("+", maxsplit=1)[0] == "2.13.0"
        assert environment["transformers"] == "5.13.0"
        assert str(environment["cuda_runtime"]).startswith("13.0")
    if precision == "fp8":
        assert status["transformer_engine_version"]
        assert environment["transformer_engine"]


def _assert_exact_inputs(
    actual_tensors: Mapping[str, torch.Tensor],
    actual_metadata: Mapping[str, object],
    expected_tensors: Mapping[str, torch.Tensor],
    expected_metadata: Mapping[str, object],
    *,
    context: str,
) -> None:
    actual_features = _feature_tensors(actual_tensors)
    expected_features = _feature_tensors(expected_tensors)
    assert actual_features.keys() == expected_features.keys(), context
    for name in actual_features:
        actual = actual_features[name]
        expected = expected_features[name]
        assert actual.dtype == expected.dtype, f"{context}: {name} dtype"
        assert actual.shape == expected.shape, f"{context}: {name} shape"
        assert torch.equal(actual, expected), f"{context}: {name} values"
    assert actual_metadata["feature_sha256"] == expected_metadata["feature_sha256"]
    assert torch.equal(
        actual_tensors["noise__initial_standard_normal"],
        expected_tensors["noise__initial_standard_normal"],
    ), f"{context}: initial diffusion noise"
    assert actual_metadata["diffusion_noise_sha256"] == expected_metadata["diffusion_noise_sha256"]


def _first_coordinate_sample(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    coordinates = _output(tensors, "sample_atom_coords").float()
    if coordinates.ndim == 4:
        coordinates = coordinates.reshape(-1, coordinates.shape[-2], 3)
    assert coordinates.ndim == 3 and coordinates.shape[-1] == 3
    return coordinates[0]


def _ca_mask(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    features = _feature_tensors(tensors)
    encoded_ca = torch.tensor([ord("C") - 32, ord("A") - 32, 0, 0])
    atom_names = features["ref_atom_name_chars"][0]
    atom_mask = features["atom_attention_mask"][0].bool()
    mask = atom_names.eq(encoded_ca).all(dim=-1) & atom_mask
    token_ids = features["atom_to_token"][0, mask]
    valid_token_ids = features["token_attention_mask"][0].nonzero(as_tuple=True)[0]
    assert torch.equal(token_ids, valid_token_ids), (
        "Each biological residue must have exactly one C-alpha atom."
    )
    return mask


def _ca_coordinates(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return _first_coordinate_sample(tensors)[_ca_mask(tensors)]


def _aligned_ca_rmsd(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_centered = actual.float() - actual.float().mean(dim=0, keepdim=True)
    expected_centered = expected.float() - expected.float().mean(dim=0, keepdim=True)
    covariance = actual_centered.T @ expected_centered
    left, _, right = torch.linalg.svd(covariance)
    correction = torch.eye(3, dtype=torch.float32)
    correction[-1, -1] = torch.sign(torch.det(left @ right))
    rotation = left @ correction @ right
    aligned = actual_centered @ rotation
    return torch.sqrt(torch.mean(torch.sum((aligned - expected_centered) ** 2, dim=-1))).item()


def _lddt_ca(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_distances = torch.cdist(actual.float(), actual.float())
    expected_distances = torch.cdist(expected.float(), expected.float())
    pair_mask = expected_distances.lt(15.0)
    pair_mask.fill_diagonal_(False)
    assert pair_mask.any(), "No valid C-alpha pairs for lDDT."
    errors = (actual_distances - expected_distances).abs()
    score = torch.stack([errors.lt(threshold).float() for threshold in (0.5, 1.0, 2.0, 4.0)]).mean(
        dim=0
    )
    return score[pair_mask].mean().item()


def _token_vector(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    sequence_length: int,
) -> torch.Tensor:
    return _output(tensors, name).float().reshape(-1, sequence_length)[0]


def _token_pair(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    sequence_length: int,
) -> torch.Tensor:
    return (
        _output(tensors, name)
        .float()
        .reshape(
            -1,
            sequence_length,
            sequence_length,
        )[0]
    )


def _structure_metrics(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    actual_features = _feature_tensors(actual)
    token_mask = actual_features["token_attention_mask"][0].bool()
    sequence_length = token_mask.numel()
    pair_mask = token_mask[:, None] & token_mask[None, :]
    return {
        "ca_rmsd": _aligned_ca_rmsd(
            _ca_coordinates(actual),
            _ca_coordinates(expected),
        ),
        "lddt_ca": _lddt_ca(
            _ca_coordinates(actual),
            _ca_coordinates(expected),
        ),
        "plddt_mae": (
            _token_vector(actual, "plddt", sequence_length)[token_mask]
            - _token_vector(expected, "plddt", sequence_length)[token_mask]
        )
        .abs()
        .mean()
        .item(),
        "pae_mae": (
            _token_pair(actual, "pae", sequence_length)[pair_mask]
            - _token_pair(expected, "pae", sequence_length)[pair_mask]
        )
        .abs()
        .mean()
        .item(),
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
    }


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
    jsd = 0.5 * (
        (actual_prob * (actual_log_prob - log_mean_prob)).sum(dim=-1)
        + (expected_prob * (expected_log_prob - log_mean_prob)).sum(dim=-1)
    )
    while mask.ndim < jsd.ndim:
        mask = mask.unsqueeze(0)
    mask = torch.broadcast_to(mask, jsd.shape)
    assert mask.any()
    return jsd[mask].mean()


def _mean_probability_jsd(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> float:
    features = _feature_tensors(actual)
    atom_mask = features["atom_attention_mask"].bool()
    token_mask = features["token_attention_mask"].bool()
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    values = []
    for name in ("distogram_logits", "plddt_logits", "pae_logits", "pde_logits"):
        key = f"output__{name}"
        if key not in actual or key not in expected:
            continue
        mask = atom_mask if name == "plddt_logits" else pair_mask
        values.append(_probability_jsd(actual[key], expected[key], mask))
    assert values, "No probability tensors were returned for JSD compliance."
    return torch.stack(values).mean().item()


def _assert_valid_geometry(
    tensors: Mapping[str, torch.Tensor],
    *,
    context: str,
) -> None:
    features = _feature_tensors(tensors)
    coordinates = _first_coordinate_sample(tensors)
    atom_mask = features["atom_attention_mask"][0].bool()
    assert torch.equal(
        _output(tensors, "atom_pad_mask").bool().reshape_as(atom_mask),
        atom_mask,
    )
    assert torch.isfinite(coordinates[atom_mask]).all(), f"{context}: non-finite coordinates"
    ca_coordinates = coordinates[_ca_mask(tensors)]
    ca_steps = torch.linalg.vector_norm(ca_coordinates[1:] - ca_coordinates[:-1], dim=-1)
    assert torch.isfinite(ca_steps).all(), f"{context}: non-finite C-alpha distances"
    assert ca_steps.gt(2.0).all() and ca_steps.lt(5.0).all(), (
        f"{context}: invalid consecutive C-alpha distances {ca_steps.tolist()}"
    )
    for name, tensor in tensors.items():
        if name.startswith("output__") and tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), f"{context}: {name} contains NaN or inf"


def _assert_thresholds(
    metrics: Mapping[str, float],
    *,
    targets: Mapping[str, float],
    hard_limits: Mapping[str, float],
    context: str,
) -> None:
    for name, hard_limit in hard_limits.items():
        value = metrics[name]
        if name == "lddt_ca":
            assert value >= hard_limit, (
                f"{context}: {name}={value:.6g} is below hard limit {hard_limit:.6g}"
            )
        else:
            assert value <= hard_limit, (
                f"{context}: {name}={value:.6g} exceeds hard limit {hard_limit:.6g}"
            )
    for name, target in targets.items():
        value = metrics[name]
        if name == "lddt_ca":
            assert value >= target, (
                f"{context}: {name}={value:.6g} misses engineering target {target:.6g}"
            )
        else:
            assert value <= target, (
                f"{context}: {name}={value:.6g} misses engineering target {target:.6g}"
            )


def _spec_parameter(spec: ModelSpec) -> object:
    return pytest.param(spec, id=spec.id, marks=pytest.mark.large)


def test_structure_bundle_reference_path_has_no_fastplms_dependency() -> None:
    """Keep the module copyable into a native service with FastPLMs absent."""

    tree = ast.parse(inspect.getsource(esmfold2_bundle))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("fastplms") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("fastplms")
    for function in (
        esmfold2_bundle._load_reference_model,
        esmfold2_bundle._run_fold,
        esmfold2_bundle.produce_reference,
    ):
        assert "fastplms" not in inspect.getsource(function).lower()


def test_structure_metric_helpers_are_exact_for_rigid_identity() -> None:
    expected = torch.tensor([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [7.2, 1.0, 0.0], [9.0, 4.0, 1.0]])
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    actual = expected @ rotation + torch.tensor([4.0, -2.0, 7.0])
    assert _aligned_ca_rmsd(actual, expected) == pytest.approx(0.0, abs=1e-5)
    assert _lddt_ca(actual, expected) == pytest.approx(1.0)
    logits = torch.tensor([[[1.0, 2.0], [0.0, -1.0]]])
    mask = torch.tensor([[True, True]])
    assert _probability_jsd(logits, logits, mask).item() == pytest.approx(0.0)


def test_prepare_structure_requests_is_manifest_exact(tmp_path: Path) -> None:
    paths = esmfold2_bundle.prepare_requests(tmp_path)
    assert tuple(path.stem for path in paths) == esmfold2_bundle.supported_model_ids
    registry = get_model_registry()
    for path in paths:
        request = load_request(path)
        spec = registry[request["model_id"]]
        assert request["official"] == _checkpoint_contract(spec.official)
        assert request["candidate"] == _checkpoint_contract(spec.fast)
        assert request["candidate_auto_model"] == spec.auto_map["AutoModel"]
        assert request["backbone_model"] == spec.family.backbone_model
        assert request["attention_backend"] == "sdpa"
        assert request["deterministic_algorithms"] is True


def test_all_prepared_requests_requires_the_exact_release_inventory(tmp_path: Path) -> None:
    paths = esmfold2_bundle.prepare_requests(tmp_path)
    selected = esmfold2_bundle._all_prepared_requests(tmp_path)
    assert selected == paths

    paths[0].unlink()
    with pytest.raises(FileNotFoundError, match=r"missing=.*esmfold2"):
        esmfold2_bundle._all_prepared_requests(tmp_path)


def test_esmfold2_semantic_config_ignores_only_packaging_and_runtime_policy() -> None:
    fields = {name: 1 for name in esmfold2_bundle._semantic_config_fields}
    reference = {
        **fields,
        "esmc_id": "biohub/ESMC-6B",
        "max_length": 20,
    }
    candidate = {
        **fields,
        "esmc_id": "Synthyra/ESMplusplus_6B",
        "attn_backend": "sdpa",
        "esmc_precision": "bf16",
    }
    expected = esmfold2_bundle._esmfold2_semantic_config(
        reference,
        backbone_model="esmc_6b",
    )
    assert (
        esmfold2_bundle._esmfold2_semantic_config(
            candidate,
            backbone_model="esmc_6b",
        )
        == expected
    )

    changed = {**candidate, "d_pair": 2}
    assert (
        esmfold2_bundle._esmfold2_semantic_config(
            changed,
            backbone_model="esmc_6b",
        )
        != expected
    )


@pytest.mark.structure
@pytest.mark.compliance
@pytest.mark.slow
@pytest.mark.parametrize(
    "spec",
    [_spec_parameter(spec) for spec in get_model_registry().by_family("esmfold2")],
)
def test_esmfold2_isolated_bf16_and_fp8_folding_compliance(
    spec: ModelSpec,
    record_property: Callable[[str, object], None],
) -> None:
    """Gate native BF16 parity and strict candidate FP8 parity for one snapshot."""

    request_path, reference_path, bf16_path, fp8_path = _bundle_paths(spec)
    request = load_request(request_path)
    reference_tensors, reference_metadata = load_bundle(reference_path)
    bf16_tensors, bf16_metadata = load_bundle(bf16_path)
    fp8_tensors, fp8_metadata = load_bundle(fp8_path)

    _assert_bundle_identity(
        reference_metadata,
        request,
        spec,
        producer="reference",
        precision="bf16",
    )
    _assert_bundle_identity(
        bf16_metadata,
        request,
        spec,
        producer="candidate",
        precision="bf16",
    )
    _assert_bundle_identity(
        fp8_metadata,
        request,
        spec,
        producer="candidate",
        precision="fp8",
    )
    assert bf16_metadata["semantic_config"] == reference_metadata["semantic_config"]
    assert fp8_metadata["semantic_config"] == reference_metadata["semantic_config"]
    assert bf16_metadata["state"] == reference_metadata["state"]
    assert fp8_metadata["state"] == reference_metadata["state"]
    _assert_exact_inputs(
        bf16_tensors,
        bf16_metadata,
        reference_tensors,
        reference_metadata,
        context=f"{spec.id} BF16 official parity",
    )
    _assert_exact_inputs(
        fp8_tensors,
        fp8_metadata,
        bf16_tensors,
        bf16_metadata,
        context=f"{spec.id} FP8/BF16 seeded inputs",
    )
    _assert_valid_geometry(reference_tensors, context=f"{spec.id} official BF16")
    _assert_valid_geometry(bf16_tensors, context=f"{spec.id} FastPLMs BF16")
    _assert_valid_geometry(fp8_tensors, context=f"{spec.id} FastPLMs FP8")

    bf16_metrics = _structure_metrics(bf16_tensors, reference_tensors)
    _assert_thresholds(
        bf16_metrics,
        targets=bf16_targets,
        hard_limits=bf16_hard_limits,
        context=f"{spec.id} BF16 official parity",
    )
    fp8_metrics = _structure_metrics(fp8_tensors, bf16_tensors)
    fp8_metrics["mean_probability_jsd"] = _mean_probability_jsd(
        fp8_tensors,
        bf16_tensors,
    )
    _assert_thresholds(
        fp8_metrics,
        targets=fp8_targets,
        hard_limits=fp8_hard_limits,
        context=f"{spec.id} FP8/BF16 parity",
    )

    record_property("fast_checkpoint_revision", spec.fast.revision)
    record_property("official_checkpoint_revision", spec.official.revision)
    record_property("feature_sha256", bf16_metadata["feature_sha256"])
    record_property(
        "diffusion_noise_sha256",
        bf16_metadata["diffusion_noise_sha256"],
    )
    for name, value in bf16_metrics.items():
        record_property(f"bf16_{name}", value)
    for name, value in fp8_metrics.items():
        record_property(f"fp8_{name}", value)
