"""Producer and read-only validator contracts for official-generated goldens."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from fastplms.registry import (
    FileDigest,
    ModelRegistry,
    ModelSpec,
    OfficialGolden,
    get_model_registry,
)
from tests.parity.support.native_reference import _select_requests
from tests.parity.support.reference_adapters.dplm2 import (
    DPLM2_3B_GENERATION_LIMITATION,
)
from tests.release.test_artifacts import _synthetic_registry
from tools.goldens import (
    GoldenBundleRecord,
    GoldenError,
    check_tier_specs,
    convert_native_result,
    golden_generation_matrix,
    missing_check_golden_ids,
    require_complete_check_goldens,
    require_declared_goldens,
    validate_golden_bundle,
    write_golden_bundle,
)
from tools.goldens.from_native import main as golden_main
from tools.remote.prepare_references import MIXED_LENGTHS, prepare_reference_requests

ROOT = Path(__file__).resolve().parents[2]


def _with_spec(registry: ModelRegistry, spec: ModelSpec) -> ModelRegistry:
    return ModelRegistry(
        schema_version=registry.schema_version,
        upstreams=registry.upstreams,
        attention_kernels=registry.attention_kernels,
        families={spec.family.id: spec.family},
        models={spec.id: spec},
        legal_files=registry.legal_files,
    )


def _official_files(spec: ModelSpec) -> list[dict[str, str]]:
    return [
        {"algorithm": item.algorithm, "digest": item.digest, "path": item.path}
        for item in spec.official.files
    ]


def _write_native_sequence_result(path: Path, spec: ModelSpec) -> None:
    tensors = {
        "input__attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "input__input_ids": torch.tensor([[1, 2, 3], [1, 3, 0]]),
        "output__hidden_0000": torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
        "output__last_hidden_state": torch.arange(
            24, dtype=torch.bfloat16
        ).reshape(2, 3, 4),
        "output__logits": torch.arange(30, dtype=torch.bfloat16).reshape(2, 3, 5),
        "residue_mask": torch.tensor(
            [[False, True, False], [False, True, False]], dtype=torch.bool
        ),
    }
    path.mkdir(parents=True)
    save_file(tensors, path / "bf16.safetensors")
    metadata = {
        "schema_version": 1,
        "model_id": spec.id,
        "family": spec.family.id,
        "reference_repo_id": spec.official.repo_id,
        "reference_revision": spec.official.revision,
        "reference_files": _official_files(spec),
        "state_transform": spec.family.state_transform,
        "environment": {
            "cuda_device": "Synthetic H100",
            "cuda_runtime": "13.0",
            "packages": "{\"torch\":\"2.13.0\"}",
            "python": "3.12.12",
            "torch": "2.13.0",
        },
        "precision_tensor_keys": {"bf16": sorted(tensors)},
    }
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _declare_golden(
    registry: ModelRegistry,
    spec: ModelSpec,
    record: GoldenBundleRecord,
) -> tuple[ModelRegistry, ModelSpec]:
    declaration = OfficialGolden(
        metadata=FileDigest(
            "tests/goldens/toy.json",
            "sha256",
            record.metadata_sha256,
        ),
        tensors=FileDigest(
            "tests/goldens/toy.safetensors",
            "sha256",
            record.tensors_sha256,
        ),
    )
    declared_spec = replace(spec, official_golden=declaration)
    return (
        ModelRegistry(
            schema_version=registry.schema_version,
            upstreams=registry.upstreams,
            families=registry.families,
            models={declared_spec.id: declared_spec},
            legal_files=registry.legal_files,
        ),
        declared_spec,
    )


def test_check_tier_official_golden_matrix_is_complete_and_valid() -> None:
    registry = get_model_registry()
    require_complete_check_goldens(registry)
    records = require_declared_goldens(ROOT, registry, tier="check")
    assert len(records) == len(check_tier_specs(registry))


def test_golden_producer_is_deterministic_and_validator_is_read_only(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    root = tmp_path / "repository"
    first = root / "tests" / "goldens"
    second = tmp_path / "second"
    tensors = {
        "H": torch.arange(24, dtype=torch.bfloat16).reshape(2, 3, 4),
        "logits": torch.arange(10, dtype=torch.float32).reshape(2, 5),
    }
    input_fingerprint = hashlib.sha256(b"synthetic golden input").hexdigest()
    environment = {
        "cuda": "13.0",
        "python": "3.12.12",
        "torch": "2.13.0",
        "transformers": "5.13.0",
        "upstream_environment": "synthetic-reference",
    }
    command = (
        "python",
        "-m",
        "tests.parity.support.generate_golden",
        "--model",
        "toy",
    )

    first_record = write_golden_bundle(
        spec,
        registry,
        tensors,
        metadata_path=first / "toy.json",
        tensors_path=first / "toy.safetensors",
        generation_command=command,
        environment=environment,
        input_fingerprint=input_fingerprint,
    )
    second_record = write_golden_bundle(
        spec,
        registry,
        tensors,
        metadata_path=second / "toy.json",
        tensors_path=second / "toy.safetensors",
        generation_command=command,
        environment=environment,
        input_fingerprint=input_fingerprint,
    )
    assert first_record == second_record

    declared_registry, declared_spec = _declare_golden(registry, spec, first_record)
    metadata = json.loads((first / "toy.json").read_text(encoding="utf-8"))
    assert metadata["sources"] == [
        {
            "id": "toy",
            "revision": "4" * 40,
            "url": "https://github.com/example/toy.git",
        }
    ]
    assert metadata["checkpoint"]["repo_id"] == "upstream/ToyModel"
    assert metadata["checkpoint"]["revision"] == "2" * 40
    assert metadata["environment"]["fingerprint"] == hashlib.sha256(
        json.dumps(environment, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert metadata["generation_command"] == list(command)
    assert metadata["input_fingerprint"] == input_fingerprint
    assert metadata["source_files"] == {}
    assert metadata["tensors"]["H"]["shape"] == [2, 3, 4]
    assert metadata["tensors"]["H"]["dtype"] == "bfloat16"

    paths = (first / "toy.json", first / "toy.safetensors")
    before = tuple(path.stat().st_mtime_ns for path in paths)
    validated = validate_golden_bundle(
        declared_spec,
        declared_registry,
        metadata_path=paths[0],
        tensors_path=paths[1],
        declaration=declared_spec.official_golden,
    )
    check_records = require_declared_goldens(root, declared_registry, tier="check")
    after = tuple(path.stat().st_mtime_ns for path in paths)
    assert validated == first_record
    assert check_records == (first_record,)
    assert after == before


def test_missing_golden_blocks_only_check_when_manifest_declares_it(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    placeholder = GoldenBundleRecord(
        metadata_sha256="a" * 64,
        tensors_sha256="b" * 64,
        tensor_hashes={},
    )
    declared_registry, _ = _declare_golden(registry, spec, placeholder)

    assert require_declared_goldens(tmp_path, declared_registry, tier="compliance") == ()
    assert require_declared_goldens(tmp_path, declared_registry, tier="artifact") == ()
    with pytest.raises(GoldenError, match="Missing required official golden for check tier: toy"):
        require_declared_goldens(tmp_path, declared_registry, tier="check")


def test_golden_validator_rejects_tampering(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, spec = _synthetic_registry(source_root, checkpoint)
    golden_root = tmp_path / "repository" / "tests" / "goldens"
    record = write_golden_bundle(
        spec,
        registry,
        {"output": torch.tensor([1.0, 2.0])},
        metadata_path=golden_root / "toy.json",
        tensors_path=golden_root / "toy.safetensors",
        generation_command=("python", "generate.py"),
        environment={"python": "3.12", "reference": "synthetic"},
        input_fingerprint=hashlib.sha256(b"input").hexdigest(),
    )
    declared_registry, declared_spec = _declare_golden(registry, spec, record)
    tensor_path = golden_root / "toy.safetensors"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tampered")

    with pytest.raises(GoldenError, match="tensor-file digest mismatch"):
        validate_golden_bundle(
            declared_spec,
            declared_registry,
            metadata_path=golden_root / "toy.json",
            tensors_path=tensor_path,
            declaration=declared_spec.official_golden,
        )


def test_native_sequence_converter_is_compact_deterministic_and_fail_closed(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, original_spec = _synthetic_registry(source_root, checkpoint)
    family = replace(original_spec.family, test_tiers=("check",))
    spec = replace(original_spec, family=family)
    registry = _with_spec(registry, spec)
    native = tmp_path / "native" / spec.id
    _write_native_sequence_result(native, spec)
    command = (
        "python",
        "-m",
        "tools.goldens.from_native",
        "--model",
        spec.id,
    )

    first_root = tmp_path / "repository-one"
    second_root = tmp_path / "repository-two"
    first = convert_native_result(
        spec,
        registry,
        native,
        first_root / "tests" / "goldens",
        generation_command=command,
    )
    second = convert_native_result(
        spec,
        registry,
        native,
        second_root / "tests" / "goldens",
        generation_command=command,
    )
    assert first.bundle == second.bundle
    assert first.manifest_declaration(first_root).startswith("official_golden = {")

    golden_tensors = load_file(first.tensors_path, device="cpu")
    assert sorted(golden_tensors) == [
        "input__attention_mask",
        "input__input_ids",
        "output__last_hidden_state",
        "output__logits",
        "residue_mask",
    ]
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["source_files"] == {
        "native/bf16.safetensors": hashlib.sha256(
            (native / "bf16.safetensors").read_bytes()
        ).hexdigest(),
        "native/metadata.json": hashlib.sha256(
            (native / "metadata.json").read_bytes()
        ).hexdigest(),
    }
    assert metadata["environment"]["details"]["cuda_device"] == "Synthetic H100"
    assert metadata["input_fingerprint"] != "0" * 64

    broken_metadata = json.loads((native / "metadata.json").read_text(encoding="utf-8"))
    broken_metadata.pop("environment")
    (native / "metadata.json").write_text(
        json.dumps(broken_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenError, match="has no environment record"):
        convert_native_result(
            spec,
            registry,
            native,
            tmp_path / "broken",
            generation_command=command,
        )


def test_dplm2_3b_official_generation_limitation_is_explicit_and_fail_closed(
    tmp_path: Path,
) -> None:
    """A broken official sampler is evidence, never successful generation parity."""

    registry = get_model_registry()
    spec = registry["dplm2_3b"]
    native = tmp_path / "native" / spec.id
    _write_native_sequence_result(native, spec)
    metadata_path = native / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["generation_limitation"] = DPLM2_3B_GENERATION_LIMITATION
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "goldens"
    record = convert_native_result(
        spec,
        registry,
        native,
        output,
        generation_command=("python", "-m", "tools.goldens", "--model", spec.id),
    )
    compact = json.loads(record.metadata_path.read_text(encoding="utf-8"))
    assert compact["limitations"] == [
        {"capability": "generation", **DPLM2_3B_GENERATION_LIMITATION}
    ]
    validate_golden_bundle(
        spec,
        registry,
        metadata_path=record.metadata_path,
        tensors_path=record.tensors_path,
    )

    broken = json.loads(metadata_path.read_text(encoding="utf-8"))
    broken["generation_limitation"]["reason"] = "different"
    metadata_path.write_text(
        json.dumps(broken, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenError, match="generation limitation mismatch"):
        convert_native_result(
            spec,
            registry,
            native,
            tmp_path / "broken",
            generation_command=("python", "generate.py"),
        )


def test_check_tier_completeness_reports_exact_undeclared_ids(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, original_spec = _synthetic_registry(source_root, checkpoint)
    family = replace(original_spec.family, test_tiers=("check", "compliance"))
    spec = replace(original_spec, family=family)
    registry = _with_spec(registry, spec)
    assert check_tier_specs(registry) == (spec,)
    assert missing_check_golden_ids(registry) == ("toy",)
    with pytest.raises(GoldenError, match="incomplete: toy"):
        require_complete_check_goldens(registry)

    declaration = OfficialGolden(
        metadata=FileDigest("tests/goldens/toy.json", "sha256", "a" * 64),
        tensors=FileDigest("tests/goldens/toy.safetensors", "sha256", "b" * 64),
    )
    declared = replace(spec, official_golden=declaration)
    complete_registry = _with_spec(registry, declared)
    assert missing_check_golden_ids(complete_registry) == ()
    require_complete_check_goldens(complete_registry)


def test_native_structure_converter_requires_reference_hash_contract(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    registry, original_spec = _synthetic_registry(source_root, checkpoint)
    family = replace(
        original_spec.family,
        tokenizer_mode="structure",
        test_tiers=("check", "structure"),
    )
    spec = replace(original_spec, family=family, size_category="structure")
    registry = _with_spec(registry, spec)
    native = tmp_path / "native-structure" / spec.id
    native.mkdir(parents=True)
    tensors = {
        "feature__aatype": torch.tensor([0, 1, 2]),
        "output__sample_atom_coords": torch.arange(18, dtype=torch.float32).reshape(
            1, 2, 3, 3
        ),
    }
    save_file(tensors, native / "bundle.safetensors")

    def raw_hash(T: torch.Tensor) -> str:
        value = T.contiguous().view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(value).hexdigest()

    metadata = {
        "schema_version": 1,
        "producer": "reference",
        "model_id": spec.id,
        "request_sha256": "c" * 64,
        "official": {
            "repo_id": spec.official.repo_id,
            "revision": spec.official.revision,
            "files": _official_files(spec),
        },
        "environment": {
            "python": "3.12.12",
            "torch": "2.13.0",
            "packages": {"transformers": "5.13.0"},
        },
        "tensor_keys": sorted(tensors),
        "tensor_hashes": {name: raw_hash(T) for name, T in tensors.items()},
    }
    metadata_path = native / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    record = convert_native_result(
        spec,
        registry,
        native,
        tmp_path / "structure-golden",
        generation_command=("python", "generate-structure.py"),
    )
    golden_metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
    assert golden_metadata["input_fingerprint"] == "c" * 64
    assert golden_metadata["environment"]["details"]["packages"] == (
        '{"transformers":"5.13.0"}'
    )

    metadata["producer"] = "candidate"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GoldenError, match="only an official reference bundle"):
        convert_native_result(
            spec,
            registry,
            native,
            tmp_path / "candidate-not-golden",
            generation_command=("python", "generate-structure.py"),
        )


def test_native_requests_carry_manifest_checkpoint_file_identities(tmp_path: Path) -> None:
    registry = get_model_registry()
    paths = prepare_reference_requests(tmp_path)
    sequence_specs = tuple(
        spec for spec in registry.values() if spec.family.tokenizer_mode != "structure"
    )
    assert len(paths) == len(sequence_specs)
    by_id = {spec.id: spec for spec in sequence_specs}
    for path in paths:
        request = json.loads(path.read_text(encoding="utf-8"))
        spec = by_id[request["model_id"]]
        assert request["reference_repo_id"] == spec.official.repo_id
        assert request["reference_revision"] == spec.official.revision
        assert request["reference_files"] == _official_files(spec)
        if spec.generation_contract == "official_unavailable":
            assert request["generation_policy"] == spec.generation_contract
            assert request["official_generation_limitation"] == (
                DPLM2_3B_GENERATION_LIMITATION
            )
        else:
            assert "generation_policy" not in request
            assert "official_generation_limitation" not in request
        assert tuple(map(len, request["sequences"])) == MIXED_LENGTHS


def test_manifest_generation_matrix_covers_every_check_checkpoint(
    tmp_path: Path,
) -> None:
    registry = get_model_registry()
    native_root = tmp_path / "native"
    output_root = tmp_path / "goldens"
    entries = golden_generation_matrix(registry, native_root, output_root)
    specs = check_tier_specs(registry)
    assert tuple(entry.model_id for entry in entries) == tuple(spec.id for spec in specs)
    assert len(entries) == 28
    assert sum(entry.kind == "sequence" for entry in entries) == 23
    assert sum(entry.kind == "structure" for entry in entries) == 5
    for entry, spec in zip(entries, specs, strict=True):
        assert entry.reference_container == spec.family.reference_container
        assert entry.metadata_path == output_root / f"{spec.id}.json"
        assert entry.tensors_path == output_root / f"{spec.id}.safetensors"
        assert not entry.native_ready
        assert not entry.converted_ready
        if entry.kind == "sequence":
            assert entry.request_path == (
                native_root
                / "requests"
                / spec.family.reference_container
                / f"{spec.id}.json"
            )
            assert entry.native_result_path == native_root / "results" / spec.id
        else:
            assert entry.request_path == (
                native_root
                / "structure"
                / "requests"
                / spec.family.reference_container
                / f"{spec.id}.json"
            )
            assert entry.native_result_path == (
                native_root / "structure" / "results" / "reference" / spec.id
            )

    nested_spec = next(
        spec for spec in specs if spec.family.tokenizer_mode == "structure"
    )
    nested_result = (
        native_root
        / "structure"
        / "results"
        / "reference"
        / nested_spec.id
        / "bf16"
    )
    nested_result.mkdir(parents=True)
    (nested_result / "metadata.json").write_text("{}\n", encoding="utf-8")
    (nested_result / "bundle.safetensors").write_bytes(b"normalized-bundle")
    fp32_result = nested_result.parent / "fp32"
    fp32_result.mkdir()
    (fp32_result / "metadata.json").write_text("{}\n", encoding="utf-8")
    (fp32_result / "bundle.safetensors").write_bytes(b"normalized-fp32-bundle")
    refreshed = {
        entry.model_id: entry
        for entry in golden_generation_matrix(registry, native_root, output_root)
    }
    assert refreshed[nested_spec.id].native_result_path == nested_result
    assert refreshed[nested_spec.id].native_ready


def test_native_reference_request_selection_is_explicit_and_fail_closed(
    tmp_path: Path,
) -> None:
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    for model_id in ("alpha", "beta"):
        (request_dir / f"{model_id}.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "deep_reference": model_id == "beta",
                }
            ),
            encoding="utf-8",
        )
    assert tuple(path.stem for path in _select_requests(request_dir, None)) == (
        "alpha",
        "beta",
    )
    assert tuple(
        path.stem for path in _select_requests(request_dir, ["beta", "alpha"])
    ) == ("beta", "alpha")
    assert tuple(
        path.stem for path in _select_requests(request_dir, None, deep_only=True)
    ) == ("beta",)
    assert tuple(
        path.stem
        for path in _select_requests(
            request_dir,
            ["alpha", "beta"],
            deep_only=True,
        )
    ) == ("beta",)
    with pytest.raises(ValueError, match="must be unique"):
        _select_requests(request_dir, ["alpha", "alpha"])
    with pytest.raises(FileNotFoundError, match="missing selected models"):
        _select_requests(request_dir, ["missing"])
    (request_dir / "alpha.json").write_text(
        json.dumps({"model_id": "different"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="filename and model ID differ"):
        _select_requests(request_dir, ["alpha"])


def test_golden_status_reports_exact_manifest_gap(capsys: pytest.CaptureFixture[str]) -> None:
    registry = get_model_registry()
    assert golden_main(["--status-only", "--report-missing"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert tuple(report["undeclared_check_goldens"]) == missing_check_golden_ids(registry)
    if missing_check_golden_ids(registry):
        with pytest.raises(GoldenError, match="Check-tier official goldens are incomplete"):
            golden_main(["--status-only", "--require-complete"])
    else:
        assert golden_main(["--status-only", "--require-complete"]) == 0


def test_golden_status_reports_manifest_wide_generation_matrix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = get_model_registry()
    assert (
        golden_main(
            [
                "--status-only",
                "--report-matrix",
                "--native-root",
                str(tmp_path / "native"),
                "--output-root",
                str(tmp_path / "goldens"),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    entries = report["check_golden_matrix"]
    assert tuple(entry["model_id"] for entry in entries) == tuple(
        spec.id for spec in check_tier_specs(registry)
    )
    assert len(entries) == 28
