"""Fail-closed tests for the native GH200 Biohub dependency lock."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import pytest
from dataclasses import asdict
from pathlib import Path

import tools.remote.biohub_reference_lock as biohub_reference_lock
from tools.remote.biohub_reference_environment import (
    BiohubReferenceEnvironmentError,
    validate_biohub_reference_environment_evidence,
)
from tools.remote.biohub_reference_lock import (
    BiohubReferenceLockError,
    assert_exact_installed_inventory,
    expected_installed_inventory,
    load_biohub_reference_lock_contract,
    materialize_biotraj_wheel_lock,
    verify_biohub_reference_lock_contract,
    verify_current_pip_check,
)


_ROOT = Path(__file__).parents[2]
_CONTRACT = _ROOT / "docker/constraints/biohub-reference-lock.json"
_CONTRACT_FILES = (
    "docker/biohub-reference-lock.Dockerfile",
    "docker/constraints/biohub-reference.in",
    "docker/constraints/biohub-reference.lock.txt",
    "docker/constraints/biohub-biotraj-build.in",
    "docker/constraints/biohub-biotraj-build.lock.txt",
    "docker/constraints/biohub-reference-lock.json",
)


def _copy_contract_root(tmp_path: Path) -> tuple[Path, Path]:
    for relative in _CONTRACT_FILES:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_ROOT / relative, destination)
    return tmp_path, tmp_path / "docker/constraints/biohub-reference-lock.json"


def test_checked_in_biohub_reference_lock_is_exact_gh200_contract() -> None:
    evidence = verify_biohub_reference_lock_contract(_ROOT, _CONTRACT)

    assert evidence == {
        "schema_version": 2,
        "target": {
            "hardware": "NVIDIA GH200 480GB",
            "operating_system": "linux",
            "architecture": "aarch64",
            "container_platform": "linux/arm64",
            "python_implementation": "CPython",
            "python_version": "3.12",
            "cuda_version": "13.0",
            "torch_backend": "cu130",
        },
        "runtime_lock_sha256": ("f87033dffffe953478b482dae82f91603fa705a68e92ee7683c1831586c94ca0"),
        "runtime_package_count": 108,
        "build_lock_sha256": ("c7864daa96028aba35081110c563b8b08c968fd312f7827449d355638f18079d"),
        "build_package_count": 7,
        "biotraj_sdist_sha256": (
            "4bcba92101ed50f369cc1487fb5dfcfe1d8402ad47adaa9232b080553271663a"
        ),
        "biotraj_wheel_sha256": (
            "253c1354c401e97d6e951f29e0d768deb5263de6662001281870425c37719f6b"
        ),
        "pip_check_platform_exceptions": [
            {
                "distribution": "nvidia-cusparselt-cu13",
                "version": "0.8.1",
                "wheel_filename": (
                    "nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_aarch64.whl"
                ),
                "wheel_sha256": (
                    "4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f"
                ),
                "filename_platform_tag": "py3-none-manylinux2014_aarch64",
                "wheel_metadata_platform_tag": "py3-none-manylinux2014_sbsa",
                "target_hardware": "NVIDIA GH200 480GB",
                "target_operating_system": "linux",
                "target_architecture": "aarch64",
                "accepted_diagnostic": (
                    "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform"
                ),
                "resolution": "validated-vendor-metadata-exception-no-wheel-rewrite",
            }
        ],
    }


def test_biohub_lock_contract_rejects_any_target_broadening(tmp_path: Path) -> None:
    payload = json.loads(_CONTRACT.read_text(encoding="utf-8"))
    payload["target"]["architecture"] = "x86_64"
    changed = tmp_path / "contract.json"
    changed.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(BiohubReferenceLockError, match="Unsupported Biohub lock target"):
        load_biohub_reference_lock_contract(changed)


def test_biohub_lock_rejects_byte_mutation_and_forged_digest(tmp_path: Path) -> None:
    root, contract_path = _copy_contract_root(tmp_path)
    lock = root / "docker/constraints/biohub-reference.lock.txt"
    lock.write_bytes(lock.read_bytes() + b"# injected\n")

    with pytest.raises(BiohubReferenceLockError, match="file digest differs"):
        verify_biohub_reference_lock_contract(root, contract_path)

    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["runtime"]["lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    contract_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(BiohubReferenceLockError, match="Unexpected comment"):
        verify_biohub_reference_lock_contract(root, contract_path)


def test_materialized_runtime_lock_uses_only_attested_native_wheel(
    tmp_path: Path,
) -> None:
    root, contract_path = _copy_contract_root(tmp_path)
    wheel = tmp_path / "biotraj-1.2.2-cp312-cp312-linux_aarch64.whl"
    wheel.write_bytes(b"deterministic-native-wheel")
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["biotraj"]["wheel_sha256"] = wheel_sha256
    payload["biotraj"]["wheel_size"] = wheel.stat().st_size
    contract_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    output = tmp_path / "materialized.lock.txt"
    wheel_uri = f"file:///opt/wheels/{wheel.name}"

    parsed = materialize_biotraj_wheel_lock(
        root,
        contract_path,
        wheel,
        output,
        wheel_uri=wheel_uri,
    )

    rendered = output.read_text(encoding="utf-8")
    assert len(parsed.inventory) == 108
    assert parsed.inventory["biotraj"] == "1.2.2"
    assert "--no-binary biotraj" not in rendered
    assert f"biotraj @ {wheel_uri}" in rendered
    assert f"--hash=sha256:{wheel_sha256}" in rendered
    assert "biotraj-1.2.2.tar.gz" not in rendered

    wheel.write_bytes(wheel.read_bytes() + b"mutation")
    with pytest.raises(BiohubReferenceLockError, match="size differs"):
        materialize_biotraj_wheel_lock(
            root,
            contract_path,
            wheel,
            output,
            wheel_uri=wheel_uri,
        )


def test_exact_inventory_profiles_include_only_declared_overlays() -> None:
    build = expected_installed_inventory(_ROOT, _CONTRACT, profile="build")
    runtime = expected_installed_inventory(_ROOT, _CONTRACT, profile="runtime")
    final = expected_installed_inventory(_ROOT, _CONTRACT, profile="final")

    assert len(build) == 7
    assert len(runtime) == 109
    assert len(final) == 112
    assert runtime["pip"] == "26.1.1"
    assert "esm" not in runtime and "transformers" not in runtime and "uv" not in runtime
    assert final["esm"] == "3.3.0"
    assert final["transformers"] == "4.57.6"
    assert final["uv"] == "0.10.12"


def test_inventory_comparison_normalizes_names_and_torch_cuda_local_version() -> None:
    expected = {"huggingface-hub": "0.36.2", "torch": "2.13.0+cu130"}
    observed = {"huggingface_hub": "0.36.2", "Torch": "2.13.0+cu13_0"}

    assert assert_exact_installed_inventory(expected, observed) == {
        "huggingface-hub": "0.36.2",
        "torch": "2.13.0+cu130",
    }


@pytest.mark.parametrize(
    "observed",
    (
        {"torch": "2.13.0+cu130"},
        {"torch": "2.13.0+cu130", "numpy": "1.26.4", "rogue": "1"},
        {"torch": "2.13.0+cpu", "numpy": "1.26.4"},
    ),
)
def test_inventory_comparison_rejects_missing_extra_and_changed(
    observed: dict[str, str],
) -> None:
    expected = {"torch": "2.13.0+cu130", "numpy": "1.26.4"}

    with pytest.raises(BiohubReferenceLockError, match="inventory differs"):
        assert_exact_installed_inventory(expected, observed)


def _patch_pip_check_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    wheel_tag: str = "py3-none-manylinux2014_sbsa",
    stdout: str = "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform\n",
    returncode: int = 1,
) -> None:
    class Distribution:
        @staticmethod
        def read_text(filename: str) -> str:
            assert filename == "WHEEL"
            return f"Wheel-Version: 1.0\nTag: {wheel_tag}\n"

    monkeypatch.setattr(
        biohub_reference_lock,
        "verify_current_installed_inventory",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(biohub_reference_lock.platform, "system", lambda: "Linux")
    monkeypatch.setattr(biohub_reference_lock.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        biohub_reference_lock.importlib.metadata,
        "distribution",
        lambda _name: Distribution(),
    )
    monkeypatch.setattr(
        biohub_reference_lock.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr=""
        ),
    )


def test_pip_check_accepts_only_attested_nvidia_sbsa_tag_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pip_check_runtime(monkeypatch)
    contract = load_biohub_reference_lock_contract(_CONTRACT)

    assert verify_current_pip_check(_ROOT, _CONTRACT) == {
        "status": "accepted-platform-exception",
        "returncode": 1,
        "diagnostics": ["nvidia-cusparselt-cu13 0.8.1 is not supported on this platform"],
        "accepted_platform_exceptions": [
            asdict(exception) for exception in contract.pip_check_platform_exceptions
        ],
    }


def test_pip_check_rejects_any_additional_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pip_check_runtime(
        monkeypatch,
        stdout=(
            "nvidia-cusparselt-cu13 0.8.1 is not supported on this platform\n"
            "rogue 1.0 requires missing-package\n"
        ),
    )

    with pytest.raises(BiohubReferenceLockError, match="differs from the one accepted"):
        verify_current_pip_check(_ROOT, _CONTRACT)


def test_pip_check_rejects_changed_vendor_wheel_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_pip_check_runtime(monkeypatch, wheel_tag="py3-none-manylinux2014_aarch64")

    with pytest.raises(BiohubReferenceLockError, match="WHEEL tag differs"):
        verify_current_pip_check(_ROOT, _CONTRACT)


def _canonical_digest(value: object) -> str:
    serialized = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(serialized).hexdigest()


def _reference_environment_payload() -> dict[str, object]:
    contract = load_biohub_reference_lock_contract(_CONTRACT)
    inventory = expected_installed_inventory(_ROOT, _CONTRACT, profile="final")
    digest = "sha256:" + "a" * 64
    image = {
        "content_digest": digest,
        "image_id": digest,
        "os": "linux",
        "architecture": "arm64",
        "resolved_platform": "linux/arm64",
    }
    container_identity = {
        "schema_version": 1,
        "resolved_platform": "linux/arm64",
        "docker_server": {
            "Version": "28.0.0",
            "ApiVersion": "1.48",
            "Os": "linux",
            "Arch": "arm64",
        },
        "docker_buildx": "github.com/docker/buildx v0.25.0 deadbeef",
        "images": {
            "biohub-biotraj-wheel": dict(image),
            "reference-biohub-esm": dict(image),
        },
    }
    return {
        "schema_version": 2,
        "contract": contract.contract,
        "contract_sha256": hashlib.sha256(_CONTRACT.read_bytes()).hexdigest(),
        "target": asdict(contract.target),
        "build_container": asdict(contract.container),
        "locks": verify_biohub_reference_lock_contract(_ROOT, _CONTRACT),
        "biotraj": asdict(contract.biotraj),
        "installed_inventory": inventory,
        "installed_inventory_sha256": _canonical_digest(inventory),
        "pip_check": {
            "status": "accepted-platform-exception",
            "returncode": 1,
            "diagnostics": [
                exception.accepted_diagnostic
                for exception in contract.pip_check_platform_exceptions
            ],
            "accepted_platform_exceptions": [
                asdict(exception) for exception in contract.pip_check_platform_exceptions
            ],
        },
        "reference_container_target": "reference-biohub-esm",
        "container_identity": container_identity,
        "container_identity_sha256": _canonical_digest(container_identity),
        "runtime": {
            "operating_system": "linux",
            "architecture": "aarch64",
            "python_implementation": "CPython",
            "python_version": "3.12.11",
            "torch": inventory["torch"],
            "cuda_runtime": "13.0",
            "cuda_driver": "580.65.06",
            "gpu": {
                "name": "NVIDIA GH200 480GB",
                "capability": [9, 0],
                "total_memory_bytes": 480_000_000_000,
            },
            "uname": {
                "system": "Linux",
                "release": "6.8.0",
                "version": "#1 SMP PREEMPT_DYNAMIC",
                "machine": "aarch64",
            },
        },
    }


def test_reference_environment_evidence_is_exact_and_deterministic() -> None:
    payload = _reference_environment_payload()

    assert validate_biohub_reference_environment_evidence(
        payload,
        repository_root=_ROOT,
        contract_path=_CONTRACT,
    ) == {field: payload[field] for field in sorted(payload)}

    ephemeral = copy.deepcopy(payload)
    assert isinstance(ephemeral["runtime"], dict)
    assert isinstance(ephemeral["runtime"]["uname"], dict)
    ephemeral["runtime"]["uname"]["node"] = "container-id"
    with pytest.raises(BiohubReferenceEnvironmentError, match="uname identity"):
        validate_biohub_reference_environment_evidence(
            ephemeral,
            repository_root=_ROOT,
            contract_path=_CONTRACT,
        )

    missing_build_image = copy.deepcopy(payload)
    assert isinstance(missing_build_image["container_identity"], dict)
    images = missing_build_image["container_identity"]["images"]
    assert isinstance(images, dict)
    del images["biohub-biotraj-wheel"]
    missing_build_image["container_identity_sha256"] = _canonical_digest(
        missing_build_image["container_identity"]
    )
    with pytest.raises(BiohubReferenceEnvironmentError, match="Required Biohub"):
        validate_biohub_reference_environment_evidence(
            missing_build_image,
            repository_root=_ROOT,
            contract_path=_CONTRACT,
        )

    drifted_inventory = copy.deepcopy(payload)
    assert isinstance(drifted_inventory["installed_inventory"], dict)
    drifted_inventory["installed_inventory"]["torch"] = "2.13.1+cu130"
    drifted_inventory["installed_inventory_sha256"] = _canonical_digest(
        drifted_inventory["installed_inventory"]
    )
    with pytest.raises(BiohubReferenceEnvironmentError, match="installed_inventory"):
        validate_biohub_reference_environment_evidence(
            drifted_inventory,
            repository_root=_ROOT,
            contract_path=_CONTRACT,
        )

    broadened_pip_check = copy.deepcopy(payload)
    assert isinstance(broadened_pip_check["pip_check"], dict)
    assert isinstance(broadened_pip_check["pip_check"]["diagnostics"], list)
    broadened_pip_check["pip_check"]["diagnostics"].append("rogue dependency failure")
    with pytest.raises(BiohubReferenceEnvironmentError, match="pip_check differs"):
        validate_biohub_reference_environment_evidence(
            broadened_pip_check,
            repository_root=_ROOT,
            contract_path=_CONTRACT,
        )


def test_biohub_environment_image_validation_survives_python_optimized_mode() -> None:
    source = (_ROOT / "tools/remote/biohub_reference_environment.py").read_text(encoding="utf-8")

    assert "assert isinstance(images, Mapping)" not in source
    assert source.count("Reference container identity images must be a mapping.") == 2
