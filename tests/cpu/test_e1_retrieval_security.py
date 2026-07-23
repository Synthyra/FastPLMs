"""Offline security contracts for E1's local MMseqs2 runtime."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from fastplms.models.e1 import retrieval


def _completed(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)


def _inspect_payload(
    *,
    digest: str = retrieval.MMSEQS2_CPU_MANIFEST_DIGEST,
    image_id: str = "sha256:" + "a" * 64,
    architecture: str | None = None,
    repository: str = retrieval.MMSEQS2_IMAGE_REPOSITORY,
) -> str:
    if architecture is None:
        architecture = retrieval._docker_architecture()
    return json.dumps(
        [
            {
                "RepoDigests": [f"{repository}@{digest}"],
                "Id": image_id,
                "Os": "linux",
                "Architecture": architecture,
            }
        ]
    )


def _image_identity() -> retrieval._DockerImageIdentity:
    return retrieval._DockerImageIdentity(
        reference=retrieval.DOCKER_IMAGE,
        repository=retrieval.MMSEQS2_IMAGE_REPOSITORY,
        version=retrieval.MMSEQS2_VERSION,
        manifest_digest=retrieval.MMSEQS2_CPU_MANIFEST_DIGEST,
        image_id="sha256:" + "a" * 64,
        os="linux",
        architecture=retrieval._docker_architecture(),
    )


def test_mmseqs2_default_is_cpu_offline_and_immutable() -> None:
    assert retrieval.DOCKER_IMAGE == (
        "ghcr.io/soedinglab/mmseqs2:18-8cc5c@"
        "sha256:41b12b0d5f41432fa1b9976123da6e2e06e7fab49a34964f3b54ec038e5845d9"
    )
    searcher = retrieval.HomologueSearcher(target_db="target")
    assert searcher.use_gpu is False
    assert searcher.allow_pull is False
    assert searcher.allow_network is False
    assert searcher._docker_base_cmd()[3:5] == ["--network", "none"]

    with pytest.raises(ValueError, match="immutable @sha256"):
        retrieval.HomologueSearcher(
            target_db="target",
            docker_image="ghcr.io/soedinglab/mmseqs2:18-8cc5c",
        )
    with pytest.raises(ValueError, match="CPU-only"):
        retrieval.HomologueSearcher(target_db="target", use_gpu=True)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"target_db": ""},
        {"target_db": "target", "sensitivity": float("nan")},
        {"target_db": "target", "max_seqs": True},
        {"target_db": "target", "min_seq_id": -0.1},
        {"target_db": "target", "coverage": 1.1},
        {"target_db": "target", "phase_timeout": float("inf")},
        {"target_db": "target", "allow_pull": 1},
    ),
)
def test_mmseqs2_constructor_rejects_unsafe_runtime_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        retrieval.HomologueSearcher(**kwargs)


def test_mmseqs2_missing_image_fails_without_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    searcher = retrieval.HomologueSearcher(target_db="target")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            return _completed(command, returncode=1, stderr="not found")
        if command[:2] == ["docker", "pull"]:
            raise AssertionError("allow_pull=False must not pull")
        return _completed(command)

    monkeypatch.setattr(searcher, "_run_docker_command", fake_run)
    with pytest.raises(RuntimeError, match="allow_pull=False"):
        searcher._ensure_docker_image()
    assert not any(command[:2] == ["docker", "pull"] for command in calls)


def test_mmseqs2_explicit_pull_is_reinspected_and_digest_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searcher = retrieval.HomologueSearcher(target_db="target", allow_pull=True)
    inspect_count = 0
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        nonlocal inspect_count
        del kwargs
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            inspect_count += 1
            if inspect_count == 1:
                return _completed(command, returncode=1, stderr="not found")
            return _completed(command, stdout=_inspect_payload())
        return _completed(command)

    monkeypatch.setattr(searcher, "_run_docker_command", fake_run)
    identity = searcher._ensure_docker_image()

    assert identity.manifest_digest == retrieval.MMSEQS2_CPU_MANIFEST_DIGEST
    assert identity.image_id == "sha256:" + "a" * 64
    assert inspect_count == 2
    assert ["docker", "pull", retrieval.DOCKER_IMAGE] in calls


@pytest.mark.parametrize(
    ("payload", "cause"),
    (
        (_inspect_payload(digest="sha256:" + "b" * 64), "RepoDigests"),
        (_inspect_payload(repository="example.invalid/mmseqs2"), "RepoDigests"),
        (_inspect_payload(image_id="mutable-image-id"), "image ID"),
        (_inspect_payload(architecture="s390x"), "architecture"),
    ),
)
def test_mmseqs2_inspect_rejects_wrong_digest_and_image_id(
    payload: str,
    cause: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searcher = retrieval.HomologueSearcher(target_db="target")

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        if command[:3] == ["docker", "image", "inspect"]:
            return _completed(command, stdout=payload)
        return _completed(command)

    monkeypatch.setattr(searcher, "_run_docker_command", fake_run)
    with pytest.raises(RuntimeError) as raised:
        searcher._ensure_docker_image()
    assert cause in str(raised.value.__cause__)


def test_mmseqs2_inspect_preserves_non_missing_docker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searcher = retrieval.HomologueSearcher(target_db="target")

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        if command[:3] == ["docker", "image", "inspect"]:
            return _completed(command, returncode=13, stderr="permission denied")
        return _completed(command)

    monkeypatch.setattr(searcher, "_run_docker_command", fake_run)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        searcher._ensure_docker_image()
    assert raised.value.returncode == 13
    assert raised.value.stderr == "permission denied"


def test_mmseqs2_phase_timeout_preserves_subprocess_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searcher = retrieval.HomologueSearcher(target_db="target", phase_timeout=7.5)
    expired = subprocess.TimeoutExpired(["docker", "run"], timeout=7.5)

    def timeout(*args, **kwargs):
        del args
        assert kwargs["timeout"] == 7.5
        raise expired

    monkeypatch.setattr(retrieval.subprocess, "run", timeout)
    with pytest.raises(TimeoutError, match=r"search.*7\.5") as raised:
        searcher._run_docker_command(["docker", "run"], phase="search", check=True)
    assert raised.value.__cause__ is expired


def test_mmseqs2_realpath_validation_rejects_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working = tmp_path / "working"
    outside = tmp_path / "outside"
    working.mkdir()
    outside.mkdir()
    (working / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.chdir(working)
    searcher = retrieval.HomologueSearcher(target_db="escape/target")

    with pytest.raises(ValueError, match="resolve under"):
        searcher._validate_paths_under_cwd("escape/target")


@pytest.mark.parametrize(
    ("sequence", "seq_id"),
    (
        ("ACD\n>injected", "query"),
        ("ACDEFG", "query\n>injected"),
    ),
)
def test_mmseqs2_rejects_fasta_and_filename_injection_before_docker(
    sequence: str,
    seq_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    searcher = retrieval.HomologueSearcher(target_db="target")
    monkeypatch.setattr(
        searcher,
        "_ensure_docker_image",
        lambda: (_ for _ in ()).throw(AssertionError("Docker must not run")),
    )

    with pytest.raises(ValueError):
        searcher.search(sequence, "results", seq_id=seq_id)


def test_mmseqs2_result_provenance_controls_cache_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    target_prefix = Path("database/target")
    target_prefix.parent.mkdir()
    target_dbtype = Path(f"{target_prefix}.dbtype")
    target_dbtype.write_bytes(b"db-v1")
    searcher = retrieval.HomologueSearcher(
        target_db=str(target_prefix),
        target_db_identity="uniref30-test-revision",
    )
    identity = _image_identity()
    monkeypatch.setattr(searcher, "_ensure_docker_image", lambda: identity)
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(command)
        if "result2msa" in command:
            index = command.index("result2msa")
            output = Path(command[index + 4])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(">query\nACDEFG\n", encoding="utf-8")
        return _completed(command)

    monkeypatch.setattr(searcher, "_run_docker_command", fake_run)
    result = searcher.search("ACDEFG", "results", seq_id="query")
    provenance_path = Path(result).with_name(searcher._PROVENANCE_FILENAME)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    assert provenance["runtime"]["image_id"] == identity.image_id
    assert provenance["runtime"]["manifest_digest"] == identity.manifest_digest
    assert provenance["request"]["target_db"]["identity"] == "uniref30-test-revision"
    assert provenance["cache_identity_sha256"]
    assert all("--network" in command and "none" in command for command in calls)

    calls.clear()
    monkeypatch.setattr(
        searcher,
        "_ensure_docker_image",
        lambda: (_ for _ in ()).throw(AssertionError("valid cache must not inspect Docker")),
    )
    assert searcher.search("ACDEFG", "results", seq_id="query") == result
    assert calls == []

    Path(result).write_text(">query\nTAMPERED\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="valid cache must not inspect Docker"):
        searcher.search("ACDEFG", "results", seq_id="query")


def test_mmseqs2_digest_is_disclosed_in_e1_documentation() -> None:
    documentation = (Path(__file__).resolve().parents[2] / "docs" / "models.md").read_text(
        encoding="utf-8"
    )
    assert retrieval.MMSEQS2_VERSION in documentation
    assert retrieval.MMSEQS2_CPU_MANIFEST_DIGEST in documentation
    assert "allow_pull=False" in documentation
    assert "search-provenance.json" in documentation
