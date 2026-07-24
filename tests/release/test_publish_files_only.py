from __future__ import annotations

import base64
import hashlib
import io
import json
import shutil
import subprocess
import pytest
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from huggingface_hub import CommitOperationAdd, CommitOperationDelete

from fastplms.registry import FileDigest, ModelSpec, get_model_registry
from tools.artifacts import ArtifactError, hash_file
from tools.artifacts import publish as publish_module
from tools.artifacts.build import (
    _RELEASE_TOOL_SCOPE_PATHS,
    _artifact_auto_map,
    _checkpoint_identity_hash,
    _expected_registry_provenance,
    _materialize_model_card,
    _render_artifact_requirements,
    _validated_release_tool_snapshot,
    _validated_runtime_snapshot,
    _write_bootstrap,
    _write_runtime_bundle,
    render_model_card,
)
from tools.artifacts.publish import (
    CompletePublishPlan,
    _is_weight_path,
    _obsolete_registry_pinned_paths,
    _run_required_complete_autoclass_probe,
    _selected_specs,
    _validated_release_text_snapshot,
    main,
    prepare_complete_plan,
    prepare_files_only_plan,
    publish_complete,
    publish_files_only,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def _clean_publication_source(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Exercise publication against a clean snapshot of the current source bytes.

    Publication must reject dirty or untracked release inputs.  The repository under
    test is necessarily dirty while a change is being developed, so using it as the
    positive fixture prevents the tests from reaching the attack they intend to
    exercise.  Commit the bounded publication inputs in an isolated repository and
    point both the artifact builder and publisher at that immutable snapshot instead.

    This fixture does not bypass the production cleanliness checks: every source byte
    consumed below is tracked in the temporary repository, and the production code
    still validates status, membership, revision, and Git-archive bytes itself.
    """

    global ROOT

    original_root = ROOT
    original_publish_file = publish_module.__file__
    source_root = tmp_path_factory.mktemp("clean-publication-source")
    ignored = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    for relative_name in (
        "src/fastplms",
        "model_cards",
        "LICENSES",
        "requirements",
        "tools/artifacts",
        "tools/conversion",
    ):
        source = original_root.joinpath(*relative_name.split("/"))
        destination = source_root.joinpath(*relative_name.split("/"))
        shutil.copytree(source, destination, ignore=ignored)
    for relative_name in (
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "kernels.lock",
        "tools/remote/biohub_reference_environment.py",
        "tools/source_provenance.py",
    ):
        source = original_root / relative_name
        destination = source_root / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    git = ["git", "-c", f"safe.directory={source_root.as_posix()}"]
    subprocess.run(
        [*git, "init", "--initial-branch=main"],
        cwd=source_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [*git, "config", "user.email", "tests@example.invalid"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(
        [*git, "config", "user.name", "FastPLMs Tests"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(
        [*git, "config", "core.autocrlf", "false"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(
        [*git, "config", "commit.gpgsign", "false"],
        cwd=source_root,
        check=True,
    )
    subprocess.run(
        [*git, "config", "core.hooksPath", ".git/disabled-hooks"],
        cwd=source_root,
        check=True,
    )
    subprocess.run([*git, "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [*git, "commit", "-m", "immutable publication fixture"],
        cwd=source_root,
        check=True,
        capture_output=True,
    )

    ROOT = source_root
    publish_module.__file__ = str(source_root / "tools" / "artifacts" / "publish.py")
    try:
        yield
    finally:
        ROOT = original_root
        publish_module.__file__ = original_publish_file


class FakeApi:
    def __init__(self, spec: ModelSpec, *, corrupt_weight: bool = False) -> None:
        siblings: list[SimpleNamespace] = []
        for expected in spec.fast.files:
            digest = (
                "0" * len(expected.digest)
                if corrupt_weight and _is_weight_path(expected.path)
                else expected.digest
            )
            siblings.append(
                SimpleNamespace(
                    rfilename=expected.path,
                    blob_id=digest if expected.algorithm == "git-sha1" else None,
                    lfs={"sha256": digest} if expected.algorithm == "sha256" else None,
                )
            )
        self.info = SimpleNamespace(sha="a" * 40, siblings=siblings)
        self.model_info_calls: list[dict[str, Any]] = []
        self.create_commit_calls: list[dict[str, Any]] = []

    def model_info(self, repo_id: str, **kwargs: Any) -> SimpleNamespace:
        self.model_info_calls.append({"repo_id": repo_id, **kwargs})
        return self.info

    def create_commit(self, **kwargs: Any) -> SimpleNamespace:
        self.create_commit_calls.append(kwargs)
        return SimpleNamespace(
            oid="b" * 40,
            commit_url=f"https://huggingface.co/{kwargs['repo_id']}/commit/{'b' * 40}",
        )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_release_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _self_attest_runtime_mutation(artifact: Path, relative_names: tuple[str, ...]) -> None:
    runtime_attestation_path = artifact / "runtime-attestation.json"
    runtime_attestation = json.loads(runtime_attestation_path.read_text(encoding="utf-8"))
    manifest_path = artifact / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative_name in relative_names:
        path = artifact.joinpath(*relative_name.split("/"))
        digest = f"sha256:{hash_file(path)}"
        runtime_attestation["files"][relative_name] = digest
        manifest[relative_name] = digest
    _write_json(runtime_attestation_path, runtime_attestation)
    manifest["runtime-attestation.json"] = f"sha256:{hash_file(runtime_attestation_path)}"
    _write_json(manifest_path, manifest)


def _rewrite_materialized_card(artifact: Path, spec: ModelSpec) -> None:
    provenance = json.loads((artifact / "provenance.json").read_text(encoding="utf-8"))
    card_source = ROOT / "model_cards" / f"{spec.id}.md"
    card_template = (
        card_source.read_text(encoding="utf-8")
        if card_source.is_file()
        else render_model_card(spec)
    )
    materialized = _materialize_model_card(
        card_template,
        runtime_revision=provenance["runtime_revision"],
        source_tree_sha256=provenance["source_tree_sha256"],
        runtime_bundle_sha256=provenance["runtime_bundle_sha256"],
    )
    (artifact / "README.md").write_text(
        materialized,
        encoding="utf-8",
        newline="\n",
    )


def _initialize_release_text_repository(root: Path, spec: ModelSpec) -> Path:
    for relative_name in _RELEASE_TOOL_SCOPE_PATHS:
        tool_path = root.joinpath(*relative_name.split("/"))
        tool_path.parent.mkdir(parents=True, exist_ok=True)
        tool_path.write_text(f"# immutable test tool: {relative_name}\n", encoding="utf-8")
    (root / "LICENSE").write_text("test license\n", encoding="utf-8")
    (root / "THIRD_PARTY_NOTICES.md").write_text(
        "test notices\n",
        encoding="utf-8",
    )
    registry = get_model_registry()
    for source_id in spec.family.upstreams:
        source = registry.upstreams[source_id]
        for item in source.distribution_files:
            legal_path = root.joinpath(
                "LICENSES",
                source_id,
                *item.path.split("/"),
            )
            legal_path.parent.mkdir(parents=True, exist_ok=True)
            legal_path.write_text("test upstream license\n", encoding="utf-8")
    card = root / "model_cards" / f"{spec.id}.md"
    card.parent.mkdir(parents=True)
    card.write_text("tracked card\n", encoding="utf-8")
    subprocess.run(["git", "init", "--initial-branch=main"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "FastPLMs Tests"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "tracked card"], cwd=root, check=True)
    return card


def _files_only_artifact(root: Path, spec: ModelSpec) -> Path:
    artifact = root / spec.fast.repo_id.split("/", maxsplit=1)[1]
    registry = get_model_registry()
    card_source = ROOT / "model_cards" / f"{spec.id}.md"
    release_tool_revision, release_tool_sha256, release_tool_payloads = (
        _validated_release_tool_snapshot(ROOT)
    )
    safe_files: dict[str, str | bytes] = {
        "README.md": (
            card_source.read_bytes()
            if card_source.is_file()
            else render_model_card(spec).encode()
        ),
        "config.json": "{}\n",
        "fastplms_bundle.py": "",
        "modeling_fastplms.py": "",
        "requirements.txt": _render_artifact_requirements(spec, release_tool_payloads),
        "THIRD_PARTY_NOTICES.md": _canonical_release_bytes(
            ROOT / "THIRD_PARTY_NOTICES.md"
        ),
        "LICENSES/FastPLMs-Apache-2.0.txt": _canonical_release_bytes(ROOT / "LICENSE"),
    }
    for source_id in spec.family.upstreams:
        source = registry.upstreams[source_id]
        for item in source.distribution_files:
            safe_files[f"LICENSES/{source_id}/{item.path}"] = _canonical_release_bytes(
                ROOT.joinpath("LICENSES", source_id, *item.path.split("/"))
            )
    for relative_name, contents in safe_files.items():
        path = artifact.joinpath(*relative_name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, bytes):
            path.write_bytes(contents)
        else:
            path.write_text(contents, encoding="utf-8", newline="\n")

    source_revision, runtime_payloads, source_tree_sha256 = _validated_runtime_snapshot(
        ROOT,
        registry,
        spec,
    )
    packaged_runtime_names: list[str] = []
    for relative_name, payload in runtime_payloads.items():
        packaged_name = f"fastplms/{relative_name}"
        path = artifact.joinpath(*packaged_name.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        packaged_runtime_names.append(packaged_name)
    runtime_bundle_sha256 = _write_runtime_bundle(
        artifact / "fastplms_bundle.py",
        artifact / "fastplms",
    )
    card_template = (
        card_source.read_text(encoding="utf-8")
        if card_source.is_file()
        else render_model_card(spec)
    )
    materialized_card = _materialize_model_card(
        card_template,
        runtime_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        runtime_bundle_sha256=runtime_bundle_sha256,
    )
    (artifact / "README.md").write_text(
        materialized_card,
        encoding="utf-8",
        newline="\n",
    )
    _write_bootstrap(artifact / "modeling_fastplms.py", spec, runtime_bundle_sha256)
    selected_checkpoint = spec.artifact_checkpoint
    _write_json(
        artifact / "config.json",
        {
            "auto_map": _artifact_auto_map(spec),
            "fastplms_model_id": spec.id,
            "fastplms_checkpoint_repo_id": selected_checkpoint.repo_id,
            "fastplms_checkpoint_revision": selected_checkpoint.revision,
            "fastplms_checkpoint_hash": _checkpoint_identity_hash(selected_checkpoint),
            "fastplms_weights_revision": selected_checkpoint.revision,
            "fastplms_runtime_revision": source_revision,
            "fastplms_source_tree_sha256": source_tree_sha256,
            "fastplms_runtime_bundle_sha256": runtime_bundle_sha256,
            "fastplms_release_tool_revision": release_tool_revision,
            "fastplms_release_tool_sha256": release_tool_sha256,
        },
    )
    provenance = {
        **_expected_registry_provenance(registry, spec),
        "model_id": spec.id,
        "runtime_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "release_tool_revision": release_tool_revision,
        "release_tool_sha256": release_tool_sha256,
        "attestations": {
            "complete_artifact": {
                "scope": "weights+runtime",
                "weights_revision": selected_checkpoint.revision,
                "runtime_revision": source_revision,
                "release_tool_revision": release_tool_revision,
                "release_tool_sha256": release_tool_sha256,
                "weights_license_status": (
                    "resolved"
                    if spec.family.weights_publication_allowed
                    else "unresolved"
                ),
                "redistributable": spec.family.weights_publication_allowed,
            },
            "runtime_update": {
                "path": "runtime-attestation.json",
                "scope": "runtime-only",
                "weights_repo_id": spec.fast.repo_id,
                "weights_revision": spec.fast.revision,
                "release_tool_revision": release_tool_revision,
                "release_tool_sha256": release_tool_sha256,
                "weights_license_status": (
                    "resolved"
                    if spec.family.weights_publication_allowed
                    else "unresolved"
                ),
                "redistributable": spec.family.weights_publication_allowed,
            },
        },
        "canonical_weights": {
            "index": "model.safetensors.index.json",
            "shards": {
                "model-00001-of-00001.safetensors": "sha256:" + "1" * 64,
            },
        },
    }
    _write_json(artifact / "provenance.json", provenance)
    (artifact / "model.safetensors.index.json").write_bytes(b"test index")
    (artifact / "model-00001-of-00001.safetensors").write_bytes(b"test shard")
    runtime_files = {
        relative_name: f"sha256:{hash_file(artifact.joinpath(*relative_name.split('/')))}"
        for relative_name in (*safe_files, *packaged_runtime_names)
    }
    runtime_attestation = {
        "schema_version": 2,
        "scope": "runtime-only",
        "model_id": spec.id,
        "weights": {"repo_id": spec.fast.repo_id, "revision": spec.fast.revision},
        "runtime_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "release_tool_revision": release_tool_revision,
        "release_tool_sha256": release_tool_sha256,
        "weights_license_status": (
            "resolved" if spec.family.weights_publication_allowed else "unresolved"
        ),
        "redistributable": spec.family.weights_publication_allowed,
        "files": runtime_files,
    }
    _write_json(artifact / "runtime-attestation.json", runtime_attestation)
    manifest = {
        relative_name: f"sha256:{hash_file(artifact.joinpath(*relative_name.split('/')))}"
        for relative_name in (*safe_files, *packaged_runtime_names)
    }
    manifest.update(
        {
            "provenance.json": f"sha256:{hash_file(artifact / 'provenance.json')}",
            "runtime-attestation.json": (
                f"sha256:{hash_file(artifact / 'runtime-attestation.json')}"
            ),
            "model.safetensors.index.json": (
                f"sha256:{hash_file(artifact / 'model.safetensors.index.json')}"
            ),
            "model-00001-of-00001.safetensors": (
                f"sha256:{hash_file(artifact / 'model-00001-of-00001.safetensors')}"
            ),
        }
    )
    _write_json(artifact / "artifact-manifest.json", manifest)
    return artifact


def test_files_only_plan_excludes_weights_and_complete_artifact_attestations(
    tmp_path: Path,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    api = FakeApi(spec)

    plan = prepare_files_only_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
    )

    assert plan.artifact_path == artifact
    assert plan.repo_id == spec.fast.repo_id
    assert "README.md" in plan.files
    assert "config.json" in plan.files
    assert "fastplms/__init__.py" in plan.files
    assert "requirements.txt" in plan.files
    assert "runtime-attestation.json" in plan.files
    assert "artifact-manifest.json" not in plan.files
    assert "provenance.json" not in plan.files
    assert not any(_is_weight_path(path) for path in plan.files)
    assert api.model_info_calls == [
        {
            "repo_id": spec.fast.repo_id,
            "revision": "main",
            "files_metadata": True,
        }
    ]


@pytest.mark.parametrize(
    ("relative_name", "error_match"),
    (
        ("README.md", "model card differs from the current source tree"),
        (
            "LICENSES/FastPLMs-Apache-2.0.txt",
            "legal text differs from the current source",
        ),
    ),
)
def test_files_only_plan_rejects_self_attested_stale_release_text(
    tmp_path: Path,
    relative_name: str,
    error_match: str,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    forged = artifact.joinpath(*relative_name.split("/"))
    forged.write_bytes(b"self-attested but not current\n")

    manifest = json.loads(
        (artifact / "artifact-manifest.json").read_text(encoding="utf-8")
    )
    forged_digest = f"sha256:{hash_file(forged)}"
    manifest[relative_name] = forged_digest
    runtime_attestation = json.loads(
        (artifact / "runtime-attestation.json").read_text(encoding="utf-8")
    )
    runtime_attestation["files"][relative_name] = forged_digest
    _write_json(artifact / "runtime-attestation.json", runtime_attestation)
    manifest["runtime-attestation.json"] = (
        f"sha256:{hash_file(artifact / 'runtime-attestation.json')}"
    )
    _write_json(artifact / "artifact-manifest.json", manifest)

    with pytest.raises(ArtifactError, match=error_match):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_files_only_plan_rejects_self_attested_dependency_change(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    requirements = artifact / "requirements.txt"
    requirements.write_text("fastplms @ git+https://example.invalid/fastplms.git\n")
    _self_attest_runtime_mutation(artifact, ("requirements.txt",))

    with pytest.raises(ArtifactError, match="direct dependency contract"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_files_only_plan_rejects_self_attested_card_runtime_placeholder(
    tmp_path: Path,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    readme = artifact / "README.md"
    text = readme.read_text(encoding="utf-8")
    readme.write_text(
        text.replace("requirements.txt", "requirements.txt@<runtime-revision>", 1),
        encoding="utf-8",
        newline="\n",
    )
    _self_attest_runtime_mutation(artifact, ("README.md",))

    with pytest.raises(ArtifactError, match="unresolved runtime-revision placeholder"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_release_snapshot_rejects_tracked_card_deleted_from_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    source_root = tmp_path / "source"
    source_root.mkdir()
    card = _initialize_release_text_repository(source_root, spec)
    card.unlink()
    monkeypatch.setattr(
        "tools.artifacts.publish.__file__",
        str(source_root / "tools" / "artifacts" / "publish.py"),
    )

    with pytest.raises(ArtifactError, match="regular non-symlink file"):
        _validated_release_text_snapshot(
            spec,
            get_model_registry(),
            runtime_revision="a" * 40,
            source_tree_sha256="b" * 64,
            runtime_bundle_sha256="c" * 64,
        )


def test_release_snapshot_rejects_untracked_card_symlink_to_tracked_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    source_root = tmp_path / "source"
    source_root.mkdir()
    tracked_card = _initialize_release_text_repository(source_root, spec)
    tracked_target = source_root / "tracked-target.md"
    tracked_target.write_text(tracked_card.read_text(encoding="utf-8"), encoding="utf-8")
    subprocess.run(["git", "add", tracked_target.name], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "-m", "tracked target"], cwd=source_root, check=True)
    tracked_card.unlink()
    subprocess.run(["git", "add", "-u"], cwd=source_root, check=True)
    subprocess.run(["git", "commit", "-m", "remove declared card"], cwd=source_root, check=True)
    try:
        tracked_card.symlink_to(tracked_target)
    except OSError:
        tracked_card.write_text("symlink substitute\n", encoding="utf-8")
        original_is_symlink = Path.is_symlink
        monkeypatch.setattr(
            Path,
            "is_symlink",
            lambda path: path == tracked_card or original_is_symlink(path),
        )
    monkeypatch.setattr(
        "tools.artifacts.publish.__file__",
        str(source_root / "tools" / "artifacts" / "publish.py"),
    )

    with pytest.raises(ArtifactError, match="untracked at the validated revision"):
        _validated_release_text_snapshot(
            spec,
            get_model_registry(),
            runtime_revision="a" * 40,
            source_tree_sha256="b" * 64,
            runtime_bundle_sha256="c" * 64,
        )


def test_files_only_plan_rejects_self_attested_invalid_bundle_data(
    tmp_path: Path,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    provenance = json.loads((artifact / "provenance.json").read_text(encoding="utf-8"))
    (artifact / "fastplms_bundle.py").write_text(
        "\n".join(
            (
                '"""Forged runtime data."""',
                "",
                f'RUNTIME_HASH = "{provenance["runtime_bundle_sha256"]}"',
                'RUNTIME_DATA = "☃"',
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    _self_attest_runtime_mutation(artifact, ("fastplms_bundle.py",))

    with pytest.raises(ArtifactError, match="invalid base85 archive data"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_files_only_plan_rejects_self_attested_substituted_bundle_member(
    tmp_path: Path,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    runtime_root = artifact / "fastplms"
    files = {
        f"fastplms/{path.relative_to(runtime_root).as_posix()}": path.read_bytes()
        for path in runtime_root.rglob("*")
        if path.is_file()
    }
    target = sorted(files)[0]
    substituted = bytearray(files[target])
    if not substituted:
        raise AssertionError(f"Runtime substitution target is unexpectedly empty: {target}")
    substituted[0] ^= 0x01
    files[target] = bytes(substituted)
    buffer = io.BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for relative_name, payload in sorted(files.items()):
            info = ZipInfo(relative_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload, compress_type=ZIP_DEFLATED, compresslevel=9)
    archive_bytes = buffer.getvalue()
    runtime_hash = hashlib.sha256(archive_bytes).hexdigest()
    encoded = base64.b85encode(archive_bytes).decode("ascii")
    chunks = [encoded[index : index + 100] for index in range(0, len(encoded), 100)]
    (artifact / "fastplms_bundle.py").write_text(
        "\n".join(
            (
                '"""Generated deterministic archive of unchanged FastPLMs runtime sources."""',
                "",
                f'RUNTIME_HASH = "{runtime_hash}"',
                "RUNTIME_DATA = (",
                *(f"    {chunk!r}" for chunk in chunks),
                ")",
                "",
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    provenance_path = artifact / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime_bundle_sha256"] = runtime_hash
    _write_json(provenance_path, provenance)
    config_path = artifact / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["fastplms_runtime_bundle_sha256"] = runtime_hash
    _write_json(config_path, config)
    runtime_attestation_path = artifact / "runtime-attestation.json"
    runtime_attestation = json.loads(runtime_attestation_path.read_text(encoding="utf-8"))
    runtime_attestation["runtime_bundle_sha256"] = runtime_hash
    _write_json(runtime_attestation_path, runtime_attestation)
    _write_bootstrap(artifact / "modeling_fastplms.py", spec, runtime_hash)
    _rewrite_materialized_card(artifact, spec)
    _self_attest_runtime_mutation(
        artifact,
        ("README.md", "config.json", "fastplms_bundle.py", "modeling_fastplms.py"),
    )
    manifest_path = artifact / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance.json"] = f"sha256:{hash_file(provenance_path)}"
    _write_json(manifest_path, manifest)

    with pytest.raises(ArtifactError, match="archive differs at"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_files_only_plan_rejects_self_attested_modified_bootstrap(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    bootstrap = artifact / "modeling_fastplms.py"
    bootstrap.write_text(
        bootstrap.read_text(encoding="utf-8") + "\nUNAPPROVED_CODE = True\n",
        encoding="utf-8",
        newline="\n",
    )
    _self_attest_runtime_mutation(artifact, ("modeling_fastplms.py",))

    with pytest.raises(ArtifactError, match="bootstrap differs"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("model_id", tuple(get_model_registry()))
def test_files_only_plan_supports_every_manifest_model(
    tmp_path: Path,
    model_id: str,
) -> None:
    spec = get_model_registry()[model_id]
    _files_only_artifact(tmp_path, spec)
    api = FakeApi(spec)

    if spec.family.requires_complete_weight_publication:
        with pytest.raises(ArtifactError, match="complete weights-plus-runtime"):
            prepare_files_only_plan(
                spec,
                artifact_root=tmp_path,
                revision="main",
                api=api,  # type: ignore[arg-type]
            )
        return

    plan = prepare_files_only_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
    )
    assert plan.model_id == model_id
    assert plan.repo_id == spec.fast.repo_id
    assert not any(_is_weight_path(path) for path in plan.files)


def test_files_only_publish_uses_additions_and_parent_commit_only(
    tmp_path: Path,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    _files_only_artifact(tmp_path, spec)
    api = FakeApi(spec)
    plan = prepare_files_only_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
    )

    results = publish_files_only(
        (plan,),
        api=api,  # type: ignore[arg-type]
        commit_message="Update runtime files",
    )

    assert results[0].commit_oid == "b" * 40
    assert len(api.create_commit_calls) == 1
    call = api.create_commit_calls[0]
    assert call["repo_id"] == spec.fast.repo_id
    assert call["repo_type"] == "model"
    assert call["revision"] == "main"
    assert call["parent_commit"] == "a" * 40
    assert all(isinstance(operation, CommitOperationAdd) for operation in call["operations"])
    assert {operation.path_in_repo for operation in call["operations"]} == set(plan.files)
    assert not any(_is_weight_path(operation.path_in_repo) for operation in call["operations"])


def test_files_only_dry_run_performs_no_commit(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    _files_only_artifact(tmp_path, spec)
    api = FakeApi(spec)
    plan = prepare_files_only_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
    )

    assert (
        publish_files_only(
            (plan,),
            api=api,  # type: ignore[arg-type]
            commit_message="Unused",
            dry_run=True,
        )
        == ()
    )
    assert api.create_commit_calls == []


def test_files_only_rejects_local_model_identity_mismatch(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    config = {"fastplms_model_id": "esm2_35m"}
    _write_json(artifact / "config.json", config)
    manifest = json.loads((artifact / "artifact-manifest.json").read_text(encoding="utf-8"))
    manifest["config.json"] = f"sha256:{hash_file(artifact / 'config.json')}"
    _write_json(artifact / "artifact-manifest.json", manifest)

    with pytest.raises(ArtifactError, match="fastplms_model_id"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_files_only_rejects_remote_weight_identity_mismatch(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    _files_only_artifact(tmp_path, spec)

    with pytest.raises(ArtifactError, match="Hub weight identity differs"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec, corrupt_weight=True),  # type: ignore[arg-type]
        )


def test_files_only_selection_defaults_to_every_manifest_model() -> None:
    registry = get_model_registry()
    assert [spec.id for spec in _selected_specs(registry, ("esm2_8m",), all_models=False)] == [
        "esm2_8m"
    ]
    expected = list(registry)
    assert [spec.id for spec in _selected_specs(registry, (), all_models=False)] == expected
    assert [spec.id for spec in _selected_specs(registry, (), all_models=True)] == expected
    with pytest.raises(ArtifactError, match="not both"):
        _selected_specs(registry, ("esm2_8m",), all_models=True)
    with pytest.raises(ArtifactError, match="Unknown model IDs"):
        _selected_specs(registry, ("not-a-model",), all_models=False)


def test_publish_cli_requires_explicit_files_only_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["publish", "esm2_8m"])
    with pytest.raises(SystemExit, match="explicit --files-only or --complete"):
        main()


def test_complete_publish_cli_requires_explicit_model_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["publish", "--complete"])
    with pytest.raises(SystemExit, match="requires explicit model IDs"):
        main()


def test_files_only_plan_rejects_unlisted_and_sensitive_files(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    (artifact / "token.txt").write_text("must never be uploaded", encoding="utf-8")

    with pytest.raises(ArtifactError, match="sensitive publication path"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_files_only_plan_rejects_unknown_manifest_path(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    unknown = artifact / "helper.exe"
    unknown.write_bytes(b"binary")
    manifest = json.loads((artifact / "artifact-manifest.json").read_text(encoding="utf-8"))
    manifest["helper.exe"] = f"sha256:{hash_file(unknown)}"
    _write_json(artifact / "artifact-manifest.json", manifest)

    with pytest.raises(ArtifactError, match="outside the publication allowlist"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_files_only_plan_rejects_undeclared_legal_path(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    undeclared = artifact / "LICENSES" / "UNDECLARED.md"
    undeclared.write_text("must not be published", encoding="utf-8")
    manifest = json.loads((artifact / "artifact-manifest.json").read_text(encoding="utf-8"))
    manifest["LICENSES/UNDECLARED.md"] = f"sha256:{hash_file(undeclared)}"
    _write_json(artifact / "artifact-manifest.json", manifest)

    with pytest.raises(ArtifactError, match="outside the publication allowlist"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_files_only_plan_rejects_unlisted_regular_file(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    (artifact / "notes.md").write_text("not in the manifest", encoding="utf-8")

    with pytest.raises(ArtifactError, match="file inventory differs"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_files_only_plan_rejects_forged_registry_provenance(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    provenance_path = artifact / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["fast_checkpoint"]["revision"] = "f" * 40
    _write_json(provenance_path, provenance)
    manifest = json.loads((artifact / "artifact-manifest.json").read_text(encoding="utf-8"))
    manifest["provenance.json"] = f"sha256:{hash_file(provenance_path)}"
    _write_json(artifact / "artifact-manifest.json", manifest)

    with pytest.raises(ArtifactError, match="current registry"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_files_only_plan_rejects_stale_runtime_revision(tmp_path: Path) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    stale = "e" * 40
    provenance_path = artifact / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime_revision"] = stale
    _write_json(provenance_path, provenance)
    config_path = artifact / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["fastplms_runtime_revision"] = stale
    _write_json(config_path, config)
    attestation_path = artifact / "runtime-attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["runtime_revision"] = stale
    _write_json(attestation_path, attestation)
    _rewrite_materialized_card(artifact, spec)
    _self_attest_runtime_mutation(artifact, ("README.md", "config.json"))
    manifest = json.loads((artifact / "artifact-manifest.json").read_text(encoding="utf-8"))
    manifest["provenance.json"] = f"sha256:{hash_file(provenance_path)}"
    _write_json(artifact / "artifact-manifest.json", manifest)

    with pytest.raises(ArtifactError, match="current clean source revision"):
        prepare_files_only_plan(
            spec,
            artifact_root=tmp_path,
            revision="main",
            api=FakeApi(spec),  # type: ignore[arg-type]
        )


def test_files_only_publish_uses_preflighted_bytes_after_local_mutation(
    tmp_path: Path,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    artifact = _files_only_artifact(tmp_path, spec)
    api = FakeApi(spec)
    plan = prepare_files_only_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
    )
    expected = (artifact / "README.md").read_bytes()
    (artifact / "README.md").write_text("mutated after preflight", encoding="utf-8")

    publish_files_only(
        (plan,),
        api=api,  # type: ignore[arg-type]
        commit_message="Runtime update",
    )

    operations = api.create_commit_calls[0]["operations"]
    readme = next(operation for operation in operations if operation.path_in_repo == "README.md")
    assert readme.path_or_fileobj.getvalue() == expected


def test_files_only_publish_rejects_source_change_after_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    _files_only_artifact(tmp_path, spec)
    api = FakeApi(spec)
    plan = prepare_files_only_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        "tools.artifacts.publish._validated_runtime_snapshot",
        lambda *_: (plan.runtime_revision, {}, "0" * 64),
    )

    with pytest.raises(ArtifactError, match="changed after publication preflight"):
        publish_files_only(
            (plan,),
            api=api,  # type: ignore[arg-type]
            commit_message="Must fail",
        )
    assert not api.create_commit_calls


def test_files_only_publish_rejects_release_text_change_after_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    _files_only_artifact(tmp_path, spec)
    api = FakeApi(spec)
    plan = prepare_files_only_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        "tools.artifacts.publish._validated_release_text_snapshot",
        lambda *_, **__: (plan.release_revision, "0" * 64, {}),
    )

    with pytest.raises(ArtifactError, match="release texts changed after publication preflight"):
        publish_files_only(
            (plan,),
            api=api,  # type: ignore[arg-type]
            commit_message="Must fail",
        )
    assert not api.create_commit_calls


def test_files_only_publish_rejects_release_tool_change_after_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    _files_only_artifact(tmp_path, spec)
    api = FakeApi(spec)
    plan = prepare_files_only_plan(
        spec,
        artifact_root=tmp_path,
        revision="main",
        api=api,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        "tools.artifacts.publish._validated_release_tool_snapshot",
        lambda *_: ("f" * 40, "e" * 64, {}),
    )

    with pytest.raises(ArtifactError, match="Release tools changed after publication preflight"):
        publish_files_only(
            (plan,),
            api=api,  # type: ignore[arg-type]
            commit_message="Must fail",
        )
    assert not api.create_commit_calls


def test_complete_publish_rehashes_every_file_before_atomic_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    first = artifact / "config.json"
    second = artifact / "model.safetensors"
    first.write_bytes(b"config")
    second.write_bytes(b"weights")
    plan = CompletePublishPlan(
        model_id="toy",
        repo_id="Synthyra/Toy",
        revision="main",
        parent_commit="a" * 40,
        artifact_path=artifact,
        files=("config.json", "model.safetensors"),
        digests=(
            ("config.json", f"sha256:{hash_file(first)}"),
            ("model.safetensors", f"sha256:{hash_file(second)}"),
        ),
    )
    api = FakeApi(get_model_registry()["esm2_8m"])
    monkeypatch.setattr(publish_module, "_revalidate_complete_plan", lambda *_: None)

    results = publish_complete(
        (plan,),
        api=api,  # type: ignore[arg-type]
        commit_message="Complete update",
    )

    assert len(results) == 1
    assert len(api.create_commit_calls) == 1
    call = api.create_commit_calls[0]
    assert call["parent_commit"] == "a" * 40
    assert {operation.path_in_repo for operation in call["operations"]} == set(plan.files)

    first.write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="digest differs"):
        publish_complete(
            (plan,),
            api=api,  # type: ignore[arg-type]
            commit_message="Must fail",
        )
    assert len(api.create_commit_calls) == 1


def test_complete_publish_rejects_hand_built_unknown_plan(tmp_path: Path) -> None:
    artifact = tmp_path / "forged-artifact"
    artifact.mkdir()
    plan = CompletePublishPlan(
        model_id="unregistered_model",
        repo_id="Synthyra/UnregisteredModel",
        revision="main",
        parent_commit="a" * 40,
        artifact_path=artifact,
        files=(),
        digests=(),
    )
    api = FakeApi(get_model_registry()["esm2_8m"])

    with pytest.raises(ArtifactError, match="absent from the current registry"):
        publish_complete(
            (plan,),
            api=api,  # type: ignore[arg-type]
            commit_message="Must fail",
        )
    assert not api.create_commit_calls


def test_complete_plan_deletes_only_superseded_pinned_monolith() -> None:
    spec = get_model_registry()["esm2_8m"]
    new_inventory = {
        item.path for item in spec.fast.files if item.path != "model.safetensors"
    }
    new_inventory.update(
        {
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        }
    )

    assert _obsolete_registry_pinned_paths(
        spec,
        FakeApi(spec).info,
        new_inventory,
    ) == ("model.safetensors",)


def test_complete_plan_deletes_superseded_pinned_shards_and_index() -> None:
    spec = get_model_registry()["esm2_8m"]
    legacy_index = FileDigest.parse(
        "model.safetensors.index.json=git-sha1:" + "d" * 40
    )
    legacy_shard = FileDigest.parse(
        "model-00001-of-00001.safetensors=sha256:" + "e" * 64
    )
    legacy = replace(
        spec,
        fast=replace(spec.fast, files=(*spec.fast.files, legacy_index, legacy_shard)),
    )
    new_inventory = {item.path for item in spec.fast.files}

    assert _obsolete_registry_pinned_paths(
        legacy,
        FakeApi(legacy).info,
        new_inventory,
    ) == (legacy_shard.path, legacy_index.path)


@pytest.mark.parametrize(
    "relative_name",
    (
        "pytorch_model.bin",
        "alternate.safetensors",
        "stale/model-00001-of-00002.safetensors",
        "stale/pytorch_model.bin.index.json",
    ),
)
def test_complete_plan_rejects_unpinned_competing_remote_weight(
    relative_name: str,
) -> None:
    spec = get_model_registry()["esm2_8m"]
    api = FakeApi(spec)
    api.info.siblings.append(
        SimpleNamespace(
            rfilename=relative_name,
            blob_id=None,
            lfs={"sha256": "f" * 64},
        )
    )
    new_inventory = {
        "model.safetensors.index.json",
        "model-00001-of-00001.safetensors",
    }

    with pytest.raises(ArtifactError, match="unpinned competing weight files"):
        _obsolete_registry_pinned_paths(spec, api.info, new_inventory)


def test_complete_publish_rejects_arbitrary_delete_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    config = artifact / "config.json"
    config.write_bytes(b"config")
    index = artifact / "model.safetensors.index.json"
    shard = artifact / "model-00001-of-00001.safetensors"
    index.write_bytes(b"index")
    shard.write_bytes(b"replacement")
    spec = get_model_registry()["esm2_8m"]
    files = (config.name, index.name, shard.name)
    plan = CompletePublishPlan(
        model_id=spec.id,
        repo_id=spec.fast.repo_id,
        revision="main",
        parent_commit="a" * 40,
        artifact_path=artifact,
        files=files,
        digests=tuple(
            (name, f"sha256:{hash_file(artifact / name)}") for name in files
        ),
        deletes=("unrelated-user-file.txt",),
        replacement_weight_paths=(index.name, shard.name),
    )
    api = FakeApi(spec)
    monkeypatch.setattr(publish_module, "_revalidate_complete_plan", lambda *_: spec)

    with pytest.raises(ArtifactError, match="unproven obsolete weight delete"):
        publish_complete(
            (plan,),
            api=api,  # type: ignore[arg-type]
            commit_message="Must fail",
        )
    assert not api.create_commit_calls


def test_complete_publish_rejects_config_only_weight_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    config = artifact / "config.json"
    config.write_bytes(b"config")
    spec = get_model_registry()["esm2_8m"]
    plan = CompletePublishPlan(
        model_id=spec.id,
        repo_id=spec.fast.repo_id,
        revision="main",
        parent_commit="a" * 40,
        artifact_path=artifact,
        files=("config.json",),
        digests=(("config.json", f"sha256:{hash_file(config)}"),),
        deletes=("model.safetensors",),
    )
    api = FakeApi(spec)
    monkeypatch.setattr(publish_module, "_revalidate_complete_plan", lambda *_: spec)

    with pytest.raises(ArtifactError, match="canonical replacement weight set"):
        publish_complete(
            (plan,),
            api=api,  # type: ignore[arg-type]
            commit_message="Must fail",
        )
    assert not api.create_commit_calls


def test_complete_publish_includes_proven_guarded_weight_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    index = artifact / "model.safetensors.index.json"
    shard = artifact / "model-00001-of-00001.safetensors"
    index.write_bytes(b"index")
    shard.write_bytes(b"replacement")
    spec = get_model_registry()["esm2_8m"]
    files = (index.name, shard.name)
    plan = CompletePublishPlan(
        model_id=spec.id,
        repo_id=spec.fast.repo_id,
        revision="main",
        parent_commit="a" * 40,
        artifact_path=artifact,
        files=files,
        digests=tuple(
            (name, f"sha256:{hash_file(artifact / name)}") for name in files
        ),
        deletes=("model.safetensors",),
        replacement_weight_paths=files,
    )
    api = FakeApi(spec)
    monkeypatch.setattr(publish_module, "_revalidate_complete_plan", lambda *_: spec)

    publish_complete(
        (plan,),
        api=api,  # type: ignore[arg-type]
        commit_message="Atomic migration",
    )

    operations = api.create_commit_calls[0]["operations"]
    assert any(isinstance(operation, CommitOperationAdd) for operation in operations)
    assert [
        operation.path_in_repo
        for operation in operations
        if isinstance(operation, CommitOperationDelete)
    ] == ["model.safetensors"]


def test_complete_publish_uploads_validated_snapshot_after_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    weights = artifact / "model.safetensors"
    original = b"validated-weight-payload"
    weights.write_bytes(original)
    spec = get_model_registry()["esm2_8m"]
    plan = CompletePublishPlan(
        model_id="toy",
        repo_id="Synthyra/Toy",
        revision="main",
        parent_commit="a" * 40,
        artifact_path=artifact,
        files=("model.safetensors",),
        digests=(("model.safetensors", f"sha256:{hash_file(weights)}"),),
    )

    class MutatingApi(FakeApi):
        uploaded: bytes | None = None

        def create_commit(self, **kwargs: Any) -> SimpleNamespace:
            weights.write_bytes(b"mutated-in-place-after-validation")
            operation = next(
                item
                for item in kwargs["operations"]
                if isinstance(item, CommitOperationAdd)
            )
            operation.path_or_fileobj.seek(0)
            self.uploaded = operation.path_or_fileobj.read()
            return super().create_commit(**kwargs)

    monkeypatch.setattr("tools.artifacts.publish._MAX_RETAINED_COMPLETE_BYTES", 1)
    monkeypatch.setattr(publish_module, "_revalidate_complete_plan", lambda *_: None)
    api = MutatingApi(spec)

    publish_complete(
        (plan,),
        api=api,  # type: ignore[arg-type]
        commit_message="Frozen snapshot",
    )

    assert api.uploaded == original


def test_required_complete_probe_groups_ankh_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_spec = get_model_registry()["ankh_base"]
    spec = replace(
        current_spec,
        family=replace(
            current_spec.family,
            requires_complete_weight_publication=True,
        ),
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        cases_path = Path(command[command.index("--cases-file") + 1])
        output_path = Path(command[command.index("--output") + 1])
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        assert [case["auto_class"] for case in cases] == [
            "AutoModel",
            "AutoModelForSeq2SeqLM",
        ]
        output_path.write_text(
            json.dumps({case["auto_class"]: {"state": "ok"} for case in cases}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("tools.artifacts.publish.subprocess.run", fake_run)

    assert _run_required_complete_autoclass_probe(spec, artifact) == (
        "AutoModel",
        "AutoModelForSeq2SeqLM",
    )


def test_complete_ankh_publish_rejects_missing_probe_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_spec = get_model_registry()["ankh_base"]
    spec = replace(
        current_spec,
        family=replace(
            current_spec.family,
            requires_complete_weight_publication=True,
        ),
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = artifact / "artifact-manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    index = artifact / "model.safetensors.index.json"
    shard = artifact / "model-00001-of-00001.safetensors"
    index.write_bytes(b"index")
    shard.write_bytes(b"replacement")
    files = (manifest.name, index.name, shard.name)
    plan = CompletePublishPlan(
        model_id=spec.id,
        repo_id=spec.fast.repo_id,
        revision="main",
        parent_commit="a" * 40,
        artifact_path=artifact,
        files=files,
        digests=tuple(
            (name, f"sha256:{hash_file(artifact / name)}") for name in files
        ),
        replacement_weight_paths=(index.name, shard.name),
    )
    api = FakeApi(spec)
    monkeypatch.setattr(publish_module, "_revalidate_complete_plan", lambda *_: spec)

    with pytest.raises(ArtifactError, match="required AutoClass probe"):
        publish_complete(
            (plan,),
            api=api,  # type: ignore[arg-type]
            commit_message="Must fail",
        )
    assert not api.create_commit_calls


def test_complete_publication_rejects_synthetic_unresolved_checkpoint_license(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = get_model_registry()["dplm_150m"]
    unresolved_family = replace(
        spec.family,
        checkpoint_license="Unresolved synthetic checkpoint terms",
        hub_license="other",
        hub_license_name="Synthetic checkpoint license unresolved",
        hub_license_link="https://example.invalid/checkpoint-license",
        weights_publication_allowed=False,
    )
    unresolved_spec = replace(spec, family=unresolved_family)
    monkeypatch.setattr(
        publish_module,
        "get_model_registry",
        lambda: {unresolved_spec.id: unresolved_spec},
    )
    api = FakeApi(unresolved_spec)

    with pytest.raises(ArtifactError, match="unresolved weight-license"):
        prepare_complete_plan(
            unresolved_spec,
            artifact_root=tmp_path,
            revision="main",
            api=api,  # type: ignore[arg-type]
        )
    assert not api.model_info_calls
    assert not api.create_commit_calls
