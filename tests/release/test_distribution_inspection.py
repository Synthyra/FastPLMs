"""Fail-closed contracts for wheel and source-distribution inspection."""

from __future__ import annotations

import io
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.remote import distribution_inspect
from tools.remote.distribution_inspect import (
    DistributionInspectionError,
    inspect_sdist,
    inspect_wheel,
)

_DIST_INFO = "fastplms-1.0.0.dist-info"


def _source_files() -> dict[str, bytes]:
    return {
        "src/fastplms/__init__.py": b"__version__ = '1.0.0'\n",
        "src/fastplms/models.toml": b"schema_version = 1\n",
        "pyproject.toml": b"[project]\nname = 'fastplms'\nversion = '1.0.0'\n",
        "README.md": b"# FastPLMs\n",
        "kernels.lock": b"[]\n",
        "LICENSE": b"FastPLMs license\n",
        "LICENSES/README.md": b"Legal inventory\n",
        "LICENSES/toy/LICENSE": b"Toy license\n",
        "THIRD_PARTY_NOTICES.md": b"Third-party notices\n",
    }


def _source_snapshot(source_files: dict[str, bytes]) -> distribution_inspect.SourceSnapshot:
    return ("0" * 40, source_files, "1" * 64)


def _wheel_members(source_files: dict[str, bytes]) -> dict[str, bytes]:
    legal_files = {
        relative_name: payload
        for relative_name, payload in source_files.items()
        if relative_name in {"LICENSE", "THIRD_PARTY_NOTICES.md"}
        or relative_name.startswith("LICENSES/")
    }
    metadata = "\n".join(
        [
            "Metadata-Version: 2.4",
            "Name: fastplms",
            "Version: 1.0.0",
            *(f"License-File: {relative_name}" for relative_name in sorted(legal_files)),
            "",
        ]
    ).encode()
    members = {
        "fastplms/__init__.py": source_files["src/fastplms/__init__.py"],
        "fastplms/models.toml": source_files["src/fastplms/models.toml"],
        f"{_DIST_INFO}/METADATA": metadata,
        f"{_DIST_INFO}/RECORD": b"",
        f"{_DIST_INFO}/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{_DIST_INFO}/kernels.lock": source_files["kernels.lock"],
    }
    members.update(
        {
            f"{_DIST_INFO}/licenses/{relative_name}": payload
            for relative_name, payload in legal_files.items()
        }
    )
    return members


def _write_wheel(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for relative_name, payload in sorted(members.items()):
            archive.writestr(relative_name, payload)


def _write_sdist(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        root = tarfile.TarInfo("fastplms-1.0.0")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        for relative_name, payload in sorted(members.items()):
            member = tarfile.TarInfo(f"fastplms-1.0.0/{relative_name}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


def test_complete_tracked_legal_inventory_is_accepted_in_both_distributions(
    tmp_path: Path,
) -> None:
    source_files = _source_files()
    wheel = tmp_path / "fastplms-1.0.0-py3-none-any.whl"
    sdist = tmp_path / "fastplms-1.0.0.tar.gz"
    _write_wheel(wheel, _wheel_members(source_files))
    _write_sdist(sdist, source_files)

    assert inspect_wheel(
        wheel,
        project_root=tmp_path,
        _source_snapshot=_source_snapshot(source_files),
    )["kind"] == "wheel"
    assert inspect_sdist(
        sdist,
        project_root=tmp_path,
        _source_snapshot=_source_snapshot(source_files),
    )["kind"] == "sdist"


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("missing", "Wheel legal inventory differs"),
        ("extra", "Wheel legal inventory differs"),
        ("mutated", "Wheel legal members differ"),
    ),
)
def test_wheel_inspection_rejects_inexact_tracked_legal_inventory(
    tmp_path: Path,
    change: str,
    message: str,
) -> None:
    source_files = _source_files()
    wheel = tmp_path / "fastplms-1.0.0-py3-none-any.whl"
    members = _wheel_members(source_files)
    nested_license = f"{_DIST_INFO}/licenses/LICENSES/toy/LICENSE"
    if change == "missing":
        members.pop(nested_license)
    elif change == "extra":
        members[f"{_DIST_INFO}/licenses/LICENSES/extra/NOTICE"] = b"undeclared\n"
    else:
        members[nested_license] = b"mutated\n"
    _write_wheel(wheel, members)

    with pytest.raises(DistributionInspectionError, match=message):
        inspect_wheel(
            wheel,
            project_root=tmp_path,
            _source_snapshot=_source_snapshot(source_files),
        )


def test_wheel_inspection_rejects_incomplete_license_file_metadata(tmp_path: Path) -> None:
    source_files = _source_files()
    wheel = tmp_path / "fastplms-1.0.0-py3-none-any.whl"
    members = _wheel_members(source_files)
    metadata_name = f"{_DIST_INFO}/METADATA"
    members[metadata_name] = members[metadata_name].replace(
        b"License-File: LICENSES/toy/LICENSE\n",
        b"",
    )
    _write_wheel(wheel, members)

    with pytest.raises(DistributionInspectionError, match="METADATA License-File"):
        inspect_wheel(
            wheel,
            project_root=tmp_path,
            _source_snapshot=_source_snapshot(source_files),
        )


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("missing", "Source-distribution legal inventory differs"),
        ("extra", "Source-distribution legal inventory differs"),
        ("mutated", "Source-distribution legal members differ"),
    ),
)
def test_sdist_inspection_rejects_inexact_tracked_legal_inventory(
    tmp_path: Path,
    change: str,
    message: str,
) -> None:
    source_files = _source_files()
    sdist = tmp_path / "fastplms-1.0.0.tar.gz"
    members = dict(source_files)
    nested_license = "LICENSES/toy/LICENSE"
    if change == "missing":
        members.pop(nested_license)
    elif change == "extra":
        members["LICENSES/extra/NOTICE"] = b"undeclared\n"
    else:
        members[nested_license] = b"mutated\n"
    _write_sdist(sdist, members)

    with pytest.raises(DistributionInspectionError, match=message):
        inspect_sdist(
            sdist,
            project_root=tmp_path,
            _source_snapshot=_source_snapshot(source_files),
        )


def test_wheel_inspection_rejects_path_escape(tmp_path: Path) -> None:
    wheel = tmp_path / "fastplms-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("../credentials.json", b"secret")

    with pytest.raises(DistributionInspectionError, match="Non-portable"):
        inspect_wheel(wheel, project_root=tmp_path)


def test_wheel_inspection_rejects_symlink(tmp_path: Path) -> None:
    wheel = tmp_path / "fastplms-1.0.0-py3-none-any.whl"
    info = zipfile.ZipInfo("fastplms/injected.py")
    info.create_system = 3
    info.external_attr = (0o120777 << 16) | 0xA000
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(info, "../../outside.py")

    with pytest.raises(DistributionInspectionError, match="symlink"):
        inspect_wheel(wheel, project_root=tmp_path)


def test_wheel_inspection_rejects_native_or_non_package_payloads(tmp_path: Path) -> None:
    wheel = tmp_path / "fastplms-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fastplms/injected.so", b"native")

    with pytest.raises(DistributionInspectionError, match="Native binary"):
        inspect_wheel(wheel, project_root=tmp_path)

    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("unrelated/payload.txt", b"unexpected")

    with pytest.raises(DistributionInspectionError, match="package allowlist"):
        inspect_wheel(wheel, project_root=tmp_path)

    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fastplms/checkpoint.safetensors", b"weights")

    with pytest.raises(DistributionInspectionError, match="serialized binary"):
        inspect_wheel(wheel, project_root=tmp_path)


def test_wheel_inspection_requires_the_pure_python_filename(tmp_path: Path) -> None:
    wheel = tmp_path / "fastplms-1.0.0-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fastplms/__init__.py", b"")

    with pytest.raises(DistributionInspectionError, match="py3-none-any"):
        inspect_wheel(wheel, project_root=tmp_path)


def test_wheel_inspection_rejects_oversized_members_before_reading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "fastplms-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fastplms/oversized.py", b"x")
    monkeypatch.setattr(distribution_inspect, "_MAX_MEMBER_BYTES", 0)

    with pytest.raises(DistributionInspectionError, match="32 MiB"):
        inspect_wheel(wheel, project_root=tmp_path)


def test_sdist_inspection_rejects_non_regular_members(tmp_path: Path) -> None:
    sdist = tmp_path / "fastplms-1.0.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        root = tarfile.TarInfo("fastplms-1.0.0")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        symlink = tarfile.TarInfo("fastplms-1.0.0/src/fastplms/injected.py")
        symlink.type = tarfile.SYMTYPE
        symlink.linkname = "../../../../outside.py"
        archive.addfile(symlink)

    with pytest.raises(DistributionInspectionError, match="non-regular"):
        inspect_sdist(sdist, project_root=tmp_path)


def test_sdist_inspection_rejects_sensitive_member(tmp_path: Path) -> None:
    sdist = tmp_path / "fastplms-1.0.0.tar.gz"
    payload = b"token"
    with tarfile.open(sdist, "w:gz") as archive:
        member = tarfile.TarInfo("fastplms-1.0.0/.env")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(DistributionInspectionError, match="Sensitive"):
        inspect_sdist(sdist, project_root=tmp_path)


def test_distribution_source_snapshot_rejects_untracked_runtime_injection(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    runtime = project / "src" / "fastplms"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").write_text("__version__ = '1.0.0'\n", encoding="utf-8")
    commands = (
        ("init", "--initial-branch=main"),
        ("config", "user.email", "tests@example.invalid"),
        ("config", "user.name", "FastPLMs Tests"),
        ("config", "commit.gpgsign", "false"),
        ("config", "core.autocrlf", "false"),
        ("add", "."),
        ("commit", "-m", "tracked distribution source"),
    )
    for arguments in commands:
        subprocess.run(
            ["git", *arguments],
            cwd=project,
            check=True,
            capture_output=True,
        )
    (runtime / "injected.py").write_text("TOKEN = 'not packaged'\n", encoding="utf-8")

    with pytest.raises(DistributionInspectionError, match="clean and fully tracked"):
        distribution_inspect._tracked_source_snapshot(project)
