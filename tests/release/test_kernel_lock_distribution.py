"""Distribution-location contracts for the immutable Hugging Face kernel lock."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

from fastplms.attention import _kernel_lock

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "kernels.lock"
WHEEL_LOCK = "fastplms-1.0.0.dist-info/kernels.lock"
WHEEL_NOTICES = "fastplms-1.0.0.dist-info/licenses/THIRD_PARTY_NOTICES.md"
WHEEL_LICENSE_PREFIX = "fastplms-1.0.0.dist-info/licenses/"
WHEEL_METADATA = "fastplms-1.0.0.dist-info/METADATA"


def _python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def test_direct_repository_resolves_the_tracked_kernel_lock() -> None:
    assert _kernel_lock._kernel_lock_path().resolve() == LOCK.resolve()
    assert len(json.loads(LOCK.read_text(encoding="utf-8"))) == 2


def test_editable_install_and_built_distributions_embed_exact_release_files(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()

    venv = tmp_path / "editable"
    created = subprocess.run(
        ("uv", "venv", "--python", sys.executable, str(venv)),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert created.returncode == 0, created.stdout + created.stderr
    installed = subprocess.run(
        (
            "uv",
            "pip",
            "install",
            "--python",
            str(_python(venv)),
            "--no-deps",
            "--editable",
            str(ROOT),
        ),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    editable_lock = subprocess.run(
        (
            str(_python(venv)),
            "-I",
            "-c",
            (
                "import importlib.metadata as m; "
                "value=m.distribution('fastplms').read_text('kernels.lock'); "
                "assert value is not None; print(value, end='')"
            ),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert editable_lock.returncode == 0, editable_lock.stdout + editable_lock.stderr
    assert editable_lock.stdout.encode() == LOCK.read_bytes()

    wheel_dir = tmp_path / "wheel"
    built = subprocess.run(
        ("uv", "build", "--out-dir", str(wheel_dir), str(ROOT)),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheels = tuple(wheel_dir.glob("fastplms-1.0.0-*.whl"))
    sdists = tuple(wheel_dir.glob("fastplms-1.0.0.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1
    legal_sources = {
        "LICENSE": ROOT / "LICENSE",
        "THIRD_PARTY_NOTICES.md": ROOT / "THIRD_PARTY_NOTICES.md",
        **{
            path.relative_to(ROOT).as_posix(): path
            for path in sorted((ROOT / "LICENSES").rglob("*"))
            if path.is_file()
        },
    }
    with zipfile.ZipFile(wheels[0]) as archive:
        assert archive.read(WHEEL_LOCK) == LOCK.read_bytes()
        assert archive.read(WHEEL_NOTICES) == (ROOT / "THIRD_PARTY_NOTICES.md").read_bytes()
        wheel_legal_files = {
            name.removeprefix(WHEEL_LICENSE_PREFIX)
            for name in archive.namelist()
            if name.startswith(WHEEL_LICENSE_PREFIX) and not name.endswith("/")
        }
        assert wheel_legal_files == set(legal_sources)
        for relative_name, source in legal_sources.items():
            assert archive.read(f"{WHEEL_LICENSE_PREFIX}{relative_name}") == source.read_bytes()

        metadata = archive.read(WHEEL_METADATA).decode("utf-8")
        declared_licenses = {
            line.removeprefix("License-File: ")
            for line in metadata.splitlines()
            if line.startswith("License-File: ")
        }
        assert declared_licenses == set(legal_sources)

    sdist_prefix = "fastplms-1.0.0/"
    with tarfile.open(sdists[0], mode="r:gz") as archive:
        names = {member.name for member in archive.getmembers() if member.isfile()}
        sdist_legal_files = {
            name.removeprefix(sdist_prefix)
            for name in names
            if name in {
                f"{sdist_prefix}LICENSE",
                f"{sdist_prefix}THIRD_PARTY_NOTICES.md",
            }
            or name.startswith(f"{sdist_prefix}LICENSES/")
        }
        assert sdist_legal_files == set(legal_sources)
        for relative_name, source in legal_sources.items():
            stream = archive.extractfile(f"{sdist_prefix}{relative_name}")
            assert stream is not None
            assert stream.read() == source.read_bytes()
