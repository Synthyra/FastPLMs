"""Distribution-location contracts for the immutable Hugging Face kernel lock."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from fastplms.attention import _kernel_lock

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "kernels.lock"
WHEEL_LOCK = "fastplms-1.0.0.dist-info/kernels.lock"
WHEEL_NOTICES = "fastplms-1.0.0.dist-info/licenses/THIRD_PARTY_NOTICES.md"


def _python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def test_direct_repository_resolves_the_tracked_kernel_lock() -> None:
    assert _kernel_lock._kernel_lock_path().resolve() == LOCK.resolve()
    assert len(json.loads(LOCK.read_text(encoding="utf-8"))) == 2


def test_editable_install_and_built_wheel_embed_the_exact_kernel_lock(
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
        ("uv", "build", "--wheel", "--out-dir", str(wheel_dir), str(ROOT)),
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheels = tuple(wheel_dir.glob("fastplms-1.0.0-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        assert archive.read(WHEEL_LOCK) == LOCK.read_bytes()
        assert archive.read(WHEEL_NOTICES) == (ROOT / "THIRD_PARTY_NOTICES.md").read_bytes()
