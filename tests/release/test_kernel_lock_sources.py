"""Source-checkout and Hub-artifact contracts for the immutable kernel lock."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

from fastplms.attention import _kernel_lock


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "kernels.lock"


def test_source_checkout_resolves_the_tracked_kernel_lock() -> None:
    assert _kernel_lock._kernel_lock_path().resolve() == LOCK.resolve()
    assert len(json.loads(LOCK.read_text(encoding="utf-8"))) == 2


def test_hub_artifact_resolves_its_embedded_kernel_lock(tmp_path: Path) -> None:
    package = tmp_path / "fastplms"
    attention = package / "attention"
    attention.mkdir(parents=True)
    artifact_lock = package / "kernels.lock"
    shutil.copyfile(LOCK, artifact_lock)
    shutil.copyfile(Path(_kernel_lock.__file__), attention / "_kernel_lock.py")

    module_spec = importlib.util.spec_from_file_location(
        "artifact_kernel_lock",
        attention / "_kernel_lock.py",
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    assert module._kernel_lock_path() == artifact_lock
    assert artifact_lock.read_bytes() == LOCK.read_bytes()
