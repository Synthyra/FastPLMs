"""Which working-tree paths a Modal evidence worker receives.

Only named directories and files are uploaded, never the repository root, so
root-level credential files cannot be included. The exclusion predicate is a
second barrier for credential-shaped names nested inside an uploaded directory.
"""

from __future__ import annotations

import io
import os
import subprocess
import tarfile

from pathlib import Path

from tools.execution.source import (
    SourceSnapshot,
    excluded_from_upload as excluded_from_upload,
    stage_source_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = "/workspace"
# ``src/`` at a Git revision, so one worker can compare the working tree against it.
BASELINE_DIRECTORY = ROOT / "artifacts/gpu_evidence/baseline"
BASELINE_WORKSPACE = "/baseline"

SOURCE_DIRECTORIES = (
    "src",
    "tests",
    "tools",
    "benchmarks",
    "docs",
    "model_cards",
    "LICENSES",
    "requirements",
    "docker",
    "examples",
    # The pinned official ESMFold2 sources that exact source parity compares against.
    "vendor/upstream/biohub-transformers/src/transformers/models/esmfold2",
    "vendor/upstream/biohub-esm/esm",
    # The pinned official ESMC source that the sparse-autoencoder tests compare against.
    "vendor/upstream/biohub-transformers/src/transformers/models/esmc",
)
SOURCE_FILES = (
    "pytest.ini",
    "ruff.toml",
    "mypy.ini",
    "kernels.lock",
    "evidence.toml",
    "vendor/README.md",
    # The source-inventory parity test checks the pinned submodule list.
    ".gitmodules",
    ".dockerignore",
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
)

REFERENCE_SOURCE_DIRECTORIES = (
    "vendor/upstream/biohub-transformers",
    "vendor/upstream/biohub-esm",
)
REFERENCE_EXCLUDED_DIRECTORIES = frozenset({
    ".git", ".github", "tests", "docs", "examples", "benchmark", "benchmark_v2",
    "notebooks", "templates", "i18n", "docker", "scripts", "cookbook", "_assets",
})


def stage_upload_source(destination: Path) -> SourceSnapshot:
    """Freeze candidate source and the isolated official reference build inputs."""
    def exclude(path: Path) -> bool:
        for root_name in REFERENCE_SOURCE_DIRECTORIES:
            root = Path(root_name)
            if path.is_relative_to(root):
                relative = path.relative_to(root)
                if relative.parts and relative.parts[0] in REFERENCE_EXCLUDED_DIRECTORIES:
                    return True
        return excluded_from_upload(path)

    return stage_source_snapshot(
        ROOT, destination,
        directories=(*SOURCE_DIRECTORIES, *REFERENCE_SOURCE_DIRECTORIES),
        files=SOURCE_FILES,
        exclude=exclude,
    )


def upload_source_root() -> Path:
    """Use the launcher's frozen snapshot when building local Modal images."""
    return Path(os.environ.get("FASTPLMS_EVIDENCE_SOURCE_ROOT", str(ROOT)))


def baseline_source_root() -> Path:
    return Path(os.environ.get("FASTPLMS_EVIDENCE_BASELINE_ROOT", str(BASELINE_DIRECTORY)))


def export_baseline_source(revision: str = "HEAD", *, destination: Path | None = None) -> str:
    """Materialize the runtime source at a Git revision and return the resolved commit.

    ``kernels.lock`` travels with ``src/`` because the FlashAttention loader reads it
    from the root of the tree it was imported from.
    """
    destination = destination or BASELINE_DIRECTORY
    resolved = subprocess.run(
        [
            "git", "-c", f"safe.directory={ROOT.as_posix()}",
            "rev-parse", "--verify", f"{revision}^{{commit}}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    archive = subprocess.run(
        [
            "git", "-c", f"safe.directory={ROOT.as_posix()}",
            "archive", "--format=tar", resolved, "src", "kernels.lock",
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    destination = destination.resolve()
    if not destination.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Baseline source destination must stay inside workspace artifacts")
    if destination.exists():
        raise FileExistsError(f"Baseline source already exists: {destination}")
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if excluded_from_upload(path) or member.issym() or member.islnk():
                raise RuntimeError(f"Baseline archive contains a forbidden path: {member.name}")
        tar.extractall(destination, filter="data")
    (destination / "REVISION").write_text(f"{resolved}\n", encoding="utf-8")
    return resolved
