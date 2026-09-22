"""Which working-tree paths a Modal evidence worker receives.

Only named directories and files are uploaded, never the repository root, so
root-level credential files cannot be included. The exclusion predicate is a
second barrier for credential-shaped names nested inside an uploaded directory.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tarfile

from pathlib import Path


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
    # The source-inventory parity test checks the pinned submodule list.
    ".gitmodules",
    ".dockerignore",
    # Materialized small-ESMFold2 configurations (no weights) read by unit tests.
    "artifacts/esmfold2-small/fold300/config.json",
    "artifacts/esmfold2-small/fold600/config.json",
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
)

_CREDENTIAL_NAMES = frozenset(
    {".netrc", ".npmrc", ".pypirc", ".git-credentials", ".envrc", "credentials"}
)
_CREDENTIAL_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx"})
_BUILD_DIRECTORIES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})


def excluded_from_upload(path: Path) -> bool:
    """Return whether a path must stay on the workstation."""
    if _BUILD_DIRECTORIES.intersection(path.parts) or path.suffix == ".pyc":
        return True
    name = path.name.lower()
    if name in _CREDENTIAL_NAMES or path.suffix.lower() in _CREDENTIAL_SUFFIXES:
        return True
    # ``.env``, ``.env.local``, ``prod.env``, and ``.secrets.env`` style files.
    return name == ".env" or name.startswith(".env.") or name.endswith(".env")


def export_baseline_source(revision: str = "HEAD") -> str:
    """Materialize the runtime source at a Git revision and return the resolved commit.

    ``kernels.lock`` travels with ``src/`` because the FlashAttention loader reads it
    from the root of the tree it was imported from.
    """
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    archive = subprocess.run(
        ["git", "archive", "--format=tar", resolved, "src", "kernels.lock"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    if BASELINE_DIRECTORY.exists():
        shutil.rmtree(BASELINE_DIRECTORY)
    BASELINE_DIRECTORY.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(BASELINE_DIRECTORY, filter="data")
    (BASELINE_DIRECTORY / "REVISION").write_text(f"{resolved}\n", encoding="utf-8")
    return resolved
