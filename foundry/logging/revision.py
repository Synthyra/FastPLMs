"""Which code a run executed: a git commit where one describes it, else a hash of the sources.

A GPU machine runs a projection, a git clone whose commit names the code exactly. The workspace
has no git and a tarball deploy has none either, so every revision also carries `tree_sha256`,
a hash over the source files that needs no git. It comes out the same wherever the same files
are checked out:

- Text is hashed with LF line endings, as git stores it. The workspace holds files as their
  repositories store them, but an editor on Windows can still save one with CRLF, and a clone
  made with `core.autocrlf=true` writes CRLF throughout.
- READMEs and AGENTS.md are left out. The workspace README is a manifest, a projection's README
  is the public one, and publish appends a notice to AGENTS.md.

So a run's tree hash can be checked against the workspace by hashing the projection `ws publish`
builds, which is the tree the clone checked out, vendored `foundry` included.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath


SOURCE_PATHS = ("src", "scripts", "configs", "pyproject.toml")
DOCUMENT_NAMES = frozenset({"README.md", "PUBLIC_README.md", "AGENTS.md"})
GENERATED_DIRECTORIES = frozenset(
    {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints", "wandb", "results", "runs", "outputs"}
)
GENERATED_SUFFIXES = frozenset({".pyc", ".pyo"})
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class Revision:
    """The code a run executed. `kind` and `value` are the identity to cite."""

    kind: str  # "git_commit" for a clean checkout rooted at the sources, else "source_tree_sha256"
    value: str
    tree_sha256: str
    file_count: int
    git_commit: str | None  # HEAD, only when the source root is the top of a git checkout
    git_clean: bool | None  # every hashed file committed and unmodified, when git_commit is known

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return asdict(self)


def source_revision(root: Path, include: Sequence[str] = SOURCE_PATHS) -> Revision:
    """The revision of the files under `include`, which name directories and files in `root`.

    `root` is usually the project root. Code deployed without its project, as on Modal, passes
    the directory holding its packages and names them in `include`, so the tree hash is the
    same there as in a checkout. Raises when nothing matches: a hash of no files identifies
    nothing.
    """
    root = root.resolve()
    files = source_files(root, include)
    if not files:
        raise ValueError(f"no source files under {list(include)} in {root}; name this project's code in include=.")
    tree = tree_sha256(root, files)
    commit, clean = git_state(root, files)
    if commit is not None and clean:
        return Revision("git_commit", commit, tree, len(files), commit, clean)
    return Revision("source_tree_sha256", tree, tree, len(files), commit, clean)


def revision_of(path: Path) -> Revision:
    """The revision of the code `path` belongs to, found without being told where it lives.

    That is the nearest project above `path` (a directory with `pyproject.toml`) whose sources
    include it; a script outside any project, such as one in an experiment directory, is
    revised with the directory that holds it.
    """
    path = path.resolve()
    for root in path.parents:
        if (root / "pyproject.toml").is_file() and path.relative_to(root).as_posix() in source_files(root, SOURCE_PATHS):
            return source_revision(root)
    return source_revision(path.parent.parent, include=(path.parent.name,))


def source_files(root: Path, include: Sequence[str]) -> list[str]:
    """Root-relative posix paths of the source files under `include`, sorted."""
    found: set[str] = set()
    for name in include:
        path = root / name
        candidates = path.rglob("*") if path.is_dir() else [path] if path.is_file() else []
        for candidate in candidates:
            relative = candidate.relative_to(root)
            if candidate.is_file() and not _is_excluded(relative):
                found.add(relative.as_posix())
    return sorted(found)


def tree_sha256(root: Path, files: Sequence[str]) -> str:
    """sha256 over canonical JSON of each file's path, and the sha256 and size of its LF content."""
    records: list[dict[str, str | int]] = []
    for relative in files:
        content = normalized_bytes(root / relative)
        records.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)})
    canonical = json.dumps(records, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def normalized_bytes(path: Path) -> bytes:
    """A file's content as git stores it: CRLF becomes LF in text, and binary is untouched."""
    content = path.read_bytes()
    return content if b"\0" in content else content.replace(b"\r\n", b"\n")


def git_state(root: Path, files: Sequence[str]) -> tuple[str | None, bool | None]:
    """HEAD, and whether every file is committed unmodified, or (None, None) when `root` is not
    the top of a git checkout. A project inside some larger repository would otherwise cite
    that repository's commit for code it does not describe.
    """

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False)

    try:
        top_level = git("rev-parse", "--show-toplevel")
    except OSError:  # no git on this machine
        return None, None
    if top_level.returncode != 0 or Path(top_level.stdout.strip()).resolve() != root:
        return None, None
    head = git("rev-parse", "--verify", "HEAD")
    commit = head.stdout.strip().lower()
    if head.returncode != 0 or GIT_COMMIT.fullmatch(commit) is None:
        return None, None

    # A modified or untracked file shows in status; an ignored one does not, so every hashed
    # file must also be tracked for the commit to describe it.
    scope = sorted({PurePath(relative).parts[0] for relative in files})
    status = git("status", "--porcelain=v1", "--untracked-files=all", "--", *scope)
    tracked = git("ls-files", "--cached", "--", *scope)
    if status.returncode != 0 or tracked.returncode != 0:
        return commit, None
    return commit, not status.stdout.strip() and set(files) <= set(tracked.stdout.splitlines())


def _is_excluded(relative: PurePath) -> bool:
    return (
        relative.name in DOCUMENT_NAMES
        or relative.suffix in GENERATED_SUFFIXES
        or any(part in GENERATED_DIRECTORIES or part.endswith(".egg-info") for part in relative.parts)
    )
