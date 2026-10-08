"""A Modal image built from a workspace environment, with the code a project imports mounted.

    from foundry.modal_image import workspace_image

    image = workspace_image("fastplms_runtime", packages=["biotite"], apt=["git"])

The environment's manifest names the base image (`modal_base`), its Python, and where torch
comes from (`torch_index`); its `lock.txt` holds exact versions. The torch family installs from
its index first, then the rest of the lock together with `packages`, so every version the lock
pins is the one installed and everything else resolves around it.

The code is what `ws run` puts on PYTHONPATH, so an app is launched through it:

    ws run <project> -- python -m modal run app.py

Each import root is mounted at `/ws/tree/<its workspace path>` and foundry at
`/ws/import-roots/foundry`, the layout `ws remote` uses. Caches, version control, and credential
files are never mounted.

Any other directory an image adds takes `ignore=ignore_with_credentials(patterns)`, which
leaves out what `patterns` match and, at any depth, every cache, version control directory, and
credential file. A clone of a projection keeps
`.secrets.env` at its root, and no image may carry it.

An app that builds its own image and imports foundry adds `workspace_foundry(source)` beside
its source directory. A clone vendors foundry into that directory already. The workspace keeps
its one copy at the root, so a launch from the workspace adds that copy.

Modal imports the app module again inside the container, where there is no workspace. There
this returns a bare image, because the container already runs the image built on the machine
that launched it.
"""

from __future__ import annotations

import os
import re
import modal

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from foundry.logging.credentials import is_credential_path


TREE = PurePosixPath("/ws/tree")
FOUNDRY_ROOT = PurePosixPath("/ws/import-roots")
TORCH_FAMILY = frozenset({"torch", "torchvision", "torchaudio"})
SKIPPED_DIRECTORIES = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".ipynb_checkpoints", ".venv", "venv", "node_modules", "wandb"}
)
SKIPPED_SUFFIXES = frozenset({".pyc", ".pyo"})
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)$")
LAUNCH = "ws run <project> -- python -m modal run <app.py>"


@dataclass(frozen=True, slots=True)
class Stack:
    """What an environment's manifest and lock say its image holds."""

    environment: str
    base: str
    python: str
    torch_index: str | None
    pins: tuple[str, ...]

    @property
    def torch_pins(self) -> list[str]:
        return [pin for pin in self.pins if pin_name(pin) in TORCH_FAMILY]

    @property
    def other_pins(self) -> list[str]:
        return [pin for pin in self.pins if pin_name(pin) not in TORCH_FAMILY]


@dataclass(frozen=True, slots=True)
class Mount:
    """A local directory, where it appears in the container, and the PYTHONPATH entry it serves."""

    local: Path
    remote: PurePosixPath
    path_entry: PurePosixPath


def workspace_image(
    environment: str,
    *,
    packages: Sequence[str] = (),
    apt: Sequence[str] = (),
    commands: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
    copy: bool = False,
    registry_secret: modal.Secret | None = None,
    force_build: bool = False,
) -> modal.Image:
    """The image for `environment`, with `packages` resolved against its lock and the import roots mounted.

    `commands` run after the packages install, for a build step that needs them. `copy=True`
    builds the code into the image instead of adding it at container start.
    """
    if not modal.is_local():
        return modal.Image.debian_slim()
    root = workspace_root(Path(__file__))
    stack = load_stack(root, environment)
    mounts = import_mounts(root, os.environ.get("PYTHONPATH", ""))

    image = modal.Image.from_registry(stack.base, secret=registry_secret, add_python=stack.python, force_build=force_build)
    if apt:
        image = image.apt_install(*apt)
    if stack.torch_pins:
        image = image.uv_pip_install(*stack.torch_pins, index_url=stack.torch_index)
    if stack.other_pins or packages:
        image = image.uv_pip_install(*stack.other_pins, *packages)
    if commands:
        image = image.run_commands(*commands)
    # Settings go before the code, since nothing may follow files added at container start.
    path_entries = list(dict.fromkeys(str(mount.path_entry) for mount in mounts))
    image = image.env({**(env or {}), "PYTHONPATH": ":".join(path_entries)})
    for mount in outermost(mounts):
        image = image.add_local_dir(mount.local, str(mount.remote), copy=copy, ignore=skipped)
    return image


def workspace_root(start: Path) -> Path:
    """The workspace this module runs from: the nearest directory above it with AGENTS.md and environments/."""
    for candidate in start.resolve().parents:
        if (candidate / "AGENTS.md").is_file() and (candidate / "environments").is_dir():
            return candidate
    raise RuntimeError(
        f"workspace_image builds from the workspace's environments/, and {start} is not inside a workspace. "
        "A projection has no environments; launch from the workspace."
    )


def load_stack(root: Path, environment: str) -> Stack:
    """Read `environments/<environment>/README.md` and its lock."""
    import yaml  # the container never reaches here, and need not have yaml

    directory = root / "environments" / environment
    manifest = directory / "README.md"
    if not manifest.is_file():
        raise ValueError(f"no environment '{environment}': {manifest} does not exist")
    text = manifest.read_text(encoding="utf-8")
    fields = yaml.safe_load(text.split("---", 2)[1]) if text.startswith("---") else {}
    missing = [name for name in ("modal_base", "python") if not fields.get(name)]
    if missing:
        raise ValueError(f"{manifest} names no {' or '.join(missing)}, so no Modal image can be built from it")
    return Stack(
        environment=environment,
        base=str(fields["modal_base"]),
        python=str(fields["python"]),
        torch_index=str(fields["torch_index"]) if fields.get("torch_index") else None,
        pins=read_pins(directory / "lock.txt"),
    )


def read_pins(lock: Path) -> tuple[str, ...]:
    """The `name==version` lines of a lock, as written."""
    lines = [line.strip() for line in lock.read_text(encoding="utf-8").splitlines()]
    pins = [line for line in lines if line and not line.startswith("#")]
    malformed = [line for line in pins if not PIN.match(line)]
    if malformed:
        raise ValueError(f"{lock}: not a name==version pin: {malformed[0]}")
    return tuple(pins)


def pin_name(pin: str) -> str:
    """A pin's package name, lowercased with `_` as `-`, so `Typing_Extensions` is `typing-extensions`."""
    return PIN.match(pin).group(1).lower().replace("_", "-")


def import_mounts(root: Path, pythonpath: str) -> list[Mount]:
    """Each workspace directory on PYTHONPATH, and foundry for the shim `ws run` puts there.

    Raises when PYTHONPATH carries no workspace directory, which means the app was not launched
    through `ws run`, and when a directory it names is missing.
    """
    mounts: list[Mount] = []
    for entry in filter(None, pythonpath.split(os.pathsep)):
        path = Path(entry).resolve()
        if path == root:
            raise RuntimeError(f"PYTHONPATH holds the workspace root itself; launch with `{LAUNCH}`, which never does")
        if path.is_relative_to(root):
            remote = TREE / path.relative_to(root).as_posix()
            mounts.append(Mount(path, remote, remote))
        elif (path / "foundry" / "__init__.py").is_file():
            mounts.append(Mount(root / "foundry", FOUNDRY_ROOT / "foundry", FOUNDRY_ROOT))
    if not any(mount.remote.is_relative_to(TREE) for mount in mounts):
        raise RuntimeError(f"PYTHONPATH names no workspace directory; launch with `{LAUNCH}`")
    absent = [str(mount.local) for mount in mounts if not mount.local.is_dir()]
    if absent:
        raise FileNotFoundError(f"import roots missing: {', '.join(absent)}")
    return mounts


def outermost(mounts: Sequence[Mount]) -> list[Mount]:
    """The mounts not inside another, so a directory is added once."""
    unique = {mount.local: mount for mount in mounts}
    return [mount for local, mount in unique.items() if not any(other != local and local.is_relative_to(other) for other in unique)]


def skipped(path: Path) -> bool:
    """Whether Modal leaves this file out: a cache, version control, or a credential file."""
    return (
        any(part in SKIPPED_DIRECTORIES for part in path.parts)
        or path.suffix in SKIPPED_SUFFIXES
        or is_credential_path(path.as_posix())
    )


class _PatternsAndCredentials(modal.FilePatternMatcher):
    """Modal's pattern matcher, which also matches every file `skipped` does.

    Modal skips a directory the patterns match without walking it only when the rule is a
    `FilePatternMatcher`. For a plain function it walks every file, so the rule keeps the type.
    """

    def __call__(self, path: Path) -> bool:
        return super().__call__(path) or skipped(path)


def ignore_with_credentials(patterns: Sequence[str] = ()) -> modal.FilePatternMatcher:
    """Modal's `ignore` for a local directory: what `patterns` match, and every cache, version
    control directory, and credential file at any depth.

    Modal passes each path relative to the directory it adds, which is the form
    `is_credential_path` reads. A pattern is a dockerignore-style glob, as Modal takes one: a bare
    name such as `__pycache__` matches only at the top level, `**/__pycache__` at every depth, and
    a leading `/` matches nothing on Windows.
    """
    return _PatternsAndCredentials(*patterns)


def workspace_foundry(source: str | Path) -> Path | None:
    """The workspace's foundry package, for an image to add beside `source`, or None.

    A clone of a projection vendors foundry into its source directory, which the image adds
    anyway, so there it is None. The workspace keeps its one copy at the root, outside every
    project, and a launch from the workspace adds it: the package this module was imported
    from. Inside the container it is None too, since the container runs the image built on the
    machine that launched it.
    """
    if not modal.is_local() or (Path(source) / "foundry" / "__init__.py").is_file():
        return None
    return Path(__file__).resolve().parents[1]


def workspace_model_family(source: str | Path, family: str) -> Path | None:
    """The workspace's `models/<family>` package, for an image to add beside `source`, or None.

    The same rule as `workspace_foundry`: None in a clone, whose source directory vendors the
    family, and inside the container.
    """
    if not modal.is_local() or (Path(source) / family / "__init__.py").is_file():
        return None
    package = workspace_root(Path(__file__)) / "models" / family
    if not (package / "__init__.py").is_file():
        raise FileNotFoundError(f"the workspace holds no model family {family!r} at {package}")
    return package


MMSEQS_RELEASE = "18-8cc5c"
MMSEQS_COMMIT = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
MMSEQS_SHA256 = "bd9b0234da5949ad528d5b5f9ea4cda9c1e23dce14b46c0791d4d919a76e61ce"
MMSEQS_BINARY = "/opt/mmseqs/bin/mmseqs"
_MMSEQS_URL = f"https://github.com/soedinglab/MMseqs2/releases/download/{MMSEQS_RELEASE}/mmseqs-linux-avx2.tar.gz"


def with_mmseqs(image: modal.Image) -> modal.Image:
    """`image` with the MMseqs2 release the workspace's clustering is pinned to, at `MMSEQS_BINARY`.

    The archive is checked against the SHA-256 GitHub records for it. Pass `MMSEQS_BINARY` as the
    binary and `MMSEQS_COMMIT` as the expected version to `foundry.datasets`.
    """
    return image.apt_install("curl", "ca-certificates").run_commands(
        f"curl -fsSL {_MMSEQS_URL} -o /tmp/mmseqs.tar.gz",
        f"echo '{MMSEQS_SHA256}  /tmp/mmseqs.tar.gz' | sha256sum -c -",
        "tar -xzf /tmp/mmseqs.tar.gz -C /opt && rm /tmp/mmseqs.tar.gz",
    )
