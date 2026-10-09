"""Pinned official implementations, imported beside FastPLMs by the CPU equivalence tests.

FastPLMs' runtime never imports official code. The equivalence tests in `tests/unit` do: each
runs a family's official modules unchanged beside FastPLMs' own, both holding one small random
network, and compares every function on the forward path.

`models.toml` pins each official repository under `[[upstreams]]`. A development checkout holds
it as a submodule at `vendor/upstream/<id>`. Elsewhere, `fetch` downloads GitHub's archive of the
pinned commit once into `<root>/<id>@<revision>/`, outside every synced or published tree, and
records beside it the commit the archive names and the SHA-256 of the archive and of the tree.
The root is `$FASTPLMS_PARITY_ORACLES`, or `C:/ws-cache/parity-oracles`, where the research
workspace keeps machine-local caches. Tests never download: a missing tree skips its tests and
names the command that fetches it.

    python -m tests.parity.support.pinned_oracles fetch fair-esm biohub-esm e1
    python -m tests.parity.support.pinned_oracles verify

Two official packages share the top-level name `esm`: fair-esm and Biohub ESM. The research
interpreter also installs fair-esm 2.0.0, whose files differ from the pinned commit's. So
`official_package` imports a package from its pinned tree with every loaded module of that name
set aside, keeps the modules the import loaded, and then puts the set-aside modules back. The
official code keeps working afterwards, because each official module holds its imports by
reference; none of the code these tests call imports its own package lazily.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import pytest
import torch

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType

from fastplms.digests import file_sha256
from fastplms.registry import UpstreamSource, get_model_registry


ORACLE_ROOT_VARIABLE = "FASTPLMS_PARITY_ORACLES"
DEFAULT_ORACLE_ROOT = Path("C:/ws-cache/parity-oracles")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ARCHIVE_URL = "https://codeload.github.com/{repository}/tar.gz/{revision}"
FETCH_COMMAND = "python -m tests.parity.support.pinned_oracles fetch"


def oracle_root() -> Path:
    return Path(os.environ.get(ORACLE_ROOT_VARIABLE, DEFAULT_ORACLE_ROOT))


def upstream_source(upstream_id: str) -> UpstreamSource:
    return get_model_registry().upstreams[upstream_id]


def fetched_tree(source: UpstreamSource) -> Path:
    return oracle_root() / f"{source.id}@{source.revision}"


def source_record(source: UpstreamSource) -> Path:
    return oracle_root() / f"{source.id}@{source.revision}.json"


def pinned_tree(upstream_id: str) -> Path | None:
    """This machine's copy of the official tree at its pinned commit, or None."""
    source = upstream_source(upstream_id)
    submodule = REPOSITORY_ROOT / source.path
    # A development checkout's submodule sits at the commit its gitlink pins.
    if submodule.is_dir() and any(submodule.iterdir()):
        return submodule
    tree, record = fetched_tree(source), source_record(source)
    if not (tree.is_dir() and record.is_file()):
        return None
    if json.loads(record.read_text(encoding="utf-8"))["revision"] != source.revision:
        return None
    return tree


def require_pinned_tree(upstream_id: str) -> Path:
    """The pinned official tree, or a skip that names the command fetching it."""
    tree = pinned_tree(upstream_id)
    if tree is None:
        revision = upstream_source(upstream_id).revision
        pytest.skip(
            f"the official {upstream_id} source at {revision[:12]} is not on this machine; "
            f"run `{FETCH_COMMAND} {upstream_id}`"
        )
    return tree


def _modules_named(package: str) -> dict[str, ModuleType]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == package or name.startswith(f"{package}.")
    }


@contextmanager
def _package_set_aside(package: str) -> Iterator[None]:
    """Hide every loaded module of one top-level name, and restore them afterwards."""
    set_aside = _modules_named(package)
    for name in set_aside:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in _modules_named(package):
            del sys.modules[name]
        sys.modules.update(set_aside)


@dataclass(frozen=True)
class OfficialPackage:
    """Modules of one official package, imported from its pinned tree."""

    upstream_id: str
    import_root: Path
    modules: Mapping[str, ModuleType]

    def __getitem__(self, name: str) -> ModuleType:
        return self.modules[name]


@cache
def official_package(
    upstream_id: str,
    package: str,
    entry_modules: tuple[str, ...],
    source_directory: str = ".",
) -> OfficialPackage:
    """Import official modules from the pinned tree, once per session.

    `source_directory` is the tree's import root, such as `src` for a src-layout repository.
    """
    import_root = (require_pinned_tree(upstream_id) / source_directory).resolve()
    with _package_set_aside(package):
        sys.path.insert(0, str(import_root))
        try:
            for name in entry_modules:
                importlib.import_module(name)
        finally:
            sys.path.remove(str(import_root))
        loaded = _modules_named(package)
    origin = Path(loaded[package].__file__).resolve()
    if import_root not in origin.parents:
        raise RuntimeError(f"{package} was imported from {origin}, not the pinned {import_root}")
    return OfficialPackage(upstream_id, import_root, MappingProxyType(loaded))


def cached_checkpoint(repo_id: str, revision: str, filenames: tuple[str, ...]) -> Path:
    """The Hugging Face cache's snapshot of a checkpoint revision, or a skip when a file is absent.

    Loading tests read only this cache. They never download a checkpoint.
    """
    from huggingface_hub import try_to_load_from_cache

    snapshot: Path | None = None
    for filename in filenames:
        cached = try_to_load_from_cache(repo_id, filename, revision=revision)
        if not isinstance(cached, str):
            pytest.skip(f"{repo_id} at {revision[:12]} has no cached {filename}")
        # A file in a subdirectory, such as data/weights/model.pth, sits that many levels down.
        snapshot = Path(cached).parents[len(PurePosixPath(filename).parts) - 1]
    assert snapshot is not None, "a checkpoint names at least one file"
    return snapshot


def randomize_parameters(module: torch.nn.Module, seed: int, scale: float) -> None:
    """Fill every parameter of an official network from a seeded normal distribution.

    Initializers leave biases at zero and norm weights at one, which would hide a swapped or
    dropped term; random values make every parameter matter to the output.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * scale)


def assert_identical(actual: torch.Tensor, expected: torch.Tensor, what: str = "") -> None:
    """Exact equality: same shape, dtype, and every value, as the equivalence tests require."""
    # actual, expected: (...) equal shapes of any rank
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0,
        atol=0,
        msg=lambda message: f"{what}: {message}" if what else message,
    )


# float32 carries about seven significant digits. A few layers of float32 sums taken in another
# order, which is all a batched GEMM or a fused attention kernel changes, cost one or two of them.
ROUNDING_FRACTION = 1e-5


def assert_equal_to_rounding(actual: torch.Tensor, expected: torch.Tensor, what: str = "") -> None:
    """Equality up to float32 rounding, bounded relative to the expected tensor's largest value.

    For comparisons whose two sides sum the same terms in different orders. A masking, weight, or
    formula error moves values by a large fraction of their scale, far outside this bound.
    """
    # actual, expected: (...) equal shapes of any rank
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"{what}: dtype {actual.dtype} != {expected.dtype}"
    bound = ROUNDING_FRACTION * expected.abs().max().item()
    difference = (actual - expected).abs().max().item()
    assert difference <= bound, f"{what}: largest difference {difference:.3e} exceeds {bound:.3e}"


def _tree_sha256(files: Mapping[str, str]) -> str:
    """One digest over a tree's relative POSIX paths and their contents' SHA-256."""
    lines = "".join(f"{path}\0{files[path]}\n" for path in sorted(files))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _disk_files(tree: Path) -> dict[str, str]:
    # Python writes bytecode beside imported source unless PYTHONPYCACHEPREFIX moves it.
    return {
        path.relative_to(tree).as_posix(): file_sha256(path)
        for path in tree.rglob("*")
        if path.is_file() and "__pycache__" not in path.relative_to(tree).parts
    }


def _archive_files(archive: tarfile.TarFile) -> dict[str, str]:
    """Contents of a GitHub archive, keyed by path below its one top-level directory."""
    files = {}
    for member in archive.getmembers():
        if not member.isfile():
            continue
        handle = archive.extractfile(member)
        assert handle is not None, f"{member.name} is a regular file"
        relative = member.name.split("/", 1)[1]
        files[relative] = hashlib.sha256(handle.read()).hexdigest()
    return files


def _github_repository(url: str) -> str:
    # models.toml requires https://github.com/<owner>/<name>.git
    return url.removeprefix("https://github.com/").removesuffix(".git")


def fetch(upstream_id: str) -> Path:
    """Download the pinned commit's archive into the oracle root, once."""
    source = upstream_source(upstream_id)
    tree, record = fetched_tree(source), source_record(source)
    if tree.is_dir() and record.is_file():
        return tree
    url = ARCHIVE_URL.format(
        repository=_github_repository(source.clone_url), revision=source.revision
    )
    with urllib.request.urlopen(url, timeout=600) as response:
        payload = response.read()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        files = _archive_files(archive)
        # git archive writes the commit it was made from into the global pax header.
        archived_commit = archive.pax_headers.get("comment")
        if archived_commit != source.revision:
            raise RuntimeError(f"{url} holds commit {archived_commit}, not {source.revision}")
        if not tree.is_dir():
            tree.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=f".{source.id}-", dir=tree.parent))
            archive.extractall(staging, filter="data")
            (top_level,) = staging.iterdir()
            top_level.rename(tree)
            staging.rmdir()
    if _disk_files(tree) != files:
        raise RuntimeError(f"{tree} differs from the archive of {source.revision}")
    record.write_text(
        json.dumps(
            {
                "id": source.id,
                "url": source.url,
                "revision": source.revision,
                "archive": url,
                "archive_sha256": hashlib.sha256(payload).hexdigest(),
                "tree_sha256": _tree_sha256(files),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return tree


def verify(upstream_id: str) -> bool:
    """Whether a fetched tree still holds exactly the files its record describes."""
    source = upstream_source(upstream_id)
    tree, record = fetched_tree(source), source_record(source)
    if not (tree.is_dir() and record.is_file()):
        return False
    expected = json.loads(record.read_text(encoding="utf-8"))
    return expected["revision"] == source.revision and _tree_sha256(_disk_files(tree)) == expected[
        "tree_sha256"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("fetch", "verify"))
    parser.add_argument("upstreams", nargs="*", help="upstream ids from models.toml; default all")
    arguments = parser.parse_args()
    upstream_ids = arguments.upstreams or sorted(get_model_registry().upstreams)
    for upstream_id in upstream_ids:
        if arguments.action == "verify":
            print(f"{upstream_id}: {'verified' if verify(upstream_id) else 'absent or changed'}")
            continue
        try:
            print(f"{upstream_id}: {fetch(upstream_id)}")
        except urllib.error.HTTPError as error:
            # A repository GitHub no longer serves, such as a fork made private, stays unfetched.
            print(f"{upstream_id}: unavailable, {error.url} answered HTTP {error.code}")


if __name__ == "__main__":
    main()
