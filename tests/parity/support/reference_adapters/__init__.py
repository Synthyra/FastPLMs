"""Helpers for loading official implementations and immutable Hub snapshots.

Adapters in this package are deliberately independent of :mod:`fastplms`.
Their only job is to invoke an upstream public API at the source and checkpoint
revisions declared by the model manifest, then normalize output containers for
the compliance harness.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_ESM_SUBMODULE = _REPOSITORY_ROOT / "vendor" / "upstream" / "biohub-esm"


class OfficialGenerationUnavailable(RuntimeError):
    """Normalized evidence that an official public sampler cannot execute."""

    def __init__(
        self,
        *,
        public_method: str,
        exception_type: str,
        reason: str,
    ) -> None:
        super().__init__(reason)
        self.public_method = public_method
        self.exception_type = exception_type
        self.reason = reason

    def as_record(self) -> dict[str, str]:
        """Return portable metadata without a traceback or host path."""

        return {
            "status": "official_unavailable",
            "public_method": self.public_method,
            "exception_type": self.exception_type,
            "reason": self.reason,
        }


def install_byprot_sequence_namespace(source_root: Path) -> None:
    """Load only ByProt packages required by its official sequence models.

    ByProt's top-level package recursively imports every data, structure, and
    task module. That discovery path requires optional compiled OpenFold
    kernels even when only the public DPLM sequence classes are requested.
    This namespace preserves the upstream package paths and exact registry
    decorator while preventing unrelated modules from being imported. Model
    classes and their public loaders still execute directly from pinned source.
    """

    package_root = source_root / "byprot"
    if not package_root.is_dir():
        raise FileNotFoundError(f"Pinned ByProt source is missing: {package_root}")
    existing = sys.modules.get("byprot")
    marker = str(package_root.resolve())
    if existing is not None:
        if getattr(existing, "_fastplms_source_root", None) != marker:
            raise RuntimeError("A different ByProt package is already imported")
        return

    package_paths = {
        "byprot": package_root,
        "byprot.models": package_root / "models",
        "byprot.models.dplm": package_root / "models" / "dplm",
        "byprot.models.dplm2": package_root / "models" / "dplm2",
        "byprot.datamodules": package_root / "datamodules",
        "byprot.datamodules.dataset": package_root / "datamodules" / "dataset",
    }
    for name, path in package_paths.items():
        module = ModuleType(name)
        module.__file__ = str(path / "__init__.py")
        module.__package__ = name
        module.__dict__["__path__"] = [str(path)]
        sys.modules[name] = module
        parent_name, _, child_name = name.rpartition(".")
        if parent_name:
            setattr(sys.modules[parent_name], child_name, module)

    registry: dict[str, type[Any]] = {}

    def register_model(name: str) -> Callable[[type[Any]], type[Any]]:
        def decorator(model_class: type[Any]) -> type[Any]:
            registry[name] = model_class
            return model_class

        return decorator

    models = sys.modules["byprot.models"]
    models.__dict__["MODEL_REGISTRY"] = registry
    models.__dict__["register_model"] = register_model
    sys.modules["byprot"].__dict__["_fastplms_source_root"] = marker


def use_esm_submodule() -> None:
    """Load ``esm`` from the pinned Biohub submodule instead of site-packages.

    The Biohub esm package uses the same top-level `esm` import as fair-esm.
    """
    path = str(_ESM_SUBMODULE)
    if not _ESM_SUBMODULE.is_dir():
        raise FileNotFoundError(
            "Biohub ESM submodule is missing; run git submodule update --init --recursive"
        )
    if path not in sys.path:
        sys.path.insert(0, path)


def snapshot_path(repo_id: str, revision: str) -> Path:
    """Resolve one immutable Hub snapshot or fail before loading an oracle."""

    if not revision:
        raise ValueError("A non-empty immutable reference revision is required")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, revision=revision)).resolve()


def move_model(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype | None,
) -> nn.Module:
    """Move an oracle without masking its stored dtype when dtype is omitted."""

    if dtype is None:
        return model.to(device=device)
    return model.to(device=device, dtype=dtype)


@contextmanager
def pinned_biohub_snapshot(repo_id: str, revision: str) -> Iterator[Path]:
    """Expose a pinned snapshot through Biohub's supported infra-provider path.

    Biohub's official builders resolve weights relative to the process working
    directory when ``INFRA_PROVIDER`` is set. The context uses that public
    deployment mode so the official implementation reads only the requested
    immutable Hub revision. It does not modify an upstream class or forward
    implementation.
    """

    snapshot = snapshot_path(repo_id, revision)
    previous_directory = Path.cwd()
    previous_provider = os.environ.get("INFRA_PROVIDER")
    os.environ["INFRA_PROVIDER"] = "1"
    os.chdir(snapshot)
    try:
        yield snapshot
    finally:
        os.chdir(previous_directory)
        if previous_provider is None:
            os.environ.pop("INFRA_PROVIDER", None)
        else:
            os.environ["INFRA_PROVIDER"] = previous_provider
