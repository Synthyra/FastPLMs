"""Load ESMFold through Meta's public API and a hash-pinned native asset."""

from __future__ import annotations

import sys
import torch
import torch.nn as nn
from collections.abc import Sequence
from pathlib import Path

from tests.parity.support.reference_adapters import move_model, snapshot_path
from tests.parity.support.reference_adapters.esm2 import _asset_field, _verified_asset


_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_FAIR_ESM_SUBMODULE = _REPOSITORY_ROOT / "vendor" / "upstream" / "fair-esm"


def load_official_model(
    reference_repo_id: str,
    reference_revision: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
    oracle_assets: Sequence[object] = (),
) -> tuple[nn.Module, None]:
    """Load Meta ESMFold v1 without allowing its mutable download path."""

    if not _FAIR_ESM_SUBMODULE.is_dir():
        raise FileNotFoundError(
            "Meta ESM submodule is missing; run git submodule update --init --recursive"
        )
    source = str(_FAIR_ESM_SUBMODULE)
    if source not in sys.path:
        sys.path.insert(0, source)

    # The Hub revision records the immutable converted packaging checkpoint.
    # The live fair-esm oracle consumes Meta's independently hash-pinned `.pt`.
    snapshot_path(reference_repo_id, reference_revision)
    by_role = {str(_asset_field(asset, "role")): asset for asset in oracle_assets}
    if set(by_role) != {"weights"}:
        raise RuntimeError("ESMFold live parity requires its hash-pinned native weights")
    weights_path = _verified_asset(by_role["weights"])

    # `esm.pretrained.esmfold_v1()` is the public official constructor. Seed
    # its standard Torch Hub cache with the already verified file so the
    # constructor cannot resolve the mutable URL over the network.
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "esmfold_3B_v1.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() or checkpoint.is_symlink():
        checkpoint.unlink()
    checkpoint.symlink_to(weights_path)

    import esm

    model = esm.pretrained.esmfold_v1()
    return move_model(model, device, dtype).eval(), None
