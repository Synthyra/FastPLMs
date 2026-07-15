"""Load Boltz2 through the pinned official Lightning checkpoint API."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from tests.parity.support.reference_adapters import move_model


def load_official_model(
    reference_repo_id: str,
    reference_revision: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, None]:
    """Load the official Boltz2 class from an immutable Hub revision."""

    from boltz.model.models.boltz2 import Boltz2
    from huggingface_hub import hf_hub_download

    checkpoint = Path(
        hf_hub_download(
            repo_id=reference_repo_id,
            filename="boltz2_conf.ckpt",
            revision=reference_revision,
        )
    )
    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        map_location="cpu",
    )
    return move_model(model, device, dtype).eval(), None
