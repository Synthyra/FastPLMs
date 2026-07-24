"""Load ESMC through the pinned Biohub Transformers public API."""

import torch
import torch.nn as nn

from tests.parity.support.reference_adapters import (
    move_model,
    snapshot_path,
)
from tests.parity.support.reference_adapters.biohub_source import (
    reference_environment as _reference_environment,
)
from tests.parity.support.reference_adapters.biohub_source import (
    reference_sources,
)


reference_environment = _reference_environment


def load_official_model(
    reference_repo_id: str,
    reference_revision: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, object]:
    """Load the official ESMC model from the pinned Biohub Transformers fork.

    Args:
        reference_repo_id: e.g. "biohub/ESMC-300M"
        device: target device
        dtype: target dtype (should be float32 for comparison)

    Returns (wrapped_model, tokenizer).
    """
    reference_sources()
    from transformers import AutoTokenizer
    from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM

    snapshot = snapshot_path(reference_repo_id, reference_revision)
    load_kwargs: dict[str, object] = {"local_files_only": True}
    if dtype is not None:
        load_kwargs["dtype"] = dtype
    official_model = ESMCForMaskedLM.from_pretrained(snapshot, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    return move_model(official_model, device, dtype).eval(), tokenizer
