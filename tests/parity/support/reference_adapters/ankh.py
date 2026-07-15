"""Load ANKH through the pinned official package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tests.parity.support.reference_adapters import move_model, snapshot_path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_ANKH_SUBMODULE = _REPOSITORY_ROOT / "vendor" / "upstream" / "ankh" / "src"


class _OfficialAnkhForwardWrapper(nn.Module):
    """Normalize encoder output names; ANKH encoders intentionally have no LM head."""

    def __init__(self, model: nn.Module, tokenizer: Any) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_kwargs: Any,
    ) -> Any:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )


def load_official_model(
    reference_repo_id: str,
    reference_revision: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, object]:
    """Load an official ANKH encoder without synthesizing an output head."""

    import sys

    if not _ANKH_SUBMODULE.is_dir():
        raise FileNotFoundError(
            "ANKH submodule is missing; run git submodule update --init --recursive"
        )
    source = str(_ANKH_SUBMODULE)
    if source not in sys.path:
        sys.path.insert(0, source)

    from ankh.models.ankh_transformers import get_specified_model
    from transformers import AutoTokenizer

    snapshot = snapshot_path(reference_repo_id, reference_revision)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = get_specified_model(
        path=str(snapshot),
        generation=False,
        output_attentions=False,
        framework="pt",
    )
    wrapped = _OfficialAnkhForwardWrapper(model, tokenizer)
    return move_model(wrapped, device, dtype).eval(), tokenizer


def load_official_seq2seq(
    reference_repo_id: str,
    reference_revision: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, object]:
    """Load ANKH's official sequence-to-sequence head at a pinned revision."""

    import sys

    if not _ANKH_SUBMODULE.is_dir():
        raise FileNotFoundError(
            "ANKH submodule is missing; run git submodule update --init --recursive"
        )
    source = str(_ANKH_SUBMODULE)
    if source not in sys.path:
        sys.path.insert(0, source)

    from ankh.models.ankh_transformers import get_specified_model
    from transformers import AutoTokenizer

    snapshot = snapshot_path(reference_repo_id, reference_revision)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = get_specified_model(
        path=str(snapshot),
        generation=True,
        output_attentions=False,
        framework="pt",
    )
    return move_model(model, device, dtype).eval(), tokenizer
