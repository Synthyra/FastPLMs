"""Load official DPLM2 model from the official/transformers submodule for comparison.

DPLM2 uses the ESM2 architecture internally, so the official weights load
directly via EsmForMaskedLM. The official/dplm submodule cannot be pip-installed
(pins incompatible torchtext==0.17.0), so we load via the transformers submodule.
"""

import torch
import torch.nn as nn
from typing import Tuple

from testing.official import use_transformers_submodule


class _OfficialDPLM2ForwardWrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )


def load_official_model(
    reference_repo_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[nn.Module, object]:
    """Load the official DPLM2 model (ESM2 architecture) from the transformers submodule.

    Returns (wrapped_model, tokenizer).
    """
    use_transformers_submodule()
    from transformers import EsmForMaskedLM, EsmTokenizer

    model = EsmForMaskedLM.from_pretrained(
        reference_repo_id,
        device_map=device,
        dtype=dtype,
    ).eval()
    tokenizer = EsmTokenizer.from_pretrained(reference_repo_id)
    wrapped = _OfficialDPLM2ForwardWrapper(model)
    return wrapped, tokenizer
