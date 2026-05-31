"""Load official DPLM model from HuggingFace transformers for comparison.

DPLM uses the ESM2 architecture internally, so the official weights load
directly via EsmForMaskedLM from HuggingFace transformers.
"""

import torch
import torch.nn as nn
from typing import Tuple

from transformers import EsmForMaskedLM, EsmTokenizer


class _OfficialDPLMForwardWrapper(nn.Module):
    def __init__(self, model: EsmForMaskedLM) -> None:
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
) -> Tuple[nn.Module, EsmTokenizer]:
    """Load the official DPLM model (ESM2 architecture) from HuggingFace.

    Returns (wrapped_model, tokenizer).
    """
    model = EsmForMaskedLM.from_pretrained(
        reference_repo_id,
        device_map=device,
        dtype=dtype,
    ).eval()
    tokenizer = EsmTokenizer.from_pretrained(reference_repo_id)
    wrapped = _OfficialDPLMForwardWrapper(model)
    return wrapped, tokenizer
