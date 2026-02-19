"""Load official ESM2 model from HuggingFace transformers for comparison."""

import torch
import torch.nn as nn
from transformers import EsmForMaskedLM, EsmTokenizer


class _OfficialESM2ForwardWrapper(nn.Module):
    def __init__(self, model: EsmForMaskedLM):
        super().__init__()
        self.model = model
        self.tokenizer = EsmTokenizer.from_pretrained(model.config._name_or_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return outputs


def load_official_model(
    reference_repo_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, EsmTokenizer]:
    """Load the official HuggingFace ESM2 model.

    Returns (wrapped_model, tokenizer).
    The wrapped model's forward returns standard HF outputs with hidden_states.
    """
    model = EsmForMaskedLM.from_pretrained(reference_repo_id).to(device=device, dtype=dtype).eval()
    tokenizer = EsmTokenizer.from_pretrained(reference_repo_id)
    wrapped = _OfficialESM2ForwardWrapper(model).to(device=device, dtype=dtype).eval()
    return wrapped, tokenizer
