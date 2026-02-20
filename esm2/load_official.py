"""Load official ESM2 model from HuggingFace transformers for comparison."""

import torch
import torch.nn as nn
from transformers import EsmForMaskedLM, EsmTokenizer


def load_official_model(
    reference_repo_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, EsmTokenizer]:
    """Load the official HuggingFace ESM2 model.

    Returns (wrapped_model, tokenizer).
    The wrapped model's forward returns standard HF outputs with hidden_states.
    """
    model = EsmForMaskedLM.from_pretrained(
        reference_repo_id,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
        position_embedding_type="rotary",
    ).eval()
    tokenizer = EsmTokenizer.from_pretrained(reference_repo_id)
    return model, tokenizer
