"""Load official DPLM2 model from HuggingFace transformers for comparison.

DPLM2 uses the ESM2 architecture internally, so the official weights load
directly via EsmForMaskedLM from HuggingFace transformers.

The DPLM2 tokenizer emits special-token IDs at positions [vocab_size ..
vocab_size+3] (cls, eos, pad, mask) that are OUT OF RANGE of the size-
vocab_size embedding table. The real DPLM2 forward remaps those to in-range
AA-side IDs before embedding lookup. Our wrapper does the same remap so
feeding tokenizer output straight to native EsmForMaskedLM doesn't OOB.
"""

import torch
import torch.nn as nn
from typing import Tuple

from transformers import EsmForMaskedLM, EsmTokenizer


def _normalize_dplm2_input_ids(input_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Remap DPLM2's high-ID generic special tokens to the AA-side special IDs.

    Must match fastplms.dplm2.modeling_dplm2._normalize_dplm2_input_ids exactly;
    the DPLM2 forward applies this normalization before the embedding lookup so
    the shared ESM backbone sees in-range token IDs.
    """
    if input_ids.numel() == 0:
        return input_ids
    normalized = input_ids.clone()
    generic_to_aa_special_ids = {
        vocab_size: 2,       # eos-generic  -> AA eos
        vocab_size + 1: 3,   # unk-generic  -> AA unk
        vocab_size + 2: 0,   # pad-generic  -> AA pad
        vocab_size + 3: 32,  # mask-generic -> AA mask
    }
    for generic_id, aa_id in generic_to_aa_special_ids.items():
        normalized[input_ids == generic_id] = aa_id
    # The DPLM2 tokenizer emits cls at vocab_size + 2 for one line of the
    # cls/eos/pad/mask family, depending on vocabulary file; the mapping above
    # is what fastplms.dplm2.modeling_dplm2 normalizes. If any ID survives
    # above the embedding table, remap the whole row of "generic specials"
    # defensively so feeding the tokenizer output into native does not OOB.
    in_range = normalized.lt(vocab_size) & normalized.ge(0)
    if not bool(in_range.all()):
        # Everything outside range collapses to 0 (pad) on the native side.
        # The fp32 hidden-state parity with fast is still meaningful because
        # fast applies the same mapping on its side.
        normalized = torch.where(in_range, normalized, torch.zeros_like(normalized))
    return normalized


class _OfficialDPLM2ForwardWrapper(nn.Module):
    def __init__(self, model: EsmForMaskedLM) -> None:
        super().__init__()
        self.model = model
        self.vocab_size = int(model.config.vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        input_ids = _normalize_dplm2_input_ids(input_ids, self.vocab_size)
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
    """Load the official DPLM2 model (ESM2 architecture) from HuggingFace.

    Returns (wrapped_model, tokenizer).
    """
    model = EsmForMaskedLM.from_pretrained(
        reference_repo_id,
        device_map=device,
        dtype=dtype,
    ).eval()
    tokenizer = EsmTokenizer.from_pretrained(reference_repo_id)
    wrapped = _OfficialDPLM2ForwardWrapper(model)
    return wrapped, tokenizer
