"""Load official ANKH model (T5EncoderModel) from the official/transformers submodule for comparison."""

import torch
import torch.nn as nn
from typing import Tuple

from testing.official import use_transformers_submodule


class _AnkhComplianceOutput:
    """Mimics HuggingFace model output so the test suite can access .logits and .hidden_states."""

    def __init__(self, last_hidden_state: torch.Tensor, hidden_states: Tuple[torch.Tensor, ...]) -> None:
        self.logits = None  # T5EncoderModel has no LM head
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _OfficialAnkhForwardWrapper(nn.Module):
    def __init__(self, model, tokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return _AnkhComplianceOutput(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )


def load_official_model(
    reference_repo_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[nn.Module, object]:
    """Load the official ANKH model as a T5EncoderModel from the transformers submodule.

    The official ElnaggarLab repos are T5ForConditionalGeneration but
    T5EncoderModel.from_pretrained extracts just the encoder.

    Returns (wrapped_model, tokenizer).
    """
    use_transformers_submodule()
    from transformers import T5EncoderModel, AutoTokenizer

    model = T5EncoderModel.from_pretrained(
        reference_repo_id,
        device_map=device,
        dtype=dtype,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(reference_repo_id)
    wrapped = _OfficialAnkhForwardWrapper(model, tokenizer).to(device=device, dtype=dtype).eval()
    return wrapped, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_official_model(
        reference_repo_id="ElnaggarLab/ankh-base",
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    print(model)
    print(tokenizer)
