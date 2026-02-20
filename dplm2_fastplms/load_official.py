"""Load official DPLM2 model from the dplm/byprot submodule for comparison."""
from typing import Optional
import torch
import torch.nn as nn


class _OfficialComplianceOutput:
    def __init__(self, logits: torch.Tensor, last_hidden_state: torch.Tensor):
        self.logits = logits
        self.hidden_states = (last_hidden_state,)


class _OfficialDPLM2ComplianceWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.tokenizer = model.tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> _OfficialComplianceOutput:
        del attention_mask, output_hidden_states, output_attentions, kwargs
        outputs = self.model(input_ids=input_ids)
        assert isinstance(outputs, dict), f"Expected dict from official DPLM2, got {type(outputs)}."
        assert "logits" in outputs, "Official DPLM2 output is missing 'logits'."
        assert "last_hidden_state" in outputs, "Official DPLM2 output is missing 'last_hidden_state'."
        return _OfficialComplianceOutput(
            logits=outputs["logits"],
            last_hidden_state=outputs["last_hidden_state"],
        )


def load_official_model(
    reference_repo_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, object]:
    """Load the official DPLM2 model from the dplm/byprot submodule.

    Args:
        reference_repo_id: e.g. "airkingbd/dplm2_150m"
        device: target device
        dtype: target dtype (should be float32 for comparison)

    Returns (wrapped_model, tokenizer).
    """
    from dplm.src.byprot.models.dplm2.dplm2 import MultimodalDiffusionProteinLanguageModel

    official_model = MultimodalDiffusionProteinLanguageModel.from_pretrained(reference_repo_id).to(device=device, dtype=dtype).eval()
    wrapped = _OfficialDPLM2ComplianceWrapper(official_model)
    tokenizer = official_model.tokenizer
    return wrapped, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_official_model(reference_repo_id="airkingbd/dplm2_150m", device=torch.device("cuda"), dtype=torch.float32)
    print(model)
    print(tokenizer)