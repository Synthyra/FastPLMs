"""Load official ESMC model from the esm package for comparison."""
import torch
import torch.nn as nn

class _ESMCComplianceOutput:
    """Mimics HuggingFace model output so the test suite can access .logits and .hidden_states."""
    def __init__(self, logits: torch.Tensor, last_hidden_state: torch.Tensor, hidden_states: tuple):
        self.logits = logits
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _OfficialESMCForwardWrapper(nn.Module):
    """Wraps official ESMC model to produce outputs compatible with our test suite."""
    def __init__(self, model: nn.Module, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        esmc_output = self.model(input_ids)
        # ESMC returns: sequence_logits, embeddings, hidden_states (stacked [n_layers, B, L, D])
        logits = esmc_output.sequence_logits
        embeddings = esmc_output.embeddings
        raw_hiddens = esmc_output.hidden_states
        # Convert stacked tensor to tuple for compatibility with hidden_states[-1]
        if raw_hiddens is not None:
            hidden_states = tuple(raw_hiddens[i] for i in range(raw_hiddens.shape[0]))
            hidden_states = hidden_states + (embeddings,)
        else:
            hidden_states = (embeddings,)
        return _ESMCComplianceOutput(
            logits=logits,
            last_hidden_state=embeddings,
            hidden_states=hidden_states,
        )


def load_official_model(
    reference_repo_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, object]:
    """Load the official ESMC model from the esm submodule.

    Args:
        reference_repo_id: e.g. "EvolutionaryScale/esmc-300m-2024-12"
        device: target device
        dtype: target dtype (should be float32 for comparison)

    Returns (wrapped_model, tokenizer).
    """
    from esm.pretrained import ESMC_300M_202412, ESMC_600M_202412

    if "300" in reference_repo_id:
        official_model = ESMC_300M_202412()
    elif "600" in reference_repo_id:
        official_model = ESMC_600M_202412()
    else:
        raise ValueError(f"Unsupported ESMC reference repo id: {reference_repo_id}")

    official_model = official_model.to(device=device, dtype=dtype).eval()
    tokenizer = official_model.tokenizer
    wrapped = _OfficialESMCForwardWrapper(official_model, tokenizer).to(device=device, dtype=dtype).eval()
    return wrapped, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_official_model(reference_repo_id="EvolutionaryScale/esmc-300m-2024-12", device=torch.device("cuda"), dtype=torch.float32)
    print(model)
    print(tokenizer)