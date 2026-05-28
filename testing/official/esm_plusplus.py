"""Load official ESMC model from the official/esm submodule for comparison."""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from testing.official import use_esm_submodule

use_esm_submodule()


class _ESMCComplianceOutput:
    """Mimics HuggingFace model output so the test suite can access .logits and .hidden_states."""
    def __init__(self, logits: torch.Tensor, last_hidden_state: torch.Tensor, hidden_states: Tuple[torch.Tensor, ...]) -> None:
        self.logits = logits
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states


class _OfficialESMCForwardWrapper(nn.Module):
    """Wraps official ESMC model to produce outputs compatible with our test suite."""
    def __init__(self, model: nn.Module, tokenizer: object) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sequence_id: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        esmc_output = self.model(sequence_tokens=input_ids)
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
) -> Tuple[nn.Module, object]:
    """Load the official ESMC model from the esm submodule.

    Args:
        reference_repo_id: e.g. "biohub/ESMC-300M" or "esmc-300"
        device: target device
        dtype: target dtype (should be float32 for comparison)

    Returns (wrapped_model, tokenizer).
    """
    from esm.models.esmc import ESMC
    from esm.tokenization import get_esmc_model_tokenizers
    from fastplms.esm_plusplus.modeling_esm_plusplus import (
        _ESMC_CHECKPOINT_SPECS,
        _load_safetensors_state_dict,
        _resolve_esmc_checkpoint_key,
        get_esmc_checkpoint_path,
    )

    key = _resolve_esmc_checkpoint_key(reference_repo_id)
    spec = _ESMC_CHECKPOINT_SPECS[key]
    with torch.device(device):
        official_model = ESMC(
            d_model=spec["hidden_size"],
            n_heads=spec["num_attention_heads"],
            n_layers=spec["num_hidden_layers"],
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=False,
        ).eval()
    _load_safetensors_state_dict(
        model_obj=official_model,
        checkpoint_path=get_esmc_checkpoint_path(reference_repo_id),
        device=device,
    )

    official_model = official_model.to(device=device, dtype=dtype).eval()
    tokenizer = official_model.tokenizer
    wrapped = _OfficialESMCForwardWrapper(official_model, tokenizer).to(device=device, dtype=dtype).eval()
    return wrapped, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_official_model(reference_repo_id="biohub/ESMC-300M", device=torch.device("cuda"), dtype=torch.float32)
    print(model)
    print(tokenizer)
