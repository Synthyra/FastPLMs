"""Load official DPLM model from the dplm/byprot submodule for comparison."""
import re
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional


class _OfficialComplianceOutput:
    def __init__(self, logits: torch.Tensor, last_hidden_state: torch.Tensor):
        self.logits = logits
        self.hidden_states = (last_hidden_state,)


class _OfficialDPLMComplianceWrapper(nn.Module):
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
        outputs = self.model(input_ids=input_ids, return_last_hidden_state=True)
        assert isinstance(outputs, tuple), f"Expected tuple from official DPLM, got {type(outputs)}."
        assert len(outputs) == 2, f"Expected 2-tuple from official DPLM, got {len(outputs)} values."
        logits, last_hidden_state = outputs
        return _OfficialComplianceOutput(logits=logits, last_hidden_state=last_hidden_state)


def _patch_py312_dplm_dataclass_default() -> None:
    candidate_paths = [
        Path("/app/dplm/src/byprot/models/dplm/dplm.py"),
        Path(__file__).resolve().parents[1] / "dplm" / "src" / "byprot" / "models" / "dplm" / "dplm.py",
    ]
    pattern = r"(?m)^(\s*lora\s*:\s*[^=\n]*LoRAConfig[^=\n]*=\s*).*$"

    for path in candidate_paths:
        if not path.exists():
            continue

        source = path.read_text()
        if "field(default_factory=LoRAConfig)" in source or "field(default_factory=lambda: LoRAConfig(" in source:
            return

        patched = source
        if "from dataclasses import dataclass, field" not in patched and "from dataclasses import dataclass" in patched:
            patched = patched.replace(
                "from dataclasses import dataclass",
                "from dataclasses import dataclass, field",
            )

        def _replace(match: re.Match[str]) -> str:
            return f"{match.group(1)}field(default_factory=LoRAConfig)"

        patched, replacements = re.subn(pattern, _replace, patched)
        if replacements > 0 and patched != source:
            path.write_text(patched)
            return


def load_official_model(
    reference_repo_id: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, object]:
    """Load the official DPLM model from the dplm/byprot submodule.

    Args:
        reference_repo_id: e.g. "airkingbd/dplm_150m"
        device: target device
        dtype: target dtype (should be float32 for comparison)

    Returns (wrapped_model, tokenizer).
    """
    _patch_py312_dplm_dataclass_default()
    from byprot.models.dplm.dplm import DiffusionProteinLanguageModel

    official_model = DiffusionProteinLanguageModel.from_pretrained(reference_repo_id).to(device=device, dtype=dtype).eval()
    wrapped = _OfficialDPLMComplianceWrapper(official_model)
    tokenizer = official_model.tokenizer
    return wrapped, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_official_model(reference_repo_id="airkingbd/dplm_150m", device=torch.device("cuda"), dtype=torch.float32)
    print(model)
    print(tokenizer)