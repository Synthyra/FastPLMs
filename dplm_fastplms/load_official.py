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
    candidate_roots = [
        Path("/app/dplm/src/byprot/models/dplm"),
        Path(__file__).resolve().parents[1] / "dplm" / "src" / "byprot" / "models" / "dplm",
    ]
    pattern = r"(?m)^(\s*[A-Za-z_]\w*\s*:\s*([A-Za-z_]\w*Config)\s*=\s*).*$"

    for root in candidate_roots:
        if not root.exists():
            continue

        for path in root.glob("**/*.py"):
            source = path.read_text()
            patched = source

            def _ensure_field_import(match: re.Match[str]) -> str:
                imported_names = [item.strip() for item in match.group(1).split(",")]
                if "field" in imported_names:
                    return match.group(0)
                return f"from dataclasses import {match.group(1)}, field"

            patched = re.sub(
                r"(?m)^from dataclasses import ([^\n]+)$",
                _ensure_field_import,
                patched,
            )

            def _replace(match: re.Match[str]) -> str:
                prefix, config_type = match.group(1), match.group(2)
                return f"{prefix}field(default_factory={config_type})"

            patched, replacements = re.subn(pattern, _replace, patched)
            if replacements > 0 and patched != source:
                path.write_text(patched)


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
    from dplm_fastplms.modeling_dplm import DPLMConfig
    from byprot.models.dplm.dplm import DiffusionProteinLanguageModel

    config = DPLMConfig.from_pretrained(reference_repo_id)
    official_model = DiffusionProteinLanguageModel(cfg=config.to_dict())
    official_model.all_tied_weights_keys = {}
    official_model = official_model.from_pretrained(reference_repo_id).to(device=device, dtype=dtype).eval()
    wrapped = _OfficialDPLMComplianceWrapper(official_model)
    tokenizer = official_model.tokenizer
    return wrapped, tokenizer


if __name__ == "__main__":
    model, tokenizer = load_official_model(reference_repo_id="airkingbd/dplm_150m", device=torch.device("cuda"), dtype=torch.float32)
    print(model)
    print(tokenizer)