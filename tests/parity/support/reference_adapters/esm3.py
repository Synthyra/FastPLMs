"""Load ESM3 through the pinned Biohub implementation's public forward API."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from tests.parity.support.reference_adapters import (
    move_model,
    pinned_biohub_snapshot,
    use_esm_submodule,
)
from tests.parity.support.reference_adapters.biohub_source import (
    reference_environment as _reference_environment,
)
from tests.parity.support.reference_adapters.biohub_source import (
    reference_sources,
)

reference_environment = _reference_environment

use_esm_submodule()


class _ESM3ComplianceOutput:
    """Normalize official output names without changing official computation."""

    def __init__(self, output: Any, hidden_states: tuple[torch.Tensor, ...]) -> None:
        self.logits = output.sequence_logits
        self.last_hidden_state = output.embeddings
        self.hidden_states = hidden_states
        self.sequence_logits = output.sequence_logits
        self.structure_logits = output.structure_logits
        self.function_logits = output.function_logits
        self.residue_logits = output.residue_logits


class _ESM3StateDictRoot(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.esm3 = model


class _OfficialESM3ForwardWrapper(nn.Module):
    """Adapt Hugging Face-style names to the official ESM3 keyword names."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = _ESM3StateDictRoot(model)
        self.tokenizer = model.tokenizers.sequence

    @property
    def esm3(self) -> nn.Module:
        return self.model.esm3

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sequence_tokens: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> _ESM3ComplianceOutput:
        if sequence_tokens is None:
            sequence_tokens = input_ids
        if sequence_id is None and attention_mask is not None:
            sequence_id = attention_mask.to(dtype=torch.bool)

        captured: dict[str, tuple[torch.Tensor, ...]] = {}

        def capture_transformer_output(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: tuple[Any, ...],
        ) -> None:
            captured["hidden_states"] = tuple(output[2])

        handle = self.esm3.transformer.register_forward_hook(capture_transformer_output)
        try:
            output = self.esm3(
                sequence_tokens=sequence_tokens,
                sequence_id=sequence_id,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key
                    in {
                        "structure_tokens",
                        "ss8_tokens",
                        "sasa_tokens",
                        "function_tokens",
                        "residue_annotation_tokens",
                        "average_plddt",
                        "per_res_plddt",
                        "structure_coords",
                        "chain_id",
                        "output_attentions",
                    }
                },
            )
        finally:
            handle.remove()

        hidden_states = captured.get("hidden_states")
        if output_hidden_states and hidden_states is None:
            raise RuntimeError("Official ESM3 transformer did not expose hidden states")
        return _ESM3ComplianceOutput(output, hidden_states or ())


def _normalize_reference_repo_id(reference_repo_id: str) -> str:
    aliases = {
        "biohub/esm3-sm-open-v1": "esm3-sm-open-v1",
        "EvolutionaryScale/esm3-sm-open-v1": "esm3-sm-open-v1",
    }
    return aliases.get(reference_repo_id, reference_repo_id)


def load_official_model(
    reference_repo_id: str,
    reference_revision: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, object]:
    """Load official weights, then expose only name-normalized outputs."""

    reference_sources()
    from esm.pretrained import load_local_model
    from esm.utils.constants.models import normalize_model_name

    model_name = normalize_model_name(_normalize_reference_repo_id(reference_repo_id))
    # Load on CPU to avoid an implicit BF16 conversion before FP32 parity runs.
    with pinned_biohub_snapshot(reference_repo_id, reference_revision):
        model = load_local_model(model_name, device=torch.device("cpu"))
    model = move_model(model, device, dtype).eval()
    wrapped = move_model(_OfficialESM3ForwardWrapper(model), device, dtype).eval()
    return wrapped, wrapped.tokenizer
