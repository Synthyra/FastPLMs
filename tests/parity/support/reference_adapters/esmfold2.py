"""Load ESMFold2 through the pinned Biohub Transformers implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from tests.parity.support.reference_adapters import move_model, snapshot_path
from tests.parity.support.reference_adapters.biohub_source import (
    reference_environment as _reference_environment,
)
from tests.parity.support.reference_adapters.biohub_source import (
    reference_sources,
)


reference_environment = _reference_environment


class _OfficialESMFold2Wrapper(nn.Module):
    """Expose projection-only output while delegating folding to Biohub."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def project_esmc_hidden_states(
        self,
        hidden_states: torch.Tensor,
        residue_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return Biohub's learned sequence summary before pair expansion."""

        # hidden_states: (..., d)
        shim = self.model.language_model
        captured: list[torch.Tensor] = []

        def capture_pair_input(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            if len(args) != 1:
                raise RuntimeError("Biohub base_z_mlp received an unexpected input signature.")
            captured.append(args[0])

        # Observe the official shim's public forward path at the boundary before
        # pair expansion. This avoids reproducing Biohub's learned projection.
        handle = shim.base_z_mlp.register_forward_pre_hook(capture_pair_input)
        try:
            shim(hidden_states, lm_dropout=0.0)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Biohub LanguageModelShim did not expose one sequence summary.")
        projected = captured[0]
        if residue_mask is not None:
            projected = projected * residue_mask.to(
                device=projected.device,
                dtype=projected.dtype,
            ).unsqueeze(-1)
        return projected


def load_official_model(
    reference_repo_id: str,
    reference_revision: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, None]:
    """Load one of the four supported ESMFold2 snapshots exactly."""

    reference_sources()
    from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from transformers.models.esmfold2.modeling_esmfold2_experimental import (
        ESMFold2ExperimentalModel,
    )

    snapshot = snapshot_path(reference_repo_id, reference_revision)
    config = ESMFold2Config.from_pretrained(snapshot, local_files_only=True)
    model_class = ESMFold2ExperimentalModel if config.type == "experimental" else ESMFold2Model
    load_kwargs = {
        "config": config,
        "local_files_only": True,
        "load_esmc": False,
    }
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    model = model_class.from_pretrained(snapshot, **load_kwargs)
    wrapped = _OfficialESMFold2Wrapper(model)
    return move_model(wrapped, device, dtype).eval(), None
