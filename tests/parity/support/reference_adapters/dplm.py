"""Official DPLM parity adapter backed by the pinned Bytedance source tree.

The adapter invokes ``DiffusionProteinLanguageModel.forward`` and only
normalizes the returned container. Forward hooks observe intermediate tensors;
they do not replace or reconstruct any upstream computation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from tests.parity.support.reference_adapters import (
    install_byprot_sequence_namespace,
    move_model,
    snapshot_path,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DPLM_SOURCE = _REPOSITORY_ROOT / "vendor" / "upstream" / "dplm" / "src"


def _install_source_path() -> None:
    if not _DPLM_SOURCE.is_dir():
        raise FileNotFoundError(
            "DPLM submodule is missing; run git submodule update --init --recursive"
        )
    source = str(_DPLM_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
    install_byprot_sequence_namespace(_DPLM_SOURCE)


class _OfficialDPLMForwardWrapper(nn.Module):
    """Expose Hugging Face-style output names around the official public API."""

    def __init__(self, oracle: nn.Module) -> None:
        super().__init__()
        self.oracle = oracle
        # The checkpoint conversion targets the official network state exactly.
        self.model = oracle.net
        self.tokenizer = oracle.tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        del attention_mask
        captured: list[torch.Tensor] = []

        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(value):
                captured.append(value)

        handles = [self.model.esm.embeddings.register_forward_hook(capture)]
        handles.extend(
            layer.register_forward_hook(capture)
            for layer in self.model.esm.encoder.layer[:-1]
        )
        try:
            # This is the upstream model's public inference entry point.
            logits, last_hidden_state = self.oracle(
                input_ids=input_ids,
                return_last_hidden_state=True,
            )
        finally:
            for handle in handles:
                handle.remove()

        hidden_states = tuple(captured)
        if not hidden_states or hidden_states[-1] is not last_hidden_state:
            hidden_states = (*hidden_states, last_hidden_state)
        return SimpleNamespace(
            logits=logits,
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
        )

    def generate(self, input_tokens: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Invoke the pinned implementation's public diffusion sampler."""

        return self.oracle.generate(input_tokens=input_tokens, **kwargs)


def load_official_model(
    reference_repo_id: str,
    reference_revision: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, object]:
    """Load DPLM through its pinned official ``from_pretrained`` method."""

    _install_source_path()
    # Register the exact official network class without triggering ByProt's
    # package-wide discovery of unrelated structure models.
    from byprot.models.dplm.dplm import DiffusionProteinLanguageModel
    from byprot.models.dplm.modules import dplm_modeling_esm as _dplm_modeling_esm

    del _dplm_modeling_esm

    snapshot = snapshot_path(reference_repo_id, reference_revision)
    oracle = DiffusionProteinLanguageModel.from_pretrained(str(snapshot))
    oracle = move_model(oracle, device, dtype).eval()
    wrapped = move_model(_OfficialDPLMForwardWrapper(oracle), device, dtype).eval()
    return wrapped, wrapped.tokenizer
