"""Official DPLM2 parity adapter backed by the pinned Bytedance source tree.

No FastPLMs loader, token remapping, or reconstructed forward pass is used.
The official tokenizer and multimodal model receive the inputs directly.
"""

from __future__ import annotations

import inspect
import sys
import torch
import torch.nn as nn
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

from tests.parity.support.reference_adapters import (
    OfficialGenerationUnavailable,
    install_byprot_sequence_namespace,
    move_model,
    snapshot_path,
)


DPLM2_3B_GENERATION_LIMITATION = {
    "status": "official_unavailable",
    "public_method": "EsmForDPLM.generate",
    "exception_type": "TypeError",
    "reason": (
        "The checkpoint-selected EsmForDPLM sampler uses tokenizer.cls_token_id "
        "as bos_id, but the pinned DPLM2 tokenizer defines no cls_token_id."
    ),
}

# Exact evidence from the pinned 150M official checkpoint. These trained head
# tensors are part of the release state contract and must never be replaced by
# random initialization when a mirror or local artifact is loaded.
DPLM2_150M_OFFICIAL_HEAD_CONTRACT = {
    "esm.contact_head.regression.bias": {
        "dtype": "torch.float32",
        "sha256": "bbce9db798883b08550850be32ea0a60cde4e06adb02e0a0ac686469a419311e",
        "shape": [1],
    },
    "esm.contact_head.regression.weight": {
        "dtype": "torch.float32",
        "sha256": "8037b6e221939baa4fdf62b3a89d9fd4a3b2430494daa85c58bb760e0514a6fc",
        "shape": [1, 600],
    },
    "esm.embeddings.word_embeddings.weight": {
        "dtype": "torch.float32",
        "sha256": "58662f66967b04570801ca4bd4c49c4bc610df0feadbe92128fe416a2fa23325",
        "shape": [8229, 640],
    },
    "lm_head.bias": {
        "dtype": "torch.float32",
        "sha256": "4ee4c69d1b4d6beea9c28a70d6440e5faff338b2e1cf927867b4e653cfa2f0f6",
        "shape": [8229],
    },
    "lm_head.decoder.weight": {
        "dtype": "torch.float32",
        "sha256": "25f3f82396eca43ba601bb7062458750a9b08177e7edc4ae24ae4d424ce2aea2",
        "shape": [8229, 640],
    },
    "lm_head.dense.bias": {
        "dtype": "torch.float32",
        "sha256": "e880655edb59ea7774c250b81870e9cefa85738258185c23d3b5846004a68daf",
        "shape": [640],
    },
    "lm_head.dense.weight": {
        "dtype": "torch.float32",
        "sha256": "f0dfa1b0a2d85e4cdc2e13727d21da29e94f09fb2267fdc3d3cfc5f2d0fd3450",
        "shape": [640, 640],
    },
    "lm_head.layer_norm.bias": {
        "dtype": "torch.float32",
        "sha256": "7c82d73e9be1b191ec39ffdfd5aa10b8da854d575daf1c334cdee261eb6406d3",
        "shape": [640],
    },
    "lm_head.layer_norm.weight": {
        "dtype": "torch.float32",
        "sha256": "edf9e1e32bec22be5eb81e2819e34324742eacf6f8cd5f410b1297705ab019b9",
        "shape": [640],
    },
}

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_DPLM_SOURCE = _REPOSITORY_ROOT / "vendor" / "upstream" / "dplm" / "src"


class _DPLM2Encoder(Protocol):
    """Static shape of the encoder fields used by the parity adapter."""

    layer: Sequence[nn.Module]


class _DPLM2Esm(Protocol):
    """Static shape of the ESM fields used by the parity adapter."""

    embeddings: nn.Module
    encoder: _DPLM2Encoder


class _DPLM2Generative(Protocol):
    """Static shape of the public sampler used by the parity adapter."""

    def generate(self, **kwargs: Any) -> Any: ...


class _DPLM2ModelWithEsm(Protocol):
    """Static shape of the checkpoint-selected model wrapper."""

    esm: _DPLM2Esm


def _install_source_path() -> None:
    if not _DPLM_SOURCE.is_dir():
        raise FileNotFoundError(
            "DPLM submodule is missing; run git submodule update --init --recursive"
        )
    source = str(_DPLM_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
    install_byprot_sequence_namespace(_DPLM_SOURCE)


def _field(output: Any, name: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _accepts_type_ids(module: nn.Module) -> bool:
    """Return whether the pinned network's public forward accepts ``type_ids``."""

    parameters = inspect.signature(module.forward).parameters.values()
    return any(
        parameter.name == "type_ids"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _call_checkpoint_forward(
    oracle: nn.Module,
    network: nn.Module,
    input_ids: torch.Tensor,
    kwargs: Mapping[str, Any],
) -> Any:
    """Call the supported public forward selected by the official checkpoint.

    The pinned DPLM2-3B checkpoint selects the official ``dplm_esm`` network.
    Its public forward does not accept ``type_ids``, while the official
    multimodal wrapper passes that keyword unconditionally. In that one case,
    the adapter invokes the checkpoint-selected public network forward with its
    supported signature. No hook is registered and no upstream object or class
    is modified.
    """

    # input_ids: (b, l)
    target = oracle if _accepts_type_ids(network) else network
    return target(input_ids=input_ids, **kwargs)


def _call_checkpoint_generate(
    oracle: nn.Module,
    network: nn.Module,
    input_tokens: torch.Tensor,
    kwargs: Mapping[str, Any],
) -> Any:
    """Call the checkpoint-selected implementation's public sampler.

    The DPLM2-3B checkpoint selects ``dplm_esm``. Its public sampler accepts a
    batch mapping, while the multimodal wrapper's sampler re-enters the broken
    ``type_ids`` forward path. Use the selected network's sampler directly for
    that checkpoint, passing only keywords declared by its public signature.
    Other DPLM2 checkpoints retain the multimodal wrapper's public sampler.
    """

    # input_tokens: (...)
    oracle_generate = cast(_DPLM2Generative, oracle).generate
    generate = getattr(network, "generate", None)
    if _accepts_type_ids(network) or not callable(generate):
        return oracle_generate(input_tokens=input_tokens, **kwargs)

    parameters = inspect.signature(generate).parameters
    supported_kwargs = {
        name: value for name, value in kwargs.items() if name in parameters
    }
    try:
        generated = generate(
            batch={"input_ids": input_tokens},
            **supported_kwargs,
        )
    except TypeError as error:
        is_pinned_public_failure = (
            type(network).__name__ == "EsmForDPLM"
            and getattr(network, "bos_id", object()) is None
            and "NoneType" in str(error)
            and "ne()" in str(error)
        )
        if not is_pinned_public_failure:
            raise
        raise OfficialGenerationUnavailable(
            public_method=DPLM2_3B_GENERATION_LIMITATION["public_method"],
            exception_type=type(error).__name__,
            reason=DPLM2_3B_GENERATION_LIMITATION["reason"],
        ) from error
    if isinstance(generated, tuple):
        return generated[0]
    return generated


class _OfficialDPLM2ForwardWrapper(nn.Module):
    """Normalize names while retaining the official public forward computation."""

    def __init__(self, oracle: nn.Module) -> None:
        super().__init__()
        self.oracle = oracle
        self.model = cast(nn.Module, oracle.net)
        self.tokenizer = oracle.tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        # input_ids: (b, l)
        del attention_mask
        captured: list[torch.Tensor] = []

        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(value):
                captured.append(value)

        esm = cast(_DPLM2ModelWithEsm, self.model).esm
        handles = [esm.embeddings.register_forward_hook(capture)]
        handles.extend(
            layer.register_forward_hook(capture)
            for layer in esm.encoder.layer[:-1]
        )
        try:
            output = _call_checkpoint_forward(
                self.oracle,
                self.model,
                input_ids,
                kwargs,
            )
        finally:
            for handle in handles:
                handle.remove()

        logits = _field(output, "logits")
        last_hidden_state = _field(output, "last_hidden_state")
        if logits is None or last_hidden_state is None:
            raise RuntimeError("Official DPLM2 output omitted logits or last_hidden_state")
        # The official multimodal wrapper calls the embedding block once, then
        # its inner ESM model calls the same block again. The second result is
        # the encoder's semantic input hidden state.
        # The multimodal wrapper computes embeddings before calling its inner
        # network, which computes them a second time. The pinned 3B checkpoint
        # selects the inner network directly, so that path has no duplicate.
        hidden_states = tuple(captured[1:] if _accepts_type_ids(self.model) else captured)
        if not hidden_states or hidden_states[-1] is not last_hidden_state:
            hidden_states = (*hidden_states, last_hidden_state)
        return SimpleNamespace(
            logits=logits,
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
        )

    def generate(self, input_tokens: torch.Tensor, **kwargs: Any) -> Any:
        """Invoke the checkpoint-selected implementation's public sampler."""

        # input_tokens: (...)
        return _call_checkpoint_generate(
            self.oracle,
            self.model,
            input_tokens,
            kwargs,
        )


def load_official_model(
    reference_repo_id: str,
    reference_revision: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, object]:
    """Load DPLM2 through its pinned official ``from_pretrained`` method."""

    _install_source_path()
    # The 3B DPLM2 checkpoint declares the upstream ``dplm_esm`` network
    # architecture, so its official registry module must be imported before
    # the public loader resolves the class name.
    from byprot.datamodules.dataset.tokenized_protein import DPLM2Tokenizer
    from byprot.models.dplm.modules import dplm_modeling_esm as _dplm_modeling_esm
    from byprot.models.dplm2.dplm2 import MultimodalDiffusionProteinLanguageModel
    from transformers import AutoTokenizer, EsmConfig

    del _dplm_modeling_esm
    # This checkpoint stores the official class name in tokenizer_config.json.
    # Registering that unmodified upstream class restores the public
    # AutoTokenizer lookup expected by the pinned DPLM loader.
    AutoTokenizer.register(  # type: ignore[no-untyped-call]
        EsmConfig,
        slow_tokenizer_class=DPLM2Tokenizer,
        exist_ok=True,
    )

    snapshot = snapshot_path(reference_repo_id, reference_revision)
    oracle = MultimodalDiffusionProteinLanguageModel.from_pretrained(str(snapshot))
    oracle = move_model(oracle, device, dtype).eval()
    wrapped = move_model(_OfficialDPLM2ForwardWrapper(oracle), device, dtype).eval()
    return wrapped, wrapped.tokenizer
