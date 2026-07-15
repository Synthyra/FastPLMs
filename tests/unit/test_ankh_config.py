"""ANKH configuration round-trip contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn

from fastplms.models.ankh.modeling_ankh import (
    FastAnkhConfig,
    FastAnkhForMaskedLMExtension,
    FastAnkhModel,
)


def _small_config() -> FastAnkhConfig:
    return FastAnkhConfig(
        vocab_size=32,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_heads=4,
        num_layers=2,
        is_encoder_decoder=True,
    )


def test_ankh_config_consumes_serialized_encoder_decoder_field() -> None:
    config = _small_config()

    restored = FastAnkhConfig.from_dict(config.to_dict())

    assert restored.is_encoder_decoder is True
    assert restored.to_dict() == config.to_dict()


def test_ankh_config_exposes_generic_t5_dimension_aliases() -> None:
    config = _small_config()

    assert config.hidden_size == config.d_model == 32
    assert config.head_dim == config.d_kv == 8
    assert config.num_attention_heads == config.num_heads == 4
    assert config.num_hidden_layers == config.num_layers == 2


@pytest.mark.parametrize("model_class", (FastAnkhModel, FastAnkhForMaskedLMExtension))
def test_ankh_encoder_embedding_alias_is_declared_and_safetensors_safe(
    model_class: type[FastAnkhModel] | type[FastAnkhForMaskedLMExtension],
    tmp_path: Path,
) -> None:
    model = model_class(_small_config())

    assert model.encoder.embed_tokens.weight is model.shared.weight
    assert model._tied_weights_keys == {"encoder.embed_tokens.weight": "shared.weight"}

    replacement = nn.Embedding(model.config.vocab_size, model.config.d_model)
    model.set_input_embeddings(replacement)
    assert model.get_input_embeddings() is replacement
    assert model.shared is replacement

    model.save_pretrained(tmp_path, safe_serialization=True)


def test_ankh_config_rejects_encoder_only_serialized_contract() -> None:
    with pytest.raises(ValueError, match="requires is_encoder_decoder=true"):
        FastAnkhConfig(is_encoder_decoder=False)
