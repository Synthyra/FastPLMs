from types import SimpleNamespace

import pytest
import torch.nn as nn

from tests.parity.support.semantic_config import SEMANTIC_PATHS, semantic_config


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=8,
            classifier_dropout=0.1,
            initializer_range=0.02,
            tie_word_embeddings=True,
        )


def test_shared_semantic_config_includes_esmc_checkpoint_fields() -> None:
    assert {"classifier_dropout", "initializer_range", "tie_word_embeddings"}.issubset(
        SEMANTIC_PATHS
    )
    assert semantic_config(_Model()) == {
        "vocab_size": 64,
        "d_model": 32,
        "n_layers": 4,
        "n_heads": 8,
        "classifier_dropout": 0.1,
        "initializer_range": 0.02,
        "tie_word_embeddings": True,
    }


def test_shared_semantic_config_fails_closed_on_missing_required_field() -> None:
    model = _Model()
    del model.config.num_attention_heads
    with pytest.raises(RuntimeError, match="n_heads"):
        semantic_config(model)
