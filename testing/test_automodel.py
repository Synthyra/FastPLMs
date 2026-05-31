"""Test that all FastPLM models load correctly via AutoModelForMaskedLM."""

import random

import pytest
import torch
from transformers import AutoModelForMaskedLM

from testing.conftest import (
    CANONICAL_AAS, FULL_MODEL_REGISTRY, MODEL_REGISTRY, SEED,
    add_model_specific_inputs, mark_by_size, tokenize_batch,
)


MODEL_KEYS = list(MODEL_REGISTRY.keys())
FULL_KEYS = list(FULL_MODEL_REGISTRY.keys())


def _tokenize_single(model, model_key: str, sequence: str, device: torch.device, registry=None):
    """Tokenize a single sequence, handling E1's sequence-mode separately."""
    return tokenize_batch(model, model_key, [sequence], device, registry=registry)


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_automodel_loads(model_key: str) -> None:
    """Model loads via AutoModelForMaskedLM with trust_remote_code=True."""
    config = MODEL_REGISTRY[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    assert model is not None
    # E1 uses sequence-mode (tokenizer lives on model.model.prep_tokens, not model.tokenizer)
    if config["uses_tokenizer"]:
        assert hasattr(model, "tokenizer")
    assert hasattr(model, "attn_backend")
    assert model.attn_backend == "sdpa"

    del model
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_automodel_forward_pass(model_key: str) -> None:
    """Single forward pass produces valid logits with no NaN."""
    random.seed(SEED)
    config = MODEL_REGISTRY[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    sequence = "M" + "".join(random.choices(CANONICAL_AAS, k=31))
    inputs = _tokenize_single(model, model_key, sequence, device)

    with torch.inference_mode():
        output = model(**inputs)

    assert hasattr(output, "logits")
    logits = output.logits
    assert logits.ndim == 3
    assert logits.shape[0] == 1
    assert not torch.isnan(logits).any(), f"NaN in logits for {model_key}"
    assert not torch.isinf(logits).any(), f"Inf in logits for {model_key}"

    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Full model registry tests: all checkpoints across all families
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("model_key", mark_by_size(FULL_KEYS, FULL_MODEL_REGISTRY))
def test_full_automodel_loads(model_key: str) -> None:
    """Every checkpoint loads via AutoModelForMaskedLM and exposes expected attributes."""
    config = FULL_MODEL_REGISTRY[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    assert model is not None
    if config["uses_tokenizer"]:
        assert hasattr(model, "tokenizer")
    assert hasattr(model, "attn_backend")
    assert model.attn_backend == "sdpa"

    del model
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", mark_by_size(FULL_KEYS, FULL_MODEL_REGISTRY))
def test_full_automodel_forward(model_key: str) -> None:
    """Every checkpoint produces valid logits on a single forward pass."""
    random.seed(SEED)
    config = FULL_MODEL_REGISTRY[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    sequence = "M" + "".join(random.choices(CANONICAL_AAS, k=31))
    inputs = _tokenize_single(model, model_key, sequence, device, registry=FULL_MODEL_REGISTRY)
    inputs = add_model_specific_inputs(inputs, config["model_type"])

    with torch.inference_mode():
        output = model(**inputs)

    assert hasattr(output, "logits")
    logits = output.logits
    assert logits.ndim == 3
    assert logits.shape[0] == 1
    assert not torch.isnan(logits).any(), f"NaN in logits for {model_key}"
    assert not torch.isinf(logits).any(), f"Inf in logits for {model_key}"

    del model
    torch.cuda.empty_cache()
