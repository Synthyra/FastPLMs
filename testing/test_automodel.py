"""Test that all FastPLM models load correctly via AutoModelForMaskedLM."""

import random

import pytest
import torch
from transformers import AutoModelForMaskedLM

from testing.conftest import CANONICAL_AAS, MODEL_REGISTRY, SEED


MODEL_KEYS = list(MODEL_REGISTRY.keys())


def _tokenize_single(model, model_key: str, sequence: str, device: torch.device):
    """Tokenize a single sequence, handling E1's sequence-mode separately."""
    if model_key == "e1":
        batch = model.model.prep_tokens.get_batch_kwargs([sequence], device=device)
        return {
            "input_ids": batch["input_ids"],
            "within_seq_position_ids": batch["within_seq_position_ids"],
            "global_position_ids": batch["global_position_ids"],
            "sequence_ids": batch["sequence_ids"],
            "attention_mask": (batch["sequence_ids"] != -1).long(),
        }
    tokenizer = model.tokenizer
    tokenized = tokenizer([sequence], return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in tokenized.items()}


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
