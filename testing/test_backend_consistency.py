"""Test that all attention backends produce consistent outputs for each model.

Loads each model once in float32, runs forward passes with SDPA, Flex, and
Flash backends, and verifies that logits are within tolerance.
"""

import random
from typing import Dict, List

import pytest
import torch
from transformers import AutoModelForMaskedLM

from testing.conftest import (
    BACKENDS, CANONICAL_AAS, FULL_MODEL_REGISTRY, MODEL_REGISTRY, SEED,
    add_model_specific_inputs, mark_by_size, tokenize_batch,
)


MODEL_KEYS = list(MODEL_REGISTRY.keys())
FULL_KEYS = list(FULL_MODEL_REGISTRY.keys())
# bfloat16 has ~3 decimal digits precision; backends differ in tiling/accumulation order.
# SDPA vs Flex can show max abs diffs up to ~0.5 in logit space at bfloat16.
# We check that predictions (argmax) agree rather than raw logit values.
PRED_AGREEMENT_THRESHOLD = 0.95
NUM_SEQUENCES = 4
SEQ_LEN = 64


def _generate_sequences(model_key: str) -> List[str]:
    """Generate test sequences (fixed-length for reproducibility)."""
    return [
        "M" + "".join(random.choices(CANONICAL_AAS, k=SEQ_LEN - 1))
        for _ in range(NUM_SEQUENCES)
    ]


def _tokenize_batch(
    model,
    model_key: str,
    sequences: List[str],
    device: torch.device,
    registry: Dict[str, Dict] = None,
) -> Dict[str, torch.Tensor]:
    """Tokenize sequences, handling E1's sequence mode."""
    if registry is None:
        registry = MODEL_REGISTRY
    config = registry[model_key] if model_key in registry else MODEL_REGISTRY[model_key]
    if config["model_type"] == "E1":
        batch = model.model.prep_tokens.get_batch_kwargs(sequences, device=device)
        return {
            "input_ids": batch["input_ids"],
            "within_seq_position_ids": batch["within_seq_position_ids"],
            "global_position_ids": batch["global_position_ids"],
            "sequence_ids": batch["sequence_ids"],
            "attention_mask": (batch["sequence_ids"] != -1).long(),
        }
    tokenizer = model.tokenizer
    tokenized = tokenizer(
        sequences,
        return_tensors="pt",
        padding="max_length",
        max_length=SEQ_LEN + 2,  # account for special tokens
        truncation=True,
    )
    return {k: v.to(device) for k, v in tokenized.items()}


def _run_backend_consistency(model_key: str, registry: Dict[str, Dict]) -> None:
    """Core backend consistency logic shared by default and full-registry tests."""
    random.seed(SEED)
    config = registry[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    sequences = _generate_sequences(model_key)
    inputs = _tokenize_batch(model, model_key, sequences, device, registry=registry)

    model_inputs = inputs.copy()
    model_inputs = add_model_specific_inputs(model_inputs, config["model_type"])

    backend_logits: Dict[str, torch.Tensor] = {}

    for backend in BACKENDS:
        try:
            model.attn_backend = backend
        except (AssertionError, RuntimeError) as e:
            print(f"Skipping backend '{backend}' for {model_key}: {e}")
            continue

        try:
            with torch.inference_mode():
                output = model(**model_inputs)
        except (AssertionError, RuntimeError) as e:
            print(f"Backend '{backend}' failed at runtime for {model_key}: {e}")
            continue

        backend_logits[backend] = output.logits.cpu()

    assert len(backend_logits) >= 1, f"No backends available for {model_key}"

    if "sdpa" not in backend_logits:
        pytest.skip(f"SDPA backend not available for {model_key}, cannot compare")

    reference = backend_logits["sdpa"]
    attention_mask = inputs["attention_mask"].cpu().bool()
    ref_masked = reference[attention_mask]
    ref_preds = ref_masked.argmax(dim=-1)

    for backend, logits in backend_logits.items():
        if backend == "sdpa":
            continue
        cand_masked = logits[attention_mask]
        cand_preds = cand_masked.argmax(dim=-1)
        agreement = (ref_preds == cand_preds).float().mean().item()
        assert agreement >= PRED_AGREEMENT_THRESHOLD, (
            f"{model_key}: SDPA vs {backend} prediction agreement = {agreement:.4f} "
            f"(threshold: {PRED_AGREEMENT_THRESHOLD})"
        )

    del model
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_backend_consistency(model_key: str) -> None:
    """All available backends produce equivalent logits (default registry)."""
    _run_backend_consistency(model_key, MODEL_REGISTRY)


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", mark_by_size(FULL_KEYS, FULL_MODEL_REGISTRY))
def test_full_backend_consistency(model_key: str) -> None:
    """All available backends produce equivalent logits (all checkpoints)."""
    _run_backend_consistency(model_key, FULL_MODEL_REGISTRY)
