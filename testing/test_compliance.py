"""Weight and forward-pass compliance tests against original implementations.

Tests that FastPLM weights are bit-exact with the originals and that forward
pass outputs (logits, hidden states) are numerically equivalent.

Marked as `slow` because each test loads two models simultaneously.
"""

import importlib
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import pytest
import torch
from torch.nn.functional import mse_loss
from transformers import AutoModelForMaskedLM

from testing.conftest import (
    CANONICAL_AAS, FULL_MODEL_REGISTRY, MODEL_REGISTRY, SEED,
    add_model_specific_inputs, mark_by_size,
)
from fastplms.weight_parity_utils import assert_state_dict_equal


MODEL_KEYS = list(MODEL_REGISTRY.keys())

TEST_NUM_BATCHES = 25
BATCH_SIZE = 8
MIN_SEQ_LEN = 16
MAX_SEQ_LEN = 128

# bfloat16 accumulates numerical divergence across many layers; ESMC (30 layers)
# can show logits MSE ~0.02 and pred accuracy ~0.94 due to accumulated rounding.
LOGITS_MSE_THRESHOLD = 0.05
PREDS_ACCURACY_THRESHOLD = 0.90


def _generate_random_batch(batch_size: int, min_len: int, max_len: int) -> List[str]:
    return [
        "M" + "".join(random.choices(CANONICAL_AAS, k=random.randint(min_len, max_len)))
        for _ in range(batch_size)
    ]


def _load_models(
    model_key: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    registry: Dict[str, Dict] = None,
) -> Tuple[torch.nn.Module, torch.nn.Module, object]:
    """Load official and fast models for a given family.

    Returns (official_wrapped, fast_model, tokenizer).
    """
    if registry is None:
        registry = MODEL_REGISTRY
    config = registry[model_key]

    # Load official
    module = importlib.import_module(config["load_official"])
    official_model, tokenizer = module.load_official_model(
        reference_repo_id=config["official_path"],
        device=device,
        dtype=dtype,
    )

    # Load fast
    fast_model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"],
        trust_remote_code=True,
        dtype=dtype,
        device_map=device,
    ).eval()

    return official_model, fast_model, tokenizer


def _tokenize_batch(
    model_key: str,
    tokenizer: object,
    batch: List[str],
    device: torch.device,
    registry: Dict[str, Dict] = None,
) -> Dict[str, torch.Tensor]:
    """Tokenize a batch, handling E1's sequence mode."""
    if registry is None:
        registry = MODEL_REGISTRY
    config = registry[model_key]
    if config["model_type"] == "E1":
        tokenized = tokenizer.get_batch_kwargs(batch, device=device)
        return {
            "input_ids": tokenized["input_ids"],
            "within_seq_position_ids": tokenized["within_seq_position_ids"],
            "global_position_ids": tokenized["global_position_ids"],
            "sequence_ids": tokenized["sequence_ids"],
            "attention_mask": (tokenized["sequence_ids"] != -1).long(),
        }
    tokenized = tokenizer(batch, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in tokenized.items()}


def _run_weight_compliance(model_key: str, registry: Dict[str, Dict]) -> None:
    """Core weight compliance logic shared by default and full-registry tests."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        official_model, fast_model, _ = _load_models(model_key, device, dtype=torch.float32, registry=registry)
    except ModuleNotFoundError as e:
        pytest.skip(f"Dependency not installed for {model_key}: {e}")

    assert_state_dict_equal(
        reference_state_dict=official_model.model.state_dict(),
        candidate_state_dict=fast_model.state_dict(),
        context=f"{model_key} weight parity",
    )

    del official_model, fast_model
    torch.cuda.empty_cache()


def _run_forward_compliance(model_key: str, registry: Dict[str, Dict]) -> None:
    """Core forward compliance logic shared by default and full-registry tests."""
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = registry[model_key]
    model_type = config["model_type"]

    try:
        official_model, fast_model, tokenizer = _load_models(
            model_key, device, dtype=torch.bfloat16, registry=registry,
        )
    except ModuleNotFoundError as e:
        pytest.skip(f"Dependency not installed for {model_key}: {e}")

    cumulative_logits_mse = 0.0
    cumulative_preds_accuracy = 0.0
    hidden_state_diffs: Dict[int, float] = defaultdict(float)

    with torch.inference_mode():
        for _ in range(TEST_NUM_BATCHES):
            batch = _generate_random_batch(BATCH_SIZE, MIN_SEQ_LEN, MAX_SEQ_LEN)
            tokenized = _tokenize_batch(model_key, tokenizer, batch, device, registry=registry)
            attention_mask = tokenized["attention_mask"].cpu().bool()

            model_inputs = tokenized.copy()
            model_inputs = add_model_specific_inputs(model_inputs, model_type)

            official_output = official_model(**model_inputs, output_hidden_states=True)
            fast_output = fast_model(**model_inputs, output_hidden_states=True)

            official_logits = official_output.logits.cpu()
            fast_logits = fast_output.logits.cpu()

            # Compare on non-pad tokens only
            official_logits_masked = official_logits[attention_mask]
            fast_logits_masked = fast_logits[attention_mask]

            cumulative_logits_mse += mse_loss(official_logits_masked, fast_logits_masked).item()
            cumulative_preds_accuracy += (
                (official_logits_masked.argmax(dim=-1) == fast_logits_masked.argmax(dim=-1))
                .float()
                .mean()
                .item()
            )

            official_hidden = official_output.hidden_states
            fast_hidden = fast_output.hidden_states
            for i in range(min(len(official_hidden), len(fast_hidden))):
                off_h = official_hidden[i][attention_mask]
                fast_h = fast_hidden[i][attention_mask]
                hidden_state_diffs[i] += mse_loss(off_h, fast_h).item()

    avg_logits_mse = cumulative_logits_mse / TEST_NUM_BATCHES
    avg_preds_accuracy = cumulative_preds_accuracy / TEST_NUM_BATCHES

    if avg_logits_mse > LOGITS_MSE_THRESHOLD or avg_preds_accuracy < PREDS_ACCURACY_THRESHOLD:
        debug_lines = [f"Layer {k}: avg MSE = {v / TEST_NUM_BATCHES:.6f}" for k, v in sorted(hidden_state_diffs.items())]
        debug_msg = "\n".join(debug_lines)
        pytest.fail(
            f"{model_key} forward compliance failed:\n"
            f"  avg logits MSE = {avg_logits_mse:.6f} (threshold: {LOGITS_MSE_THRESHOLD})\n"
            f"  avg preds accuracy = {avg_preds_accuracy:.4f} (threshold: {PREDS_ACCURACY_THRESHOLD})\n"
            f"Per-layer hidden state MSE:\n{debug_msg}"
        )

    del official_model, fast_model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Default registry tests (small models, fast CI)
# ---------------------------------------------------------------------------

# DPLM2 original has an extra contact_head not present in the FastPLM version,
# so positional state_dict comparison fails. Skip weight compliance for DPLM2.
WEIGHT_COMPLIANCE_KEYS = [k for k in MODEL_KEYS if k != "dplm2"]

# DPLM2 original has structural differences (contact head, vocab mapping) that
# cause CUDA assertion failures when running through the ESM2 forward wrapper.
FORWARD_COMPLIANCE_KEYS = [k for k in MODEL_KEYS if k != "dplm2"]


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("model_key", WEIGHT_COMPLIANCE_KEYS)
def test_weight_compliance(model_key: str) -> None:
    """FastPLM weights are bit-exact with the original implementation."""
    _run_weight_compliance(model_key, MODEL_REGISTRY)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("model_key", FORWARD_COMPLIANCE_KEYS)
def test_forward_compliance(model_key: str) -> None:
    """FastPLM forward pass outputs match the original within tolerance."""
    _run_forward_compliance(model_key, MODEL_REGISTRY)


# ---------------------------------------------------------------------------
# Full registry tests (all checkpoints across all families)
# ---------------------------------------------------------------------------

FULL_WEIGHT_KEYS = [k for k in FULL_MODEL_REGISTRY if not k.startswith("dplm2")]
FULL_FORWARD_KEYS = [k for k in FULL_MODEL_REGISTRY if not k.startswith("dplm2")]


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_key",
    mark_by_size(FULL_WEIGHT_KEYS, FULL_MODEL_REGISTRY, extra_marks=[pytest.mark.slow]),
)
def test_full_weight_compliance(model_key: str) -> None:
    """Every checkpoint's weights are bit-exact with the original implementation."""
    _run_weight_compliance(model_key, FULL_MODEL_REGISTRY)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_key",
    mark_by_size(FULL_FORWARD_KEYS, FULL_MODEL_REGISTRY, extra_marks=[pytest.mark.slow]),
)
def test_full_forward_compliance(model_key: str) -> None:
    """Every checkpoint's forward pass matches the original within tolerance."""
    _run_forward_compliance(model_key, FULL_MODEL_REGISTRY)
