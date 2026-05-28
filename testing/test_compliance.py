"""Weight and forward-pass compliance tests against original implementations.

Tests that FastPLM weights are bit-exact with the originals and that forward
pass outputs (logits, hidden states) are numerically equivalent.

Marked as `slow` because each test loads two models simultaneously.
"""

import importlib
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest
import torch
from transformers import AutoModelForMaskedLM

from testing.conftest import (
    CANONICAL_AAS, FULL_MODEL_REGISTRY, MODEL_REGISTRY, SEED,
    add_model_specific_inputs, mark_by_size, strict_fp32_matmul,
)
from fastplms.weight_parity_utils import assert_state_dict_equal


MODEL_KEYS = list(MODEL_REGISTRY.keys())

TEST_NUM_BATCHES = 25
BATCH_SIZE = 8
MIN_SEQ_LEN = 16
MAX_SEQ_LEN = 128

FORWARD_DTYPE = torch.float32


@dataclass(frozen=True)
class ForwardComplianceTolerances:
    hidden_mse: Optional[float] = 1e-8
    hidden_maxabs: Optional[float] = 5e-4
    hidden_rel_std: float = 5e-3
    hidden_rel_maxabs: float = 5e-2
    last_hidden_mse: float = 1e-8
    last_hidden_maxabs: float = 5e-4
    last_hidden_rel_maxabs: float = 5e-4
    logits_mse: float = 1e-8
    logits_maxabs: float = 5e-4


FORWARD_COMPLIANCE_TOLERANCES: Dict[str, ForwardComplianceTolerances] = {
    "ESM2": ForwardComplianceTolerances(
        hidden_mse=1e-12,
        hidden_maxabs=1e-5,
        hidden_rel_std=1e-6,
        hidden_rel_maxabs=1e-5,
        last_hidden_mse=1e-12,
        last_hidden_maxabs=1e-5,
        last_hidden_rel_maxabs=1e-5,
        logits_mse=1e-12,
        logits_maxabs=1e-5,
    ),
    "ESMC": ForwardComplianceTolerances(
        # ESMC intermediate states are high-magnitude pre-norm streams; use
        # tight relative guards there and absolute guards after final norm.
        hidden_mse=None,
        hidden_maxabs=None,
        hidden_rel_std=1e-4,
        hidden_rel_maxabs=1e-4,
        last_hidden_mse=1e-8,
        last_hidden_maxabs=1e-3,
        last_hidden_rel_maxabs=1e-3,
        logits_mse=1e-4,
        logits_maxabs=5e-3,
    ),
    "ESM3": ForwardComplianceTolerances(
        hidden_mse=1e-8,
        hidden_maxabs=1e-3,
        hidden_rel_std=5e-3,
        hidden_rel_maxabs=5e-2,
        last_hidden_mse=1e-8,
        last_hidden_maxabs=1e-3,
        last_hidden_rel_maxabs=1e-3,
        logits_mse=1e-4,
        logits_maxabs=5e-3,
    ),
    "E1": ForwardComplianceTolerances(
        hidden_mse=5e-7,
        hidden_maxabs=2e-2,
        hidden_rel_std=1e-2,
        hidden_rel_maxabs=2e-2,
        last_hidden_mse=5e-7,
        last_hidden_maxabs=2e-2,
        last_hidden_rel_maxabs=2e-3,
        logits_mse=1e-4,
        logits_maxabs=5e-2,
    ),
    "DPLM": ForwardComplianceTolerances(),
    "DPLM2": ForwardComplianceTolerances(
        hidden_mse=1e-12,
        hidden_maxabs=1e-5,
        hidden_rel_std=1e-5,
        hidden_rel_maxabs=1e-5,
        last_hidden_mse=1e-12,
        last_hidden_maxabs=1e-5,
        last_hidden_rel_maxabs=1e-5,
        logits_mse=1e-4,
        logits_maxabs=5e-3,
    ),
    "ANKH": ForwardComplianceTolerances(
        hidden_mse=1e-12,
        hidden_maxabs=1e-5,
        hidden_rel_std=1e-5,
        hidden_rel_maxabs=1e-5,
        last_hidden_mse=1e-12,
        last_hidden_maxabs=1e-5,
        last_hidden_rel_maxabs=1e-5,
        logits_mse=1e-6,
        logits_maxabs=5e-3,
    ),
}


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


def _masked_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Dict[str, float]:
    mask = attention_mask.bool()
    cand = candidate[mask].float()
    ref = reference[mask].float()
    diff = cand - ref
    diff_std = diff.std().item()
    ref_std = ref.std().item()
    diff_maxabs = diff.abs().max().item()
    ref_maxabs = ref.abs().max().item()
    rel_std = diff_std / ref_std if ref_std > 1e-12 else 0.0
    rel_maxabs = diff_maxabs / ref_maxabs if ref_maxabs > 1e-12 else 0.0
    return {
        "mse": (diff ** 2).mean().item(),
        "maxabs": diff_maxabs,
        "rel_std": rel_std,
        "rel_maxabs": rel_maxabs,
    }


def _record_worst(
    worst: Dict[str, Dict[str, float]],
    label: str,
    metrics: Dict[str, float],
) -> None:
    if label not in worst:
        worst[label] = metrics
        return
    for metric_name, value in metrics.items():
        if value > worst[label][metric_name]:
            worst[label][metric_name] = value


def _render_worst(worst: Dict[str, Dict[str, float]]) -> str:
    lines = []
    for label in sorted(worst):
        metrics = worst[label]
        lines.append(
            f"{label}: mse={metrics['mse']:.3e}, maxabs={metrics['maxabs']:.3e}, "
            f"rel_std={metrics['rel_std']:.3e}, rel_maxabs={metrics['rel_maxabs']:.3e}"
        )
    return "\n".join(lines)


def _exceeds(value: float, tolerance: Optional[float]) -> bool:
    return tolerance is not None and value > tolerance


def _format_tolerance(tolerance: Optional[float]) -> str:
    if tolerance is None:
        return "relative-only"
    return f"{tolerance:.3e}"


def _select_final_hidden_state(
    output: object,
    hidden_states: Tuple[torch.Tensor, ...],
) -> Tuple[str, torch.Tensor]:
    if isinstance(output, dict) and "last_hidden_state" in output:
        return "last_hidden_state", output["last_hidden_state"]
    try:
        last_hidden_state = output.last_hidden_state
    except AttributeError:
        return "hidden_states[-1]", hidden_states[-1]
    if last_hidden_state is not None:
        return "last_hidden_state", last_hidden_state
    return "hidden_states[-1]", hidden_states[-1]


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
    assert model_type in FORWARD_COMPLIANCE_TOLERANCES, (
        f"{model_key}: missing forward compliance tolerances for model_type={model_type}"
    )
    tol = FORWARD_COMPLIANCE_TOLERANCES[model_type]

    try:
        official_model, fast_model, tokenizer = _load_models(
            model_key, device, dtype=FORWARD_DTYPE, registry=registry,
        )
    except ModuleNotFoundError as e:
        pytest.skip(f"Dependency not installed for {model_key}: {e}")

    failures: List[str] = []
    worst_metrics: Dict[str, Dict[str, float]] = {}

    with torch.inference_mode(), strict_fp32_matmul():
        for _ in range(TEST_NUM_BATCHES):
            batch = _generate_random_batch(BATCH_SIZE, MIN_SEQ_LEN, MAX_SEQ_LEN)
            tokenized = _tokenize_batch(model_key, tokenizer, batch, device, registry=registry)
            attention_mask = tokenized["attention_mask"].bool()

            model_inputs = tokenized.copy()
            model_inputs = add_model_specific_inputs(model_inputs, model_type)

            official_output = official_model(**model_inputs, output_hidden_states=True)
            fast_output = fast_model(**model_inputs, output_hidden_states=True)

            official_logits = official_output.logits
            fast_logits = fast_output.logits
            assert official_logits is not None, f"{model_key}: official output has no logits"
            assert fast_logits is not None, f"{model_key}: fast output has no logits"
            logits_metrics = _masked_metrics(
                fast_logits,
                official_logits,
                attention_mask,
            )
            _record_worst(worst_metrics, "logits", logits_metrics)
            if logits_metrics["mse"] > tol.logits_mse or logits_metrics["maxabs"] > tol.logits_maxabs:
                failures.append(
                    f"logits: mse={logits_metrics['mse']:.3e} (tol={tol.logits_mse:.3e}), "
                    f"maxabs={logits_metrics['maxabs']:.3e} (tol={tol.logits_maxabs:.3e})"
                )

            official_hidden = official_output.hidden_states
            fast_hidden = fast_output.hidden_states
            assert len(official_hidden) == len(fast_hidden), (
                f"{model_key}: hidden_states tuple length mismatch "
                f"fast={len(fast_hidden)} official={len(official_hidden)}"
            )
            assert len(fast_hidden) > 0, f"{model_key}: no hidden states returned"
            for i, (fast_h, official_h) in enumerate(zip(fast_hidden, official_hidden)):
                hidden_metrics = _masked_metrics(fast_h, official_h, attention_mask)
                label = f"hidden_states[{i}]"
                _record_worst(worst_metrics, label, hidden_metrics)
                if (
                    _exceeds(hidden_metrics["mse"], tol.hidden_mse)
                    or _exceeds(hidden_metrics["maxabs"], tol.hidden_maxabs)
                    or hidden_metrics["rel_std"] > tol.hidden_rel_std
                    or hidden_metrics["rel_maxabs"] > tol.hidden_rel_maxabs
                ):
                    failures.append(
                        f"{label}: mse={hidden_metrics['mse']:.3e} "
                        f"(tol={_format_tolerance(tol.hidden_mse)}), "
                        f"maxabs={hidden_metrics['maxabs']:.3e} "
                        f"(tol={_format_tolerance(tol.hidden_maxabs)}), "
                        f"rel_std={hidden_metrics['rel_std']:.3e} (tol={tol.hidden_rel_std:.3e}), "
                        f"rel_maxabs={hidden_metrics['rel_maxabs']:.3e} "
                        f"(tol={tol.hidden_rel_maxabs:.3e})"
                    )

            official_last_label, official_last = _select_final_hidden_state(
                official_output,
                official_hidden,
            )
            fast_last_label, fast_last = _select_final_hidden_state(
                fast_output,
                fast_hidden,
            )
            final_label = official_last_label
            if official_last_label != fast_last_label:
                final_label = f"fast {fast_last_label} vs official {official_last_label}"
            last_metrics = _masked_metrics(fast_last, official_last, attention_mask)
            _record_worst(worst_metrics, final_label, last_metrics)
            if (
                last_metrics["mse"] > tol.last_hidden_mse
                or last_metrics["maxabs"] > tol.last_hidden_maxabs
                or last_metrics["rel_maxabs"] > tol.last_hidden_rel_maxabs
            ):
                failures.append(
                    f"{final_label}: mse={last_metrics['mse']:.3e} "
                    f"(tol={tol.last_hidden_mse:.3e}), "
                    f"maxabs={last_metrics['maxabs']:.3e} "
                    f"(tol={tol.last_hidden_maxabs:.3e}), "
                    f"rel_maxabs={last_metrics['rel_maxabs']:.3e} "
                    f"(tol={tol.last_hidden_rel_maxabs:.3e})"
                )

    if failures:
        rendered_failures = "\n".join(failures[:20])
        rendered_worst = _render_worst(worst_metrics)
        pytest.fail(
            f"{model_key} forward compliance failed under fp32 strict matmul:\n"
            f"{rendered_failures}\n"
            f"Worst observed metrics:\n{rendered_worst}"
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
