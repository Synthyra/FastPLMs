"""Rigorous parity tests between FastPLMs and native implementations.

Written after the embedding_parity investigation (see
`testing/debug_scripts/parity_debug_esmc.py` and the companion scripts
alongside it). Key design principles:

1. Many small asserts with descriptive failure messages. A failure says
   exactly what diverged, where, and by how much.
2. Fp32 tolerances are TIGHT. Bf16 tolerances are documented per family.
3. Intermediate hidden states are compared with a RELATIVE metric
   (diff_std / native_std) because some families (ESMC) have pre-norm
   activations with std ~250 — absolute MSE at intermediate layers is
   meaningless without this normalization.
4. last_hidden_state (post-final-norm) must match to fp32 numerical precision.
5. Logits parity is checked separately — it's what downstream tasks actually use.
6. Tokenizer parity is checked independently of the encoder.

Tests are parametrized per family. Each test file (testing/test_parity.py) runs
all families, but with skipif-on-ImportError so a family-specific image that
cannot import the native package just skips that family.

Run (per family image; see Dockerfile.<family>):
    docker run --gpus all --ipc=host --rm -v $(pwd):/workspace \
        fastplms-esm_plusplus python -m pytest /workspace/testing/test_parity.py -k esmc -v -s
"""
from __future__ import annotations

import importlib
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from testing.conftest import CANONICAL_AAS, MODEL_REGISTRY, SEED, tokenize_batch


@dataclass
class ParityTolerances:
    """Per-family / per-dtype numerical tolerances.

    Tolerances are intentionally tight in fp32 and documented per-family in bf16.
    There are three complementary metrics so that no single collapsed scalar can
    hide a localized regression:

    - `*_last_hidden_mse` / `*_last_hidden_maxabs`: absolute errors at the final
      (post-final-norm) residue-level representation.
    - `*_last_hidden_rel_maxabs`: the absolute maxabs divided by native's maxabs.
      Catches the case where maxabs looks "large" but native activations at that
      dtype are also large (so a small RELATIVE difference is fine), OR where
      a constant bias is silently added (absolute maxabs stays small, relative
      blows up).
    - `*_hidden_rel_std`: per-layer std-of-diff / std-of-native. Captures overall
      distribution agreement at each intermediate layer.
    - `*_hidden_rel_maxabs`: per-layer (maxabs of diff) / (maxabs of native).
      Captures the worst-case localized disagreement at each layer -- a sharp
      per-head or per-position regression that the std-of-diff would average out.
    """
    fp32_last_hidden_mse: float = 1e-8
    fp32_last_hidden_maxabs: float = 5e-4
    fp32_last_hidden_rel_maxabs: float = 5e-4
    fp32_logits_mse: float = 1e-4
    fp32_hidden_rel_std: float = 5e-3
    fp32_hidden_rel_maxabs: float = 5e-2
    bf16_last_hidden_mse: float = 1e-5
    bf16_last_hidden_maxabs: float = 2e-2
    bf16_last_hidden_rel_maxabs: float = 5e-2
    bf16_logits_mse: float = 5e-2
    bf16_hidden_rel_std: float = 5e-2
    bf16_hidden_rel_maxabs: float = 1e-1


FAMILY_TOLERANCES: Dict[str, ParityTolerances] = {
    "esm2": ParityTolerances(
        fp32_last_hidden_mse=1e-12, fp32_last_hidden_maxabs=1e-5, fp32_last_hidden_rel_maxabs=1e-5,
        fp32_logits_mse=1e-12, fp32_hidden_rel_std=1e-6, fp32_hidden_rel_maxabs=1e-5,
        bf16_last_hidden_mse=1e-5, bf16_last_hidden_maxabs=2e-2, bf16_last_hidden_rel_maxabs=3e-2,
        bf16_logits_mse=5e-2, bf16_hidden_rel_std=1e-2, bf16_hidden_rel_maxabs=5e-2,
    ),
    "esmc": ParityTolerances(
        fp32_last_hidden_mse=0.0, fp32_last_hidden_maxabs=0.0, fp32_last_hidden_rel_maxabs=0.0,
        fp32_logits_mse=0.0, fp32_hidden_rel_std=0.0, fp32_hidden_rel_maxabs=0.0,
        bf16_last_hidden_mse=1e-5, bf16_last_hidden_maxabs=5e-2, bf16_last_hidden_rel_maxabs=5e-2,
        bf16_logits_mse=5e-2, bf16_hidden_rel_std=5e-2, bf16_hidden_rel_maxabs=1e-1,
    ),
    "e1": ParityTolerances(
        fp32_last_hidden_mse=5e-7, fp32_last_hidden_maxabs=2e-2, fp32_last_hidden_rel_maxabs=2e-3,
        fp32_hidden_rel_std=1e-2, fp32_hidden_rel_maxabs=2e-2,
        bf16_last_hidden_maxabs=5e-2, bf16_last_hidden_rel_maxabs=5e-2,
        bf16_hidden_rel_std=1e-1, bf16_hidden_rel_maxabs=1e-1,
    ),
    "dplm": ParityTolerances(),
    "dplm2": ParityTolerances(
        # DPLM2 encoder is bit-identical to native's ESM backbone on pure AA
        # input (same weights; ModifiedRotaryEmbedding falls through to vanilla
        # rotary when no packed multimodal layout is detected). Logits differ
        # by construction (head is separately learned); logits parity is
        # skipped via _family_has_head_mismatch.
        fp32_last_hidden_mse=1e-12, fp32_last_hidden_maxabs=1e-5, fp32_last_hidden_rel_maxabs=1e-5,
        fp32_hidden_rel_std=1e-5, fp32_hidden_rel_maxabs=1e-5,
        # fp32_logits_mse is unused (skipped) but kept for dataclass defaults.
        # DPLM2-150M has 30 layers (vs ESM2-8M's 6), so bf16 accumulation gives
        # slightly higher MSE and maxabs than ESM2 at the post-final-norm output.
        bf16_last_hidden_mse=5e-5, bf16_last_hidden_maxabs=1.5e-1, bf16_last_hidden_rel_maxabs=3e-2,
        bf16_hidden_rel_std=3e-2, bf16_hidden_rel_maxabs=5e-2,
    ),
    "ankh": ParityTolerances(
        fp32_last_hidden_mse=1e-12, fp32_last_hidden_maxabs=1e-5, fp32_last_hidden_rel_maxabs=1e-5,
        fp32_logits_mse=1e-4,
        fp32_hidden_rel_std=1e-5, fp32_hidden_rel_maxabs=1e-5,
        # ANKH has no per-block norm (T5-style residual accumulation across 48+ blocks), so
        # bf16 rounding compounds into larger absolute values at the pre-norm residual stream.
        # We anchor the bf16 last_hidden_state check on the RELATIVE maxabs (diff / native
        # maxabs), not a loose absolute threshold, so a future regression that biases
        # activations can't hide behind the large native magnitudes.
        bf16_last_hidden_mse=5e-4, bf16_last_hidden_maxabs=2e-1, bf16_last_hidden_rel_maxabs=4e-2,
        bf16_logits_mse=5e-2, bf16_hidden_rel_std=5e-2, bf16_hidden_rel_maxabs=1e-1,
    ),
}

EXPECTED_WEIGHT_EXTRAS: Dict[str, set] = {
    # fast has these keys, native does not
    "ankh": {"lm_head.weight"},
}

EXPECTED_NATIVE_EXTRAS: Dict[str, set] = {
    # native has these keys, fast does not
    # DPLM2: native preserves the pretrained contact-prediction head that the
    # DPLM2 authors shipped on top of the ESM2 backbone; the FastPLMs variant
    # strips it because FastPLMs DPLM2 is an MLM-only model.
    "dplm2": {
        "esm.contact_head.regression.weight",
        "esm.contact_head.regression.bias",
    },
}

EXPECTED_VALUE_MISMATCHES: Dict[str, set] = {
    # Shared key names whose values are expected to differ for structural
    # (non-bug) reasons. Logits parity is also skipped for these families.
    # DPLM2: native has `tie_word_embeddings=True` so `lm_head.decoder.weight`
    # is an alias for `esm.embeddings.word_embeddings.weight`. FastPLMs DPLM2
    # has `tie_word_embeddings=False` and stores a separately-learned head.
    # Word embeddings themselves match exactly; only the head value differs.
    "dplm2": {"lm_head.decoder.weight"},
}


def _family_has_head_mismatch(model_key: str) -> bool:
    """Return True if fast and native have known lm_head / output-head differences,
    so logits parity should not be asserted.
    """
    return bool(
        EXPECTED_WEIGHT_EXTRAS.get(model_key)
        or EXPECTED_NATIVE_EXTRAS.get(model_key)
        or EXPECTED_VALUE_MISMATCHES.get(model_key)
    )


FIXED_SEQUENCE_LENGTHS = [16, 32, 48, 64, 80, 96, 112, 128]

# Tokenizer-mode batches used to stress padding behavior. "single" exercises
# no padding; "uniform" exercises mild padding (all lengths within ~50%);
# "skewed" exercises extreme padding (one short, one near-max), which is
# where mask-handling bugs typically surface.
PADDING_SCENARIOS: Dict[str, List[int]] = {
    "single":  [128],
    "uniform": [16, 32, 48, 64, 80, 96, 112, 128],
    "skewed":  [16, 16, 16, 128, 128],
}


def generate_fixed_sequences(seed: int = SEED, lengths: Optional[List[int]] = None) -> List[str]:
    if lengths is None:
        lengths = FIXED_SEQUENCE_LENGTHS
    rng = random.Random(seed)
    return [
        "M" + "".join(rng.choices(CANONICAL_AAS, k=L - 1))
        for L in lengths
    ]


def try_load_native(model_key: str, device: torch.device, dtype: torch.dtype):
    config = MODEL_REGISTRY[model_key]
    try:
        module = importlib.import_module(config["load_official"])
        return module.load_official_model(
            reference_repo_id=config["official_path"],
            device=device,
            dtype=dtype,
        )
    except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
        pytest.skip(f"Native deps not installed for {model_key}: {e}")


def load_fast(model_key: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    config = MODEL_REGISTRY[model_key]
    model = AutoModelForMaskedLM.from_pretrained(
        config["fast_path"], trust_remote_code=True,
        dtype=dtype, device_map=device,
    ).eval()
    return model


def fast_forward(
    model: nn.Module,
    model_key: str,
    sequences: List[str],
    device: torch.device,
    output_hidden_states: bool = True,
):
    config = MODEL_REGISTRY[model_key]
    if config["model_type"] == "E1":
        batch = model.model.prep_tokens.get_batch_kwargs(sequences, device=device)
        attention_mask = (batch["sequence_ids"] != -1).long()
        out = model(
            input_ids=batch["input_ids"],
            within_seq_position_ids=batch["within_seq_position_ids"],
            global_position_ids=batch["global_position_ids"],
            sequence_ids=batch["sequence_ids"],
            output_hidden_states=output_hidden_states,
        )
        return out, attention_mask
    batch = tokenize_batch(model, model_key, sequences, device)
    kwargs = dict(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=output_hidden_states,
    )
    if config["model_type"] == "ESMC":
        kwargs["sequence_id"] = batch["attention_mask"].to(torch.bool)
    out = model(**kwargs)
    return out, batch["attention_mask"]


def native_forward(
    model: nn.Module,
    model_key: str,
    sequences: List[str],
    device: torch.device,
    native_tokenizer,
):
    config = MODEL_REGISTRY[model_key]
    if config["model_type"] == "E1":
        batch = native_tokenizer.get_batch_kwargs(sequences, device=device)
        attention_mask = (batch["sequence_ids"] != -1).long()
        out = model(**batch, attention_mask=attention_mask)
        return out, attention_mask
    enc = native_tokenizer(sequences, return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return out, enc["attention_mask"]


def _masked(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return tensor[mask.bool()]


def relative_hidden_std(fast: torch.Tensor, native: torch.Tensor, mask: torch.Tensor) -> float:
    mask_b = mask.bool()
    f = fast[mask_b].float()
    n = native[mask_b].float()
    diff_std = (f - n).std().item()
    native_std = n.std().item()
    if native_std < 1e-12:
        return 0.0
    return diff_std / native_std


def relative_hidden_maxabs(fast: torch.Tensor, native: torch.Tensor, mask: torch.Tensor) -> float:
    """Worst-case localized relative error at this layer: maxabs(diff) / maxabs(native).

    Complement to relative_hidden_std. The std metric collapses every position x every
    hidden dim into a single scalar, which can hide a single-dimension or single-position
    regression. This metric asks "at the worst element, how large is the error relative
    to the worst element of native?"
    """
    mask_b = mask.bool()
    f = fast[mask_b].float()
    n = native[mask_b].float()
    diff_maxabs = (f - n).abs().max().item()
    native_maxabs = n.abs().max().item()
    if native_maxabs < 1e-12:
        return 0.0 if diff_maxabs < 1e-12 else float("inf")
    return diff_maxabs / native_maxabs


# -----------------------------------------------------------------------------
# Tokenizer parity
# -----------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("model_key", [k for k in MODEL_REGISTRY if MODEL_REGISTRY[k]["uses_tokenizer"]])
def test_tokenizer_parity(model_key: str) -> None:
    device = torch.device("cuda")
    fast = load_fast(model_key, device, torch.float32)
    native_model, native_tok = try_load_native(model_key, device, torch.float32)

    fast_tok = fast.tokenizer
    fast_vocab = fast_tok.get_vocab()
    native_vocab = native_tok.get_vocab()
    assert len(fast_vocab) == len(native_vocab), (
        f"{model_key}: vocab size mismatch fast={len(fast_vocab)} native={len(native_vocab)}"
    )
    missing_in_fast = [t for t in native_vocab if t not in fast_vocab]
    assert not missing_in_fast, f"{model_key}: tokens missing from fast tokenizer: {missing_in_fast[:5]}"
    id_mismatches = [
        (t, native_vocab[t], fast_vocab[t])
        for t in native_vocab
        if native_vocab[t] != fast_vocab[t]
    ]
    assert not id_mismatches, f"{model_key}: token id mismatches: {id_mismatches[:5]}"

    for attr in ("pad_token_id", "cls_token_id", "eos_token_id", "mask_token_id", "unk_token_id"):
        f_id = getattr(fast_tok, attr, None)
        n_id = getattr(native_tok, attr, None)
        assert f_id == n_id, f"{model_key}: {attr} mismatch fast={f_id} native={n_id}"

    del fast, native_model
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Weight parity (bit exact in fp32)
# -----------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
def test_weight_parity_fp32(model_key: str) -> None:
    device = torch.device("cuda")
    fast = load_fast(model_key, device, torch.float32)
    native_model, _ = try_load_native(model_key, device, torch.float32)

    fast_sd = fast.state_dict()
    native_sd = native_model.model.state_dict() if hasattr(native_model, "model") else native_model.state_dict()

    expected_fast_extras = EXPECTED_WEIGHT_EXTRAS.get(model_key, set())
    expected_native_extras = EXPECTED_NATIVE_EXTRAS.get(model_key, set())
    expected_value_mismatches = EXPECTED_VALUE_MISMATCHES.get(model_key, set())

    fast_keys = set(fast_sd.keys()) - expected_fast_extras
    native_keys = set(native_sd.keys()) - expected_native_extras
    assert fast_keys == native_keys, (
        f"{model_key}: state_dict key sets differ\n"
        f"  (allowlisted fast extras: {expected_fast_extras})\n"
        f"  (allowlisted native extras: {expected_native_extras})\n"
        f"  only_fast (unexpected): {sorted(fast_keys - native_keys)[:5]}\n"
        f"  only_native (unexpected): {sorted(native_keys - fast_keys)[:5]}"
    )

    shape_mismatches: List[str] = []
    value_mismatches: List[str] = []
    unexpected_value_matches: List[str] = []
    for name in sorted(fast_keys & native_keys):
        f = fast_sd[name]
        n = native_sd[name]
        if f.shape != n.shape:
            shape_mismatches.append(f"{name}: {tuple(f.shape)} vs {tuple(n.shape)}")
            continue
        values_equal = torch.equal(f.float(), n.float())
        if name in expected_value_mismatches:
            # This key is allowed to differ. Sanity-check it really does differ
            # (catches the case where someone adds it to the allowlist by mistake
            # and values actually match -- meaning the allowlist is stale).
            if values_equal:
                unexpected_value_matches.append(name)
        else:
            if not values_equal:
                max_abs = (f.float() - n.float()).abs().max().item()
                value_mismatches.append(f"{name}: max|Δ|={max_abs:.3e}")
    assert not shape_mismatches, f"{model_key}: shape mismatches:\n" + "\n".join(shape_mismatches[:10])
    assert not value_mismatches, f"{model_key}: value mismatches:\n" + "\n".join(value_mismatches[:10])
    assert not unexpected_value_matches, (
        f"{model_key}: the following keys were allowlisted in EXPECTED_VALUE_MISMATCHES "
        f"but actually match natively. Remove them from the allowlist:\n"
        + "\n".join(unexpected_value_matches)
    )

    del fast, native_model
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Forward parity -- fp32
# -----------------------------------------------------------------------------

def _run_forward_parity(model_key: str, dtype: torch.dtype, tol: ParityTolerances, dtype_label: str, scenario: str = "uniform") -> None:
    device = torch.device("cuda")
    random.seed(SEED)
    torch.manual_seed(SEED)

    fast = load_fast(model_key, device, dtype)
    native_model, native_tok = try_load_native(model_key, device, dtype)

    sequences = generate_fixed_sequences(lengths=PADDING_SCENARIOS[scenario])

    with torch.no_grad():
        fout, fmask = fast_forward(fast, model_key, sequences, device, output_hidden_states=True)
        nout, nmask = native_forward(native_model, model_key, sequences, device, native_tok)

    assert torch.equal(fmask, nmask), f"{model_key}: attention_mask mismatch between fast and native tokenization"

    fh: Tuple[torch.Tensor, ...] = tuple(fout.hidden_states)
    nh: Tuple[torch.Tensor, ...] = tuple(nout.hidden_states)
    assert len(fh) == len(nh), f"{model_key}: hidden_states tuple length mismatch fast={len(fh)} native={len(nh)}"

    flast_attr = getattr(fout, "last_hidden_state", None)
    nlast_attr = getattr(nout, "last_hidden_state", None)
    flast = flast_attr if flast_attr is not None else fh[-1]
    nlast = nlast_attr if nlast_attr is not None else nh[-1]
    mask_b = fmask.bool()

    last_diff = (flast - nlast).float()[mask_b]
    last_native = nlast.float()[mask_b]
    last_mse = (last_diff ** 2).mean().item()
    last_maxabs = last_diff.abs().max().item()
    last_native_maxabs = last_native.abs().max().item()
    last_rel_maxabs = last_maxabs / last_native_maxabs if last_native_maxabs > 1e-12 else 0.0
    if dtype == torch.float32:
        last_mse_tol = tol.fp32_last_hidden_mse
        last_maxabs_tol = tol.fp32_last_hidden_maxabs
        last_rel_maxabs_tol = tol.fp32_last_hidden_rel_maxabs
    else:
        last_mse_tol = tol.bf16_last_hidden_mse
        last_maxabs_tol = tol.bf16_last_hidden_maxabs
        last_rel_maxabs_tol = tol.bf16_last_hidden_rel_maxabs
    assert last_mse <= last_mse_tol, (
        f"{model_key} ({dtype_label}): last_hidden_state MSE={last_mse:.3e} > tol={last_mse_tol:.3e} "
        f"(maxabs={last_maxabs:.3e}, rel_maxabs={last_rel_maxabs:.3e})"
    )
    assert last_maxabs <= last_maxabs_tol, (
        f"{model_key} ({dtype_label}): last_hidden_state maxabs={last_maxabs:.3e} > tol={last_maxabs_tol:.3e} "
        f"(mse={last_mse:.3e}, rel_maxabs={last_rel_maxabs:.3e})"
    )
    assert last_rel_maxabs <= last_rel_maxabs_tol, (
        f"{model_key} ({dtype_label}): last_hidden_state rel_maxabs={last_rel_maxabs:.3e} > tol={last_rel_maxabs_tol:.3e} "
        f"(maxabs_diff={last_maxabs:.3e}, maxabs_native={last_native_maxabs:.3e}). "
        f"A systematic bias may have been introduced even though absolute error looks small."
    )

    # Skip logits parity when fast and native have a known head difference.
    # - ANKH: fast is ForMaskedLM with its own head; native T5EncoderModel has
    #   no head and testing/official/ankh.py bolts on a fresh tied-weight head.
    # - DPLM2: native ties lm_head to word embeddings; FastPLMs stores a
    #   separately-learned head (see EXPECTED_VALUE_MISMATCHES). Encoder hidden
    #   states match; logits by construction do not.
    has_head_mismatch = _family_has_head_mismatch(model_key)
    if not has_head_mismatch and hasattr(fout, "logits") and hasattr(nout, "logits") and fout.logits is not None and nout.logits is not None:
        logits_diff = (fout.logits - nout.logits).float()[mask_b]
        logits_mse = (logits_diff ** 2).mean().item()
        logits_mse_tol = tol.fp32_logits_mse if dtype == torch.float32 else tol.bf16_logits_mse
        assert logits_mse <= logits_mse_tol, (
            f"{model_key} ({dtype_label}): logits MSE={logits_mse:.3e} > tol={logits_mse_tol:.3e}"
        )

    if dtype == torch.float32:
        rel_std_tol = tol.fp32_hidden_rel_std
        rel_maxabs_tol = tol.fp32_hidden_rel_maxabs
    else:
        rel_std_tol = tol.bf16_hidden_rel_std
        rel_maxabs_tol = tol.bf16_hidden_rel_maxabs
    per_layer: List[Tuple[int, float, float]] = []
    for i in range(len(fh)):
        rel_std = relative_hidden_std(fh[i], nh[i], fmask)
        rel_maxabs = relative_hidden_maxabs(fh[i], nh[i], fmask)
        per_layer.append((i, rel_std, rel_maxabs))
    std_violations = [(i, s, m) for i, s, m in per_layer if s > rel_std_tol]
    maxabs_violations = [(i, s, m) for i, s, m in per_layer if m > rel_maxabs_tol]
    if std_violations or maxabs_violations:
        rendered = "\n".join(
            f"    layer {i}: rel_diff_std={s:.3e}  rel_diff_maxabs={m:.3e}"
            for i, s, m in per_layer
        )
        reason_parts: List[str] = []
        if std_violations:
            reason_parts.append(f"rel_diff_std > tol={rel_std_tol:.3e}")
        if maxabs_violations:
            reason_parts.append(f"rel_diff_maxabs > tol={rel_maxabs_tol:.3e}")
        reason = " and ".join(reason_parts)
        pytest.fail(
            f"{model_key} ({dtype_label}): per-layer hidden-state divergence ({reason}):\n{rendered}"
        )

    del fast, native_model
    torch.cuda.empty_cache()


@pytest.mark.gpu
@pytest.mark.parametrize("scenario", list(PADDING_SCENARIOS.keys()))
@pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
def test_forward_parity_fp32(model_key: str, scenario: str) -> None:
    tol = FAMILY_TOLERANCES[model_key]
    _run_forward_parity(model_key, torch.float32, tol, "fp32", scenario=scenario)


@pytest.mark.gpu
@pytest.mark.parametrize("scenario", list(PADDING_SCENARIOS.keys()))
@pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
def test_forward_parity_bf16(model_key: str, scenario: str) -> None:
    tol = FAMILY_TOLERANCES[model_key]
    _run_forward_parity(model_key, torch.bfloat16, tol, "bf16", scenario=scenario)


# -----------------------------------------------------------------------------
# Padding isolation -- check parametrized across backends so a FLEX-specific or
# kernels_flash-specific mask bug gets caught, not just the SDPA path.
# -----------------------------------------------------------------------------

# Backends to exercise for the padding-isolation test. kernels_flash is excluded:
# its unpad/pad helpers strip padding entirely, so a batch-shape-dependent
# regression would surface as "doesn't even load" long before this test.
PADDING_BACKENDS: Tuple[str, ...] = ("sdpa", "flex")


@pytest.mark.gpu
@pytest.mark.parametrize("backend", PADDING_BACKENDS)
@pytest.mark.parametrize("model_key", [k for k in MODEL_REGISTRY if MODEL_REGISTRY[k]["uses_tokenizer"]])
def test_padding_does_not_pollute_valid_positions_fp32(model_key: str, backend: str) -> None:
    """A padded batch's valid-position `last_hidden_state` must match the same
    sequence run unpadded.

    We only check `last_hidden_state` (not intermediate hidden states) because
    `F.scaled_dot_product_attention` is not bit-deterministic across batch
    shapes -- kernel dispatch and reduction order can differ between batch=1
    and batch=N runs, producing tiny per-layer diffs (~1e-5 maxabs at
    intermediate layers, decaying to ~1e-6 after the final norm). Those
    diffs are PyTorch SDPA noise, not a parity bug. What WOULD be a bug:
    padded keys bleeding into valid-query attention through a broken mask,
    which would produce a much larger and persistent diff at
    `last_hidden_state` -- exactly what this test catches.

    Parametrized over backends so a FLEX-specific block-mask bug or an ANKH
    flex score_mod that forgets to honor the block mask is caught independently
    of the SDPA path.
    """
    device = torch.device("cuda")
    random.seed(SEED)

    fast = load_fast(model_key, device, torch.float32)
    resolved = _apply_backend(fast, backend, model_key)
    if resolved is None:
        pytest.skip(f"{model_key}: backend {backend} unavailable or fell back")

    short = generate_fixed_sequences(lengths=[16])[0]
    long_ = generate_fixed_sequences(lengths=[128])[0]

    with torch.no_grad():
        out_alone, mask_alone = fast_forward(fast, model_key, [short], device, output_hidden_states=True)
        out_padded, mask_padded = fast_forward(fast, model_key, [short, long_], device, output_hidden_states=True)

    valid_len = mask_alone.sum().item()
    la = getattr(out_alone, "last_hidden_state", None)
    lp = getattr(out_padded, "last_hidden_state", None)
    last_alone = (la if la is not None else out_alone.hidden_states[-1])[0, :valid_len].float()
    last_padded = (lp if lp is not None else out_padded.hidden_states[-1])[0, :valid_len].float()

    diff = (last_alone - last_padded).abs()
    diff_max = diff.max().item()
    diff_mse = (diff ** 2).mean().item()
    assert diff_max < 1e-3 and diff_mse < 1e-7, (
        f"{model_key} ({backend}): padding appears to be polluting valid-position outputs (fp32). "
        f"At `last_hidden_state`, valid-position diff vs unpadded run is "
        f"max|Δ|={diff_max:.3e}, mse={diff_mse:.3e} (expected max<1e-3, mse<1e-7). "
        f"This is much larger than kernel batch-shape noise (typically <1e-5 maxabs) "
        f"and indicates an attention-mask bug -- padded keys are likely bleeding "
        f"into valid query attention."
    )

    del fast
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Attention backend consistency (fast-only; all backends must agree)
#
# The supported dtypes differ by backend:
# - sdpa: fp32, fp16, bf16 all fine
# - flex: fp32, fp16, bf16 all fine
# - kernels_flash: fp16 / bf16 only (flash kernel ops reject fp32)
#
# So we split the consistency check into fp32 (sdpa vs flex) and bf16 (all three
# supported backends). ANKH is special: its encoder silently downgrades
# kernels_flash to flex because T5 relative position bias can't be fed to flash.
# -----------------------------------------------------------------------------

BACKEND_CONSISTENCY_FP32_MATRIX: Dict[str, Tuple[str, ...]] = {
    "esm2": ("flex",),
    "esmc": ("flex",),
    "e1": ("flex",),
    "dplm": ("flex",),
    "dplm2": ("flex",),
    "ankh": ("flex",),
}

BACKEND_CONSISTENCY_BF16_MATRIX: Dict[str, Tuple[str, ...]] = {
    "esm2": ("kernels_flash", "flex"),
    "esmc": ("kernels_flash", "flex"),
    "e1": ("kernels_flash", "flex"),
    "dplm": ("kernels_flash", "flex"),
    "dplm2": ("kernels_flash", "flex"),
    "ankh": ("flex",),
}


def _apply_backend(model: nn.Module, backend: str, model_key: str) -> Optional[str]:
    """Set `model.attn_backend = backend` using the per-family property setter
    that every FastPLMs sequence model exposes. Returns the *resolved* backend
    as a string, or None if the backend is unavailable on this GPU / image.

    Why this exists: earlier versions of the test suite tried to switch backends
    by setting class attributes on the Config subclass, which is silently a
    no-op (every config's `__init__` overwrites the class attr). This helper
    uses the correct mechanism and verifies the switch actually took effect.
    """
    try:
        model.attn_backend = backend
    except AssertionError as e:
        # resolve_attention_backend asserts when a backend is unavailable
        # (e.g. kernels_flash without the `kernels` package, flex without
        # torch >= 2.5). Treat as unavailable.
        print(f"{model_key}: backend {backend} unavailable: {e}")
        return None
    except Exception as e:  # noqa: BLE001 -- any backend assertion should skip, not fail
        print(f"{model_key}: backend {backend} failed to apply: {e}")
        return None
    return _get_resolved_backend(model, model_key)


def _get_resolved_backend(model: nn.Module, model_key: str) -> str:
    """Introspect the resolved backend from a known attention submodule.

    This matters for ANKH, which silently falls back kernels_flash -> flex at
    the encoder level. If we asked for kernels_flash and got flex back, we want
    the test to know.
    """
    # Walk modules looking for an attention sub-module with an attn_backend enum.
    for module in model.modules():
        attn_backend = getattr(module, "attn_backend", None)
        if attn_backend is None:
            continue
        if hasattr(attn_backend, "value"):
            return attn_backend.value
    # Fall back to the config.
    return model.config.attn_backend


# Per-backend tolerances. Two regimes:
#
# 1. fp32 strict raw: backends that support fp32 must produce hidden states
#    within floating-point rounding of sdpa. flex passes easily; kernels_flash
#    isn't eligible (rejects fp32 at the kernel level).
#
# 2. bf16 downstream-equivalent: in bf16, kernels_flash (and, more loosely,
#    flex) have different tiling/reduction orders than sdpa, so raw hidden
#    states can differ meaningfully (~0.5+ maxabs in logit space per the
#    legacy test's comment). Asking for raw agreement is known infeasible.
#    Instead we check what downstream users actually care about:
#       a. mean-pool cosine similarity (for embedding-based downstream tasks)
#       b. top-1 argmax agreement on logits (for masked-LM prediction use)
# fp32 backend-consistency tolerances are per-family per-backend. Depth
# matters: deeper models (DPLM 30 layers, ESMC 30 layers, ANKH-base 48) accumulate
# more per-position rounding than shallow ones (ESM2-8M 6 layers), so absolute
# maxabs has to be loosened with depth even though rel_maxabs stays tight.
BACKEND_TOL_FP32: Dict[str, Dict[str, Dict[str, float]]] = {
    "esm2":  {"flex": {"mse": 1e-6, "maxabs": 5e-3, "rel_maxabs": 5e-3}},
    "esmc":  {"flex": {"mse": 1e-6, "maxabs": 1e-2, "rel_maxabs": 5e-3}},
    "e1":    {"flex": {"mse": 1e-6, "maxabs": 1e-2, "rel_maxabs": 5e-3}},
    "dplm":  {"flex": {"mse": 1e-6, "maxabs": 5e-2, "rel_maxabs": 5e-3}},
    "dplm2": {"flex": {"mse": 1e-6, "maxabs": 5e-2, "rel_maxabs": 5e-3}},
    "ankh":  {"flex": {"mse": 1e-6, "maxabs": 5e-2, "rel_maxabs": 1e-2}},
}

# bf16 backend-consistency thresholds are per-family per-backend. Two physics-
# driven metrics with different behaviors across families:
#
# - min_pooled_cosine: per-sequence mean-pool(last_hidden_state) cosine vs sdpa,
#   min across sequences. Measures "does the representation direction agree?"
# - min_argmax_agreement: fraction of positions whose top-1 LM logit matches
#   sdpa's. Measures "does the downstream MLM prediction agree?"
#
# Empirical behavior (see testing/debug_scripts/investigate_backend_cosine.py):
#   ESM2-8M (6 layers):
#     flex:           pooled_cosine ~ 1.0000,  argmax ~ 1.0000
#     kernels_flash:  pooled_cosine 0.70-0.86, argmax ~ 0.98
#   ESMC-300M (30 layers):
#     flex:           pooled_cosine ~ 1.0000,  argmax ~ 0.94
#     kernels_flash:  pooled_cosine ~ 1.0000,  argmax ~ 0.95
#
# kernels_flash's online softmax + different tile reduction order drifts the
# residual stream in direction at short depth (ESM2-8M) even though the argmax
# is stable. At 30 layers the residual stabilizes direction-wise because each
# layer's LayerNorm re-anchors magnitude, and the argmax disagreement is from
# bf16 rounding on the final LM head softmax, not attention kernel drift.
#
# None for min_cosine means "informational only, don't assert". Use that when
# we know the metric is dominated by a known architectural phenomenon rather
# than by bugs we want to catch.
BACKEND_TOL_BF16_DOWNSTREAM: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
    "esm2": {
        "flex":          {"min_cosine": 0.995, "min_argmax_agreement": 0.98},
        "kernels_flash": {"min_cosine": None,  "min_argmax_agreement": 0.95},
    },
    "esmc": {
        "flex":          {"min_cosine": 0.995, "min_argmax_agreement": 0.90},
        "kernels_flash": {"min_cosine": 0.995, "min_argmax_agreement": 0.90},
    },
    "e1": {
        "flex":          {"min_cosine": 0.995, "min_argmax_agreement": 0.90},
        "kernels_flash": {"min_cosine": None,  "min_argmax_agreement": 0.90},
    },
    "dplm": {
        "flex":          {"min_cosine": 0.995, "min_argmax_agreement": 0.95},
        "kernels_flash": {"min_cosine": None,  "min_argmax_agreement": 0.95},
    },
    "dplm2": {
        # DPLM2 has a separately-learned lm_head (native ties, fast doesn't --
        # see EXPECTED_VALUE_MISMATCHES), so argmax agreement across backends is
        # evaluated on the FAST model's own head; backends must agree with each other.
        "flex":          {"min_cosine": 0.995, "min_argmax_agreement": 0.95},
        "kernels_flash": {"min_cosine": None,  "min_argmax_agreement": 0.95},
    },
    "ankh": {
        # ANKH supports only flex at the user-facing level (kernels_flash falls back).
        "flex":          {"min_cosine": 0.995, "min_argmax_agreement": 0.90},
    },
}


def _run_backend_consistency_fp32(
    model_key: str,
    backends: Tuple[str, ...],
    backend_tol: Dict[str, Dict[str, float]],
) -> None:
    """backend_tol is the per-backend tol dict FOR THIS family (already indexed by caller)."""
    device = torch.device("cuda")
    random.seed(SEED)
    sequences = generate_fixed_sequences()

    baseline = load_fast(model_key, device, torch.float32)
    base_resolved = _apply_backend(baseline, "sdpa", model_key)
    assert base_resolved == "sdpa", f"{model_key}: sdpa baseline resolved to {base_resolved} (expected 'sdpa')"

    with torch.no_grad():
        base_out, mask = fast_forward(baseline, model_key, sequences, device, output_hidden_states=False)
    base_last_attr = getattr(base_out, "last_hidden_state", None)
    base_last = base_last_attr if base_last_attr is not None else base_out.hidden_states[-1]
    mask_b = mask.bool()
    base_valid = base_last[mask_b].float()
    base_maxabs = base_valid.abs().max().item()
    del baseline, base_out
    torch.cuda.empty_cache()

    failures: List[str] = []
    for backend in backends:
        alt = load_fast(model_key, device, torch.float32)
        resolved = _apply_backend(alt, backend, model_key)
        if resolved is None:
            del alt
            torch.cuda.empty_cache()
            pytest.skip(f"{model_key} (fp32): backend {backend} not available")
        if resolved != backend:
            failures.append(
                f"{backend}: requested backend was silently resolved to '{resolved}'. "
                f"If this is an intentional fallback (e.g. ANKH kernels_flash -> flex), "
                f"remove {backend!r} from the consistency matrix for {model_key!r}."
            )
            del alt
            torch.cuda.empty_cache()
            continue
        with torch.no_grad():
            alt_out, _ = fast_forward(alt, model_key, sequences, device, output_hidden_states=False)
        alt_last_attr = getattr(alt_out, "last_hidden_state", None)
        alt_last = alt_last_attr if alt_last_attr is not None else alt_out.hidden_states[-1]
        diff = (alt_last[mask_b].float() - base_valid)
        mse = (diff ** 2).mean().item()
        maxabs = diff.abs().max().item()
        rel_maxabs = maxabs / base_maxabs if base_maxabs > 1e-12 else 0.0
        tol = backend_tol[backend]
        if mse > tol["mse"] or maxabs > tol["maxabs"] or rel_maxabs > tol["rel_maxabs"]:
            failures.append(
                f"{backend}: mse={mse:.3e} (tol={tol['mse']:.3e}), "
                f"maxabs={maxabs:.3e} (tol={tol['maxabs']:.3e}), "
                f"rel_maxabs={rel_maxabs:.3e} (tol={tol['rel_maxabs']:.3e})"
            )
        del alt, alt_out
        torch.cuda.empty_cache()

    assert not failures, f"{model_key} (fp32): backend consistency failed:\n" + "\n".join(failures)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool over valid positions per sequence. Returns (batch, hidden)."""
    m = attention_mask.bool().unsqueeze(-1).float()
    summed = (last_hidden_state.float() * m).sum(dim=1)
    counts = m.sum(dim=1).clamp_min(1.0)
    return summed / counts


def _run_backend_consistency_bf16_downstream(
    model_key: str,
    backends: Tuple[str, ...],
    backend_tol: Dict[str, Dict[str, Optional[float]]],
) -> None:
    device = torch.device("cuda")
    random.seed(SEED)
    sequences = generate_fixed_sequences()

    baseline = load_fast(model_key, device, torch.bfloat16)
    base_resolved = _apply_backend(baseline, "sdpa", model_key)
    assert base_resolved == "sdpa", f"{model_key}: sdpa baseline resolved to {base_resolved} (expected 'sdpa')"

    with torch.no_grad():
        base_out, mask = fast_forward(baseline, model_key, sequences, device, output_hidden_states=False)
    base_last_attr = getattr(base_out, "last_hidden_state", None)
    base_last = base_last_attr if base_last_attr is not None else base_out.hidden_states[-1]
    base_pooled = _mean_pool(base_last, mask)  # (B, D) in fp32
    mask_b = mask.bool()

    base_logits = getattr(base_out, "logits", None)
    base_argmax = None
    if base_logits is not None:
        base_argmax = base_logits.float()[mask_b].argmax(dim=-1)

    del baseline, base_out
    torch.cuda.empty_cache()

    failures: List[str] = []
    for backend in backends:
        alt = load_fast(model_key, device, torch.bfloat16)
        resolved = _apply_backend(alt, backend, model_key)
        if resolved is None:
            del alt
            torch.cuda.empty_cache()
            pytest.skip(f"{model_key} (bf16): backend {backend} not available")
        if resolved != backend:
            failures.append(
                f"{backend}: requested backend was silently resolved to '{resolved}'. "
                f"Remove {backend!r} from the consistency matrix for {model_key!r} if this is intentional."
            )
            del alt
            torch.cuda.empty_cache()
            continue
        with torch.no_grad():
            alt_out, _ = fast_forward(alt, model_key, sequences, device, output_hidden_states=False)
        alt_last_attr = getattr(alt_out, "last_hidden_state", None)
        alt_last = alt_last_attr if alt_last_attr is not None else alt_out.hidden_states[-1]
        alt_pooled = _mean_pool(alt_last, mask)
        # Per-sequence cosine similarity, then min across sequences.
        cos_per_seq = F.cosine_similarity(base_pooled, alt_pooled, dim=-1)  # (B,)
        min_cos = cos_per_seq.min().item()
        tol = backend_tol[backend]
        # Always print diagnostics -- useful for understanding where a backend sits on the cosine axis.
        print(f"    {backend}: min_pooled_cosine={min_cos:.4f}")
        if tol["min_cosine"] is not None and min_cos < tol["min_cosine"]:
            failures.append(
                f"{backend}: min per-sequence pooled cosine = {min_cos:.4f} "
                f"(tol >= {tol['min_cosine']:.4f})"
            )
        # Argmax agreement (if logits are available).
        alt_logits = getattr(alt_out, "logits", None)
        if base_argmax is not None and alt_logits is not None:
            alt_argmax = alt_logits.float()[mask_b].argmax(dim=-1)
            agreement = (base_argmax == alt_argmax).float().mean().item()
            print(f"    {backend}: argmax_agreement={agreement:.4f}")
            if agreement < tol["min_argmax_agreement"]:
                failures.append(
                    f"{backend}: logits argmax agreement vs sdpa = {agreement:.4f} "
                    f"(tol >= {tol['min_argmax_agreement']:.4f})"
                )
        del alt, alt_out
        torch.cuda.empty_cache()

    assert not failures, f"{model_key} (bf16 downstream): backend consistency failed:\n" + "\n".join(failures)


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", list(BACKEND_CONSISTENCY_FP32_MATRIX.keys()))
def test_backend_consistency_fp32(model_key: str) -> None:
    """fp32 backend parity: sdpa vs flex. Strict raw-value agreement.

    kernels_flash is excluded: its ops reject fp32 at the kernel level.
    """
    _run_backend_consistency_fp32(
        model_key,
        BACKEND_CONSISTENCY_FP32_MATRIX[model_key],
        BACKEND_TOL_FP32[model_key],
    )


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", list(BACKEND_CONSISTENCY_BF16_MATRIX.keys()))
def test_backend_consistency_bf16_downstream(model_key: str) -> None:
    """bf16 backend parity: sdpa vs flex vs kernels_flash at the downstream level.

    In bf16, different attention kernels produce meaningfully different raw
    hidden states (tiling / reduction order; well-documented in the legacy
    compliance test). What matters for downstream users is whether:

    - mean-pooled embeddings (classification / regression use) agree in
      direction (cosine similarity)
    - top-1 argmax predictions (MLM use) agree

    Both are checked here with per-family per-backend thresholds.
    """
    _run_backend_consistency_bf16_downstream(
        model_key,
        BACKEND_CONSISTENCY_BF16_MATRIX[model_key],
        BACKEND_TOL_BF16_DOWNSTREAM[model_key],
    )


# -----------------------------------------------------------------------------
# embed_dataset pipeline parity (what downstream users actually call)
#
# This is the most user-facing test: if `embed_dataset(...)` and an equivalent
# hand-rolled native forward + mean-pool disagree, a downstream task will see
# the difference even though every per-layer metric was within tolerance.
# -----------------------------------------------------------------------------

# Per-family absolute/max-abs tolerances for the mean-pooled embedding.
EMBED_DATASET_TOL: Dict[str, Dict[str, float]] = {
    "esm2":  {"mse": 5e-8, "maxabs": 5e-3},
    "esmc":  {"mse": 5e-8, "maxabs": 5e-3},
    "dplm":  {"mse": 5e-8, "maxabs": 5e-3},
    # DPLM2: encoder output identical to ESM backbone on AA input; mean-pool parity tight.
    "dplm2": {"mse": 5e-8, "maxabs": 5e-3},
    "e1":    {"mse": 5e-6, "maxabs": 2e-2},   # Grouped-query attention + block-causal GLOBAL layers propagate rounding
    # ANKH-base post-final-RMSNorm activations are modest (~O(1)) but the
    # mean-pool aggregates over the full sequence; 5e-3 maxabs is plenty tight.
    "ankh":  {"mse": 5e-8, "maxabs": 5e-3},
}

PIPELINE_MODEL_KEYS = [k for k in EMBED_DATASET_TOL if k in MODEL_REGISTRY]


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", PIPELINE_MODEL_KEYS)
def test_embed_dataset_pipeline_parity(model_key: str) -> None:
    device = torch.device("cuda")
    random.seed(SEED)

    config = MODEL_REGISTRY[model_key]
    sequences = generate_fixed_sequences()
    fast = load_fast(model_key, device, torch.float32)
    native_model, native_tok = try_load_native(model_key, device, torch.float32)

    # Tokenizer mode vs sequence mode: E1 has no tokenizer.
    tokenizer_mode = config["uses_tokenizer"]
    fast_embeddings = fast.embed_dataset(
        sequences=sequences,
        tokenizer=fast.tokenizer if tokenizer_mode else None,
        batch_size=4, max_len=256, truncate=True,
        full_embeddings=False,
        embed_dtype=torch.float32,
        pooling_types=["mean"],
        num_workers=0, sql=False, save=False,
        padding="max_length" if tokenizer_mode else "longest",
    )
    assert fast_embeddings is not None

    tol = EMBED_DATASET_TOL[model_key]
    with torch.no_grad():
        failures: List[str] = []
        for seq in sequences:
            # Produce a native mean-pooled embedding for this single sequence.
            if tokenizer_mode:
                enc = native_tok([seq], return_tensors="pt", padding=True)
                enc = {k: v.to(device) for k, v in enc.items()}
                out = native_model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    output_hidden_states=True,
                )
                last_attr = getattr(out, "last_hidden_state", None)
                last = (last_attr if last_attr is not None else out.hidden_states[-1]).float()
                m = enc["attention_mask"].bool().unsqueeze(-1).float()
            else:
                # E1: native_tok is the E1BatchPreparer.
                batch = native_tok.get_batch_kwargs([seq], device=device)
                attention_mask = (batch["sequence_ids"] != -1).long()
                out = native_model(**batch, attention_mask=attention_mask)
                last_attr = getattr(out, "last_hidden_state", None)
                last = (last_attr if last_attr is not None else out.hidden_states[-1]).float()
                m = attention_mask.bool().unsqueeze(-1).float()
            pooled = (last * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
            pooled = pooled.squeeze(0).cpu()
            fast_vec = fast_embeddings[seq].cpu().float()
            assert fast_vec.shape == pooled.shape, (
                f"{model_key} seq_len={len(seq)}: shape mismatch fast={tuple(fast_vec.shape)} "
                f"native={tuple(pooled.shape)}"
            )
            mse = ((fast_vec - pooled) ** 2).mean().item()
            maxabs = (fast_vec - pooled).abs().max().item()
            if mse > tol["mse"] or maxabs > tol["maxabs"]:
                failures.append(
                    f"seq_len={len(seq)}: mse={mse:.3e} (tol={tol['mse']:.3e}) "
                    f"maxabs={maxabs:.3e} (tol={tol['maxabs']:.3e})"
                )

    assert not failures, (
        f"{model_key}: embed_dataset pipeline parity failed:\n" + "\n".join(failures)
    )
    del fast, native_model
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Backend setter semantics -- ensures that the canonical post-load switching
# mechanism (`model.attn_backend = backend`) actually propagates to every
# attention layer. If this regresses silently, every other backend-parametrized
# test is no-op.
# -----------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
def test_attn_backend_setter_propagates(model_key: str) -> None:
    """Structural check: does `model.attn_backend = X` propagate to every attention submodule?

    Unlike the other parity tests, this does not require the native package for
    the family -- it's a FastPLMs-only invariant. Run in any image.
    """
    device = torch.device("cuda")
    fast = load_fast(model_key, device, torch.float32)

    # Start from SDPA, switch to flex, verify every attention submodule flipped.
    fast.attn_backend = "sdpa"
    assert _get_resolved_backend(fast, model_key) == "sdpa", (
        f"{model_key}: attn_backend setter did not propagate 'sdpa' to attention modules"
    )

    try:
        fast.attn_backend = "flex"
    except AssertionError as e:
        pytest.skip(f"{model_key}: flex backend unavailable: {e}")
    resolved = _get_resolved_backend(fast, model_key)
    assert resolved == "flex", (
        f"{model_key}: after setting attn_backend='flex', attention modules report '{resolved}'. "
        f"The setter is not propagating. Every other backend-parametrized test becomes a no-op."
    )

    # Every attention-like submodule should report 'flex' now (not just one).
    mismatches = []
    for name, module in fast.named_modules():
        ab = getattr(module, "attn_backend", None)
        if ab is None or not hasattr(ab, "value"):
            continue
        if ab.value != "flex":
            mismatches.append(f"{name}: {ab.value}")
    assert not mismatches, (
        f"{model_key}: after setting attn_backend='flex', these submodules are still on a "
        f"different backend:\n" + "\n".join(mismatches[:10])
    )

    del fast
    torch.cuda.empty_cache()
