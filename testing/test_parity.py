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
    "ankh": ParityTolerances(
        fp32_last_hidden_mse=1e-12, fp32_last_hidden_maxabs=1e-5, fp32_last_hidden_rel_maxabs=1e-5,
        fp32_logits_mse=1e-4,
        fp32_hidden_rel_std=1e-5, fp32_hidden_rel_maxabs=1e-5,
        # ANKH has no per-block norm (T5-style residual accumulation across 48+ blocks), so
        # bf16 rounding compounds into larger absolute values at the pre-norm residual stream.
        # We anchor the bf16 last_hidden_state check on the RELATIVE maxabs (diff / native
        # maxabs), not a loose absolute threshold, so a future regression that biases
        # activations can't hide behind the large native magnitudes.
        bf16_last_hidden_mse=5e-4, bf16_last_hidden_maxabs=2e-1, bf16_last_hidden_rel_maxabs=2e-2,
        bf16_logits_mse=5e-2, bf16_hidden_rel_std=5e-2, bf16_hidden_rel_maxabs=1e-1,
    ),
}

EXPECTED_WEIGHT_EXTRAS: Dict[str, set] = {
    "ankh": {"lm_head.weight"},
}


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
@pytest.mark.parametrize("model_key", [k for k in MODEL_REGISTRY if k != "dplm2"])
def test_weight_parity_fp32(model_key: str) -> None:
    device = torch.device("cuda")
    fast = load_fast(model_key, device, torch.float32)
    native_model, _ = try_load_native(model_key, device, torch.float32)

    fast_sd = fast.state_dict()
    native_sd = native_model.model.state_dict() if hasattr(native_model, "model") else native_model.state_dict()

    expected_fast_extras = EXPECTED_WEIGHT_EXTRAS.get(model_key, set())
    fast_keys = set(fast_sd.keys()) - expected_fast_extras
    native_keys = set(native_sd.keys())
    assert fast_keys == native_keys, (
        f"{model_key}: state_dict key sets differ (after allowing expected extras={expected_fast_extras}).\n"
        f"  only_fast: {sorted(fast_keys - native_keys)[:5]}\n"
        f"  only_native: {sorted(native_keys - fast_keys)[:5]}"
    )

    shape_mismatches: List[str] = []
    value_mismatches: List[str] = []
    for name in sorted(fast_keys & native_keys):
        f = fast_sd[name]
        n = native_sd[name]
        if f.shape != n.shape:
            shape_mismatches.append(f"{name}: {tuple(f.shape)} vs {tuple(n.shape)}")
            continue
        if not torch.equal(f.float(), n.float()):
            max_abs = (f.float() - n.float()).abs().max().item()
            value_mismatches.append(f"{name}: max|Δ|={max_abs:.3e}")
    assert not shape_mismatches, f"{model_key}: shape mismatches:\n" + "\n".join(shape_mismatches[:10])
    assert not value_mismatches, f"{model_key}: value mismatches:\n" + "\n".join(value_mismatches[:10])

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

    # Skip logits parity when the fast model has an LM head that native doesn't
    # (e.g. Ankh: fast is ForMaskedLM with its own head; native T5EncoderModel has
    # no head and testing/official/ankh.py bolts on a fresh tied-weight head).
    has_head_mismatch = bool(EXPECTED_WEIGHT_EXTRAS.get(model_key))
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
@pytest.mark.parametrize("model_key", [k for k in MODEL_REGISTRY if k != "dplm2"])
def test_forward_parity_fp32(model_key: str, scenario: str) -> None:
    tol = FAMILY_TOLERANCES[model_key]
    _run_forward_parity(model_key, torch.float32, tol, "fp32", scenario=scenario)


@pytest.mark.gpu
@pytest.mark.parametrize("scenario", list(PADDING_SCENARIOS.keys()))
@pytest.mark.parametrize("model_key", [k for k in MODEL_REGISTRY if k != "dplm2"])
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
@pytest.mark.parametrize("model_key", [k for k in MODEL_REGISTRY if k != "dplm2" and MODEL_REGISTRY[k]["uses_tokenizer"]])
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
# -----------------------------------------------------------------------------

# ANKH's modeling code intentionally falls back from kernels_flash to flex
# (T5 relative position bias is incompatible with flash kernels). We therefore
# only ask ANKH to compare sdpa vs flex.
BACKEND_CONSISTENCY_MATRIX: Dict[str, Tuple[str, ...]] = {
    "esm2": ("kernels_flash", "flex"),
    "esmc": ("kernels_flash", "flex"),
    "e1": ("kernels_flash", "flex"),
    "dplm": ("kernels_flash", "flex"),
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


@pytest.mark.gpu
@pytest.mark.parametrize("model_key", list(BACKEND_CONSISTENCY_MATRIX.keys()))
def test_backend_consistency_fp32(model_key: str) -> None:
    """Every supported backend for this family must produce the same last_hidden_state
    as SDPA on fixed inputs (to documented tolerance).

    SDPA is the numerical ground truth (exact / memory-efficient). kernels_flash
    uses online softmax and tiling so its outputs differ at the 1e-3 level; flex
    is near-exact (within rounding). The tolerances below are set accordingly.
    """
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

    # Per-backend absolute and relative tolerances. kernels_flash is approximate
    # by design, so its tolerance is looser. Flex is near-exact in fp32.
    BACKEND_TOL: Dict[str, Dict[str, float]] = {
        "flex": {"mse": 1e-6, "maxabs": 5e-3, "rel_maxabs": 5e-3},
        "kernels_flash": {"mse": 1e-3, "maxabs": 1e-1, "rel_maxabs": 5e-2},
    }

    failures: List[str] = []
    for backend in BACKEND_CONSISTENCY_MATRIX[model_key]:
        alt = load_fast(model_key, device, torch.float32)
        resolved = _apply_backend(alt, backend, model_key)
        if resolved is None:
            del alt
            torch.cuda.empty_cache()
            pytest.skip(f"{model_key}: backend {backend} not available")
        if resolved != backend:
            failures.append(
                f"{backend}: requested backend was silently resolved to '{resolved}'. "
                f"If this is an intentional fallback (e.g. ANKH kernels_flash -> flex), "
                f"remove {backend!r} from BACKEND_CONSISTENCY_MATRIX[{model_key!r}]."
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
        tol = BACKEND_TOL[backend]
        if mse > tol["mse"] or maxabs > tol["maxabs"] or rel_maxabs > tol["rel_maxabs"]:
            failures.append(
                f"{backend}: mse={mse:.3e} (tol={tol['mse']:.3e}), "
                f"maxabs={maxabs:.3e} (tol={tol['maxabs']:.3e}), "
                f"rel_maxabs={rel_maxabs:.3e} (tol={tol['rel_maxabs']:.3e})"
            )
        del alt, alt_out
        torch.cuda.empty_cache()

    assert not failures, f"{model_key}: backend consistency failed:\n" + "\n".join(failures)


# -----------------------------------------------------------------------------
# embed_dataset pipeline parity (what downstream users actually call)
#
# This is the most user-facing test: if `embed_dataset(...)` and an equivalent
# hand-rolled native forward + mean-pool disagree, a downstream task will see
# the difference even though every per-layer metric was within tolerance.
# -----------------------------------------------------------------------------

# Per-family absolute/max-abs tolerances for the mean-pooled embedding.
EMBED_DATASET_TOL: Dict[str, Dict[str, float]] = {
    "esm2": {"mse": 5e-8, "maxabs": 5e-3},
    "esmc": {"mse": 5e-8, "maxabs": 5e-3},
    "dplm": {"mse": 5e-8, "maxabs": 5e-3},
    "e1":   {"mse": 5e-6, "maxabs": 2e-2},   # Grouped-query attention + block-causal GLOBAL layers propagate rounding
    # ANKH-base post-final-RMSNorm activations are modest (~O(1)) but the
    # mean-pool aggregates over the full sequence; 5e-3 maxabs is plenty tight.
    "ankh": {"mse": 5e-8, "maxabs": 5e-3},
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
