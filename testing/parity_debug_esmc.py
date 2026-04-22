"""
Standalone ESMC parity investigation — NOT a pytest test.

Run inside the fastplms Docker image:
    python /app/testing/parity_debug_esmc.py

Goal: surface where FastPLMs ESMC diverges from native EvolutionaryScale ESMC.

What this script checks, in order, printing a clear PASS/FAIL per check:
  1. Tokenizer vocab parity (size, every token mapping, special token IDs).
  2. Tokenization of fixed sequences produces identical input_ids.
  3. State dict parity in fp32 (per-parameter MSE, no aggregate).
  4. Forward parity in fp32 per layer, SDPA backend, with/without padding.
  5. Forward parity in bf16 per layer, SDPA backend.
  6. Forward parity across attention backends {sdpa, kernels_flash, flex} (fp32).
  7. `embed_dataset()` pipeline output vs a manual native-wrapped pipeline.

The point is: if everything prints PASS and downstream parity is still off on a
real task, the issue is not in the encoder — look at how embeddings are consumed.
If anything prints FAIL, the message names which layer / which parameter / which
input diverged and by how much.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

from testing.conftest import CANONICAL_AAS, SEED
from testing.official.esm_plusplus import load_official_model as load_native_esmc


FAST_PATH = "Synthyra/ESMplusplus_small"
NATIVE_REF = "esmc-300"

PAD_TOK = "<pad>"

FP32_PARAM_MSE_TOL = 0.0
FP32_HIDDEN_MSE_TOL = 5e-8
FP32_HIDDEN_MAXABS_TOL = 5e-4
BF16_HIDDEN_MSE_TOL = 1e-3
BF16_HIDDEN_MAXABS_TOL = 5e-2
BACKEND_MSE_TOL = 1e-5
BACKEND_MAXABS_TOL = 5e-3


def banner(msg: str) -> None:
    print("=" * 78)
    print(msg)
    print("=" * 78)


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}{(': ' + detail) if detail else ''}")
    return ok


def gen_sequences(seed: int, n: int, lengths: List[int]) -> List[str]:
    rng = random.Random(seed)
    assert len(lengths) == n, f"{len(lengths)} != {n}"
    out: List[str] = []
    for L in lengths:
        out.append("M" + "".join(rng.choices(CANONICAL_AAS, k=L - 1)))
    return out


@dataclass
class ForwardOutputs:
    last_hidden_state: torch.Tensor
    hidden_states: Tuple[torch.Tensor, ...]
    logits: Optional[torch.Tensor]


def load_fast(dtype: torch.dtype, device: torch.device, attn_backend: str = "sdpa") -> nn.Module:
    from fastplms.esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig
    from transformers import AutoModelForMaskedLM

    ESMplusplusConfig.attn_backend = attn_backend
    model = AutoModelForMaskedLM.from_pretrained(
        FAST_PATH,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device,
    ).eval()
    if hasattr(model, "transformer"):
        trans = model.transformer
    elif hasattr(model, "esm") and hasattr(model.esm, "transformer"):
        trans = model.esm.transformer
    else:
        trans = None
    if trans is not None and hasattr(trans, "set_attention_backend"):
        trans.set_attention_backend(attn_backend)
    return model


def fast_forward(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ForwardOutputs:
    sequence_id = attention_mask.to(dtype=torch.bool)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        sequence_id=sequence_id,
        output_hidden_states=True,
    )
    last = out.last_hidden_state if out.last_hidden_state is not None else out.hidden_states[-1]
    return ForwardOutputs(
        last_hidden_state=last,
        hidden_states=tuple(out.hidden_states),
        logits=getattr(out, "logits", None),
    )


def native_forward(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> ForwardOutputs:
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return ForwardOutputs(
        last_hidden_state=out.last_hidden_state,
        hidden_states=tuple(out.hidden_states),
        logits=out.logits,
    )


def hidden_mse_maxabs(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    a_valid = a[mask]
    b_valid = b[mask]
    mse = mse_loss(a_valid.float(), b_valid.float()).item()
    maxabs = (a_valid.float() - b_valid.float()).abs().max().item()
    return mse, maxabs


def check_tokenizer_parity(fast_model: nn.Module, native_tokenizer) -> bool:
    banner("1. Tokenizer vocab parity")
    ft = fast_model.tokenizer
    all_pass = True

    fast_vocab = ft.get_vocab()
    native_vocab = native_tokenizer.get_vocab()
    all_pass &= check(
        "vocab size equal",
        len(fast_vocab) == len(native_vocab),
        f"fast={len(fast_vocab)} native={len(native_vocab)}",
    )

    for tok, nid in sorted(native_vocab.items(), key=lambda kv: kv[1]):
        fid = fast_vocab.get(tok, None)
        if fid != nid:
            check(f"token {tok!r} id match", False, f"native={nid} fast={fid}")
            all_pass = False
    if all_pass:
        check("every token id matches", True, f"{len(native_vocab)} tokens")

    for name in ("pad_token_id", "cls_token_id", "eos_token_id", "mask_token_id", "unk_token_id", "bos_token_id"):
        fast_v = getattr(ft, name, None)
        native_v = getattr(native_tokenizer, name, None)
        all_pass &= check(f"{name} match", fast_v == native_v, f"fast={fast_v} native={native_v}")
    return all_pass


def tokenize_fast(tokenizer, sequences: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
    enc = tokenizer(sequences, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in enc.items()}


def check_tokenization(fast_model: nn.Module, native_tokenizer, sequences: List[str], device: torch.device) -> Tuple[bool, Dict[str, torch.Tensor]]:
    banner("2. Tokenization produces identical input_ids")
    fast_enc = tokenize_fast(fast_model.tokenizer, sequences, device)
    native_enc = tokenize_fast(native_tokenizer, sequences, device)
    ok_ids = torch.equal(fast_enc["input_ids"], native_enc["input_ids"])
    ok_mask = torch.equal(fast_enc["attention_mask"], native_enc["attention_mask"])
    ok = check("input_ids exact", ok_ids, f"shape={tuple(fast_enc['input_ids'].shape)}")
    ok &= check("attention_mask exact", ok_mask)
    if not ok_ids:
        diff = (fast_enc["input_ids"] != native_enc["input_ids"]).nonzero(as_tuple=False)
        print(f"    first 5 mismatches: {diff[:5].tolist()}")
    return ok, fast_enc


def check_weight_parity(fast_sd: Dict[str, torch.Tensor], native_sd: Dict[str, torch.Tensor]) -> bool:
    banner("3. State dict parity (fp32, per-parameter)")
    all_pass = True

    fast_keys = set(fast_sd.keys())
    native_keys = set(native_sd.keys())
    if fast_keys != native_keys:
        only_fast = sorted(fast_keys - native_keys)
        only_native = sorted(native_keys - fast_keys)
        check("key sets match", False, f"only_fast={only_fast[:3]} only_native={only_native[:3]}")
        all_pass = False
    else:
        check("key sets match", True, f"{len(fast_keys)} parameters")

    failed: List[str] = []
    for name in sorted(fast_keys & native_keys):
        a = fast_sd[name].float()
        b = native_sd[name].float()
        if a.shape != b.shape:
            failed.append(f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
            continue
        diff = (a - b).abs().max().item()
        if diff > FP32_PARAM_MSE_TOL:
            mse = mse_loss(a, b).item()
            failed.append(f"{name}: max|Δ|={diff:.3e} mse={mse:.3e}")
    all_pass &= check(
        f"every parameter matches (tol {FP32_PARAM_MSE_TOL})",
        not failed,
        f"{len(failed)} divergent; first: {failed[:3]}",
    )
    return all_pass


def check_forward_parity(
    fast_model: nn.Module,
    native_model: nn.Module,
    enc: Dict[str, torch.Tensor],
    dtype: torch.dtype,
    mse_tol: float,
    maxabs_tol: float,
    tag: str,
) -> bool:
    banner(f"{tag}")
    mask = enc["attention_mask"].bool()

    with torch.inference_mode():
        fo = fast_forward(fast_model, enc["input_ids"], enc["attention_mask"])
        no = native_forward(native_model, enc["input_ids"], enc["attention_mask"])

    all_pass = True
    all_pass &= check(
        "hidden_states tuple length matches",
        len(fo.hidden_states) == len(no.hidden_states),
        f"fast={len(fo.hidden_states)} native={len(no.hidden_states)}",
    )

    n = min(len(fo.hidden_states), len(no.hidden_states))
    failures: List[str] = []
    for i in range(n):
        mse, maxabs = hidden_mse_maxabs(fo.hidden_states[i], no.hidden_states[i], mask)
        if mse > mse_tol or maxabs > maxabs_tol:
            failures.append(f"layer {i}: mse={mse:.3e} maxabs={maxabs:.3e}")
        print(f"    layer {i:2d}: mse={mse:.3e} maxabs={maxabs:.3e}")
    all_pass &= check(
        f"every layer within tol (mse<{mse_tol}, maxabs<{maxabs_tol})",
        not failures,
        f"{len(failures)} divergent layers",
    )

    mse, maxabs = hidden_mse_maxabs(fo.last_hidden_state, no.last_hidden_state, mask)
    all_pass &= check(
        "last_hidden_state within tol",
        mse <= mse_tol and maxabs <= maxabs_tol,
        f"mse={mse:.3e} maxabs={maxabs:.3e}",
    )

    if fo.logits is not None and no.logits is not None:
        mse, maxabs = hidden_mse_maxabs(fo.logits, no.logits, mask)
        print(f"    logits: mse={mse:.3e} maxabs={maxabs:.3e}")
    return all_pass


def check_backend_consistency(
    fast_model_sdpa: nn.Module,
    enc: Dict[str, torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    banner(f"6. Attention backend consistency (dtype={dtype})")
    mask = enc["attention_mask"].bool()

    with torch.inference_mode():
        fo_sdpa = fast_forward(fast_model_sdpa, enc["input_ids"], enc["attention_mask"])

    all_pass = True
    for backend in ("kernels_flash", "flex"):
        try:
            fm_alt = load_fast(dtype=dtype, device=device, attn_backend=backend)
            with torch.inference_mode():
                fo_alt = fast_forward(fm_alt, enc["input_ids"], enc["attention_mask"])
        except Exception as e:
            check(f"backend {backend} loadable+runnable", False, str(e)[:120])
            all_pass = False
            continue

        mse, maxabs = hidden_mse_maxabs(fo_alt.last_hidden_state, fo_sdpa.last_hidden_state, mask)
        all_pass &= check(
            f"backend {backend} ≈ sdpa (last_hidden_state)",
            mse <= BACKEND_MSE_TOL and maxabs <= BACKEND_MAXABS_TOL,
            f"mse={mse:.3e} maxabs={maxabs:.3e}",
        )
        del fm_alt
        torch.cuda.empty_cache()
    return all_pass


def check_embed_dataset_pipeline(
    fast_model: nn.Module,
    native_model: nn.Module,
    sequences: List[str],
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    banner(f"7. embed_dataset() pipeline vs manual native (dtype={dtype})")
    fast_embeddings = fast_model.embed_dataset(
        sequences=sequences,
        tokenizer=fast_model.tokenizer,
        batch_size=4,
        max_len=max(len(s) for s in sequences) + 2,
        truncate=True,
        full_embeddings=False,
        embed_dtype=torch.float32,
        pooling_types=["mean"],
        num_workers=0,
        sql=False,
        save=False,
        padding="longest",
    )
    assert fast_embeddings is not None

    native_tokenizer = native_model.tokenizer
    native_embeddings: Dict[str, torch.Tensor] = {}
    for seq in sequences:
        enc = tokenize_fast(native_tokenizer, [seq], device)
        with torch.inference_mode():
            out = native_forward(native_model, enc["input_ids"], enc["attention_mask"])
        m = enc["attention_mask"].bool().unsqueeze(-1).float()
        pooled = (out.last_hidden_state.float() * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        native_embeddings[seq] = pooled.squeeze(0).cpu()

    failures: List[str] = []
    for seq in sequences:
        f = fast_embeddings[seq].cpu().float()
        n = native_embeddings[seq].cpu().float()
        if f.shape != n.shape:
            failures.append(f"seq_len={len(seq)}: shape {tuple(f.shape)} vs {tuple(n.shape)}")
            continue
        diff = (f - n).abs().max().item()
        mse = mse_loss(f, n).item()
        print(f"    seq_len={len(seq):3d}: mse={mse:.3e} maxabs={diff:.3e}")
        if mse > BF16_HIDDEN_MSE_TOL or diff > BF16_HIDDEN_MAXABS_TOL:
            failures.append(f"seq_len={len(seq)}: mse={mse:.3e} maxabs={diff:.3e}")

    return check(
        f"mean-pool embedding parity (mse<{BF16_HIDDEN_MSE_TOL})",
        not failures,
        f"{len(failures)} divergent sequences",
    )


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "This script requires a CUDA GPU."
    random.seed(SEED)
    torch.manual_seed(SEED)

    lengths = [16, 32, 48, 64, 80, 96, 112, 128]
    sequences = gen_sequences(SEED, len(lengths), lengths)

    native_fp32, native_tok = load_native_esmc(
        reference_repo_id=NATIVE_REF, device=device, dtype=torch.float32,
    )
    fast_fp32 = load_fast(dtype=torch.float32, device=device, attn_backend="sdpa")

    overall: List[bool] = []

    overall.append(check_tokenizer_parity(fast_fp32, native_tok))
    ok, enc = check_tokenization(fast_fp32, native_tok, sequences, device)
    overall.append(ok)
    overall.append(check_weight_parity(fast_fp32.state_dict(), native_fp32.model.state_dict()))
    overall.append(check_forward_parity(
        fast_fp32, native_fp32, enc, torch.float32,
        mse_tol=FP32_HIDDEN_MSE_TOL, maxabs_tol=FP32_HIDDEN_MAXABS_TOL,
        tag="4. Forward parity (fp32, sdpa)",
    ))

    del fast_fp32, native_fp32
    torch.cuda.empty_cache()

    fast_bf16 = load_fast(dtype=torch.bfloat16, device=device, attn_backend="sdpa")
    native_bf16, _ = load_native_esmc(
        reference_repo_id=NATIVE_REF, device=device, dtype=torch.bfloat16,
    )
    overall.append(check_forward_parity(
        fast_bf16, native_bf16, enc, torch.bfloat16,
        mse_tol=BF16_HIDDEN_MSE_TOL, maxabs_tol=BF16_HIDDEN_MAXABS_TOL,
        tag="5. Forward parity (bf16, sdpa)",
    ))

    overall.append(check_backend_consistency(fast_bf16, enc, torch.bfloat16, device))

    overall.append(check_embed_dataset_pipeline(
        fast_bf16, native_bf16, sequences, torch.bfloat16, device,
    ))

    banner("SUMMARY")
    labels = [
        "1. tokenizer vocab",
        "2. tokenization ids",
        "3. weight parity (fp32)",
        "4. forward parity (fp32, sdpa)",
        "5. forward parity (bf16, sdpa)",
        "6. backend consistency",
        "7. embed_dataset pipeline",
    ]
    for lbl, ok in zip(labels, overall):
        print(f"  [{'PASS' if ok else 'FAIL'}] {lbl}")
    return 0 if all(overall) else 1


if __name__ == "__main__":
    sys.exit(main())
