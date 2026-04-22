"""Pinpoint the root cause of FastPLMs vs native ESMC per-layer drift.

Hypotheses to test, each with its own column:
  A. Baseline: FastPLMs sdpa + default PyTorch kernel dispatch.
  B. Force MATH kernel on FastPLMs (bit-deterministic reference): if native
     also uses MATH, A-vs-native will match B-vs-native only if dispatch is
     identical. If B matches native much better than A, dispatch differs.
  C. Force MATH kernel on BOTH fast and native: if this makes per-layer
     near-zero, the drift is pure SDPA-kernel-dispatch numerics (no bug).
  D. Force MATH kernel on fast, no mask at all: rules out mask-shape bias.

Run:
    docker run --gpus all --ipc=host --rm -v $(pwd):/workspace \
        fastplms-esm_plusplus python /workspace/testing/parity_debug_kernel.py
"""
from __future__ import annotations

import random
from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from testing.conftest import CANONICAL_AAS, SEED
from testing.official.esm_plusplus import load_official_model as load_native_esmc


def per_layer_mse(fh, nh):
    return [((fh[i].float() - nh[i].float()) ** 2).mean().item() for i in range(min(len(fh), len(nh)))]


def run_pair(fast, native, enc, fast_backend, native_backend, label):
    fast_ctx = sdpa_kernel(fast_backend) if fast_backend is not None else nullcontext()
    native_ctx = sdpa_kernel(native_backend) if native_backend is not None else nullcontext()
    try:
        with torch.no_grad():
            with fast_ctx:
                fo = fast(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    output_hidden_states=True,
                )
            with native_ctx:
                no = native(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        mses = per_layer_mse(fo.hidden_states, no.hidden_states)
        print(f"  {label:50s} L0={mses[0]:.2e}  L1={mses[1]:.2e}  L15={mses[15]:.2e}  L29={mses[29]:.2e}  L30={mses[30]:.2e}")
        return mses
    except Exception as e:
        print(f"  {label:50s} skipped: {type(e).__name__}: {str(e)[:80]}")
        return None


def main() -> int:
    device = torch.device("cuda")
    random.seed(SEED)
    torch.manual_seed(SEED)

    seq = "M" + "".join(random.choices(CANONICAL_AAS, k=63))

    from transformers import AutoModelForMaskedLM
    fast = AutoModelForMaskedLM.from_pretrained(
        "Synthyra/ESMplusplus_small", trust_remote_code=True,
        dtype=torch.float32, device_map=device,
    ).eval()

    native, _ = load_native_esmc(reference_repo_id="esmc-300", device=device, dtype=torch.float32)

    enc = fast.tokenizer([seq], return_tensors="pt", padding=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    enc_no_mask = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    print("Per-layer MSE between FastPLMs and native ESMC at selected layers (fp32, no padding):\n")

    run_pair(fast, native, enc, None,                          None,                          "A. default / default")
    run_pair(fast, native, enc, SDPBackend.MATH,                None,                          "B. fast=MATH / native=default")
    run_pair(fast, native, enc, None,                           SDPBackend.MATH,               "C. fast=default / native=MATH")
    run_pair(fast, native, enc, SDPBackend.MATH,                SDPBackend.MATH,               "D. both=MATH (deterministic)")
    run_pair(fast, native, enc, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, "E. both=EFFICIENT")
    run_pair(fast, native, enc, SDPBackend.FLASH_ATTENTION,     SDPBackend.FLASH_ATTENTION,     "F. both=FLASH")

    print("\nWhich kernels get dispatched by default?")

    def probe_fast():
        x = fast.embed(enc["input_ids"])
        block = fast.transformer.blocks[0]
        q = torch.randn(1, 15, enc["input_ids"].shape[1], 64, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        attn_mask = enc["attention_mask"][:, None, None, :].bool()
        for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION):
            try:
                with sdpa_kernel(backend):
                    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
                print(f"    FastPLMs mask=(B,1,1,L) attn_mask=bool:    {backend.name}: OK")
            except Exception as e:
                print(f"    FastPLMs mask=(B,1,1,L) attn_mask=bool:    {backend.name}: {type(e).__name__}: {str(e)[:80]}")
        attn_mask2 = enc["attention_mask"][:, None, :, None].bool() & enc["attention_mask"][:, None, None, :].bool()
        for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION):
            try:
                with sdpa_kernel(backend):
                    _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask2)
                print(f"    native mask=(B,1,L,L) attn_mask=bool:      {backend.name}: OK")
            except Exception as e:
                print(f"    native mask=(B,1,L,L) attn_mask=bool:      {backend.name}: {type(e).__name__}: {str(e)[:80]}")

    probe_fast()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
