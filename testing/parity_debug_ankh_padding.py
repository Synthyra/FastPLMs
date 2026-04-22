"""Trace where ANKH padding-isolation breaks: single seq alone vs batched padded.

The forward parity tests pass with no padding but fail when padding is introduced.
Compares hidden_states at every layer for the short sequence in two settings:
  - alone: forward([short]), output[0, :16]
  - padded: forward([short, long_]), output[0, :16]

If they differ, padding is bleeding into valid attention.
"""
from __future__ import annotations

import random

import torch

from testing.conftest import CANONICAL_AAS, SEED


def diff(label, a, b):
    a = a.float()
    b = b.float()
    d = (a - b)
    print(f"  {label:50s} mse={(d**2).mean().item():.3e} maxabs={d.abs().max().item():.3e}")


def main() -> int:
    device = torch.device("cuda")
    random.seed(SEED)
    rng = random.Random(SEED)
    short = "M" + "".join(rng.choices(CANONICAL_AAS, k=15))
    long_ = "M" + "".join(rng.choices(CANONICAL_AAS, k=127))
    print(f"short len={len(short)} long len={len(long_)}")

    from transformers import AutoModelForMaskedLM
    fast = AutoModelForMaskedLM.from_pretrained(
        "Synthyra/ANKH_base", trust_remote_code=True,
        dtype=torch.float32, device_map=device,
    ).eval()

    tok = fast.tokenizer
    enc_alone = tok([short], return_tensors="pt", padding=True)
    enc_alone = {k: v.to(device) for k, v in enc_alone.items()}
    enc_padded = tok([short, long_], return_tensors="pt", padding=True)
    enc_padded = {k: v.to(device) for k, v in enc_padded.items()}
    print(f"alone: input_ids shape={tuple(enc_alone['input_ids'].shape)}, mask sum={enc_alone['attention_mask'].sum().item()}")
    print(f"padded: input_ids shape={tuple(enc_padded['input_ids'].shape)}, mask sum row0={enc_padded['attention_mask'][0].sum().item()}")

    valid_len = int(enc_alone["attention_mask"].sum().item())
    print(f"valid_len={valid_len}\n")

    with torch.no_grad():
        out_alone = fast(input_ids=enc_alone["input_ids"], attention_mask=enc_alone["attention_mask"], output_hidden_states=True)
        out_padded = fast(input_ids=enc_padded["input_ids"], attention_mask=enc_padded["attention_mask"], output_hidden_states=True)

    print("=== Per-layer hidden_states diff: alone[0, :v] vs padded[0, :v] ===")
    for i in range(len(out_alone.hidden_states)):
        ha = out_alone.hidden_states[i][0, :valid_len]
        hp = out_padded.hidden_states[i][0, :valid_len]
        diff(f"hidden_states[{i:02d}]", ha, hp)

    print("\n=== Now repeat with manual-attn path to rule out SDPA -inf issues ===")
    # Patch in _manual_attn for layer 0 to inspect raw attention scores
    from fastplms.ankh.modeling_ankh import AnkhSelfAttention, AttentionBackend
    layer0_attn = fast.encoder.block[0].layer[0].SelfAttention

    captured = {}
    orig_compute_bias = layer0_attn.compute_bias

    def cb_hook(*args, **kw):
        out = orig_compute_bias(*args, **kw)
        captured.setdefault("bias", []).append(out.clone())
        return out

    layer0_attn.compute_bias = cb_hook  # type: ignore

    with torch.no_grad():
        _ = fast(input_ids=enc_alone["input_ids"], attention_mask=enc_alone["attention_mask"], output_hidden_states=False)
        _ = fast(input_ids=enc_padded["input_ids"], attention_mask=enc_padded["attention_mask"], output_hidden_states=False)
    layer0_attn.compute_bias = orig_compute_bias  # type: ignore

    bias_alone, bias_padded = captured["bias"]
    print(f"  bias_alone shape  = {tuple(bias_alone.shape)}")
    print(f"  bias_padded shape = {tuple(bias_padded.shape)}")
    if bias_padded.shape[-1] >= valid_len and bias_padded.shape[-2] >= valid_len:
        diff("bias[upper_left v x v] alone vs padded", bias_alone[:, :, :valid_len, :valid_len], bias_padded[:, :, :valid_len, :valid_len])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
