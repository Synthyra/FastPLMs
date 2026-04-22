"""Diagnostic for backend-consistency regression.

Compares sdpa vs kernels_flash vs flex on ESM2-8M in bf16.
Measures per-position cosine, pooled cosine, maxabs, and argmax agreement.
"""
from __future__ import annotations

import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM

from testing.conftest import CANONICAL_AAS, SEED


def gen_sequences(seed: int = SEED):
    rng = random.Random(seed)
    lengths = [16, 32, 48, 64, 80, 96, 112, 128]
    return ["M" + "".join(rng.choices(CANONICAL_AAS, k=L - 1)) for L in lengths]


def run(model, sequences, device):
    tok = model.tokenizer
    enc = tok(sequences, return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc)
    return out.last_hidden_state, out.logits, enc["attention_mask"]


def main() -> int:
    device = torch.device("cuda")
    sequences = gen_sequences()

    print("Loading ESM2-8M in bf16 with sdpa ...")
    m_sdpa = AutoModelForMaskedLM.from_pretrained(
        "Synthyra/ESM2-8M", trust_remote_code=True,
        dtype=torch.bfloat16, device_map=device,
    ).eval()
    m_sdpa.attn_backend = "sdpa"
    sdpa_last, sdpa_logits, mask = run(m_sdpa, sequences, device)
    del m_sdpa
    torch.cuda.empty_cache()

    for backend in ("flex", "kernels_flash"):
        print(f"\nLoading ESM2-8M in bf16 with {backend} ...")
        m_alt = AutoModelForMaskedLM.from_pretrained(
            "Synthyra/ESM2-8M", trust_remote_code=True,
            dtype=torch.bfloat16, device_map=device,
        ).eval()
        try:
            m_alt.attn_backend = backend
        except (AssertionError, RuntimeError) as e:
            print(f"  SKIP {backend}: {e}")
            del m_alt
            torch.cuda.empty_cache()
            continue
        alt_last, alt_logits, _ = run(m_alt, sequences, device)

        # Per-position cosine vs sdpa.
        mask_b = mask.bool()
        sdpa_valid = sdpa_last[mask_b].float()
        alt_valid = alt_last[mask_b].float()
        per_pos_cos = F.cosine_similarity(sdpa_valid, alt_valid, dim=-1)
        print(f"  per-position cosine:  min={per_pos_cos.min().item():.4f}  mean={per_pos_cos.mean().item():.4f}  max={per_pos_cos.max().item():.4f}")

        # Pooled cosine per sequence.
        m = mask.bool().unsqueeze(-1).float()
        sdpa_pooled = (sdpa_last.float() * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        alt_pooled = (alt_last.float() * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        pooled_cos = F.cosine_similarity(sdpa_pooled, alt_pooled, dim=-1)
        print(f"  per-seq pooled cosine:  {[f'{c.item():.4f}' for c in pooled_cos]}")

        # Raw diffs.
        diff = (alt_last.float() - sdpa_last.float())[mask_b]
        print(f"  raw diff:  mse={(diff ** 2).mean().item():.3e}  maxabs={diff.abs().max().item():.3e}")
        print(f"  sdpa magnitudes:  mean_norm={sdpa_valid.norm(dim=-1).mean().item():.3f}  maxabs={sdpa_valid.abs().max().item():.3e}")
        print(f"  alt magnitudes:   mean_norm={alt_valid.norm(dim=-1).mean().item():.3f}  maxabs={alt_valid.abs().max().item():.3e}")
        # NaN / Inf check.
        print(f"  alt has NaN={torch.isnan(alt_last).any().item()}  Inf={torch.isinf(alt_last).any().item()}")

        # Logit argmax agreement.
        sdpa_argmax = sdpa_logits.float()[mask_b].argmax(dim=-1)
        alt_argmax = alt_logits.float()[mask_b].argmax(dim=-1)
        agreement = (sdpa_argmax == alt_argmax).float().mean().item()
        print(f"  argmax agreement: {agreement:.4f}")

        # First divergent position.
        nonmatching = (sdpa_argmax != alt_argmax).nonzero(as_tuple=False).flatten()
        if len(nonmatching) > 0:
            print(f"  first {min(5, len(nonmatching))} mismatched positions: {nonmatching[:5].tolist()}")

        del m_alt, alt_last, alt_logits
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
